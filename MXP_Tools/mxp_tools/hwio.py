"""HW I/O — Verilog $writememh / $readmemh bridge.

Input  : SW emits .mem files that the FPGA's $readmemh consumes (quantized
         A/B, scales, bit-serial patterns). Format mirrors what HW expects.
Output : HW produces .mem files via $writememh. Each line is one 32-bit hex
         word; we treat it as an IEEE-754 FP32 bit pattern. `@<addr>` header
         lines set the destination address of the next payload word (so
         sparse / non-zero-origin dumps stay aligned to the matrix), and
         `//` comments are skipped.

Bank layout is plug-in: user supplies a callable
    mapping(bank_idx, word_offset, M, N) -> (m, n) or None
to tell us which C[m, n] each (bank, offset) corresponds to. A `None` return
means "ignored / padding". Two defaults are provided for the simplest cases.

All HW input/output files are ASCII hex with LF line endings, regardless of
host OS — Verilog tools tolerate CRLF but pinning LF keeps diffs stable.
"""
import numpy as np

from .const import BLOCK_SIZE


_HEX_OPEN_KWARGS = dict(encoding="ascii", newline="\n")


# ─────────────────────────── readers ───────────────────────────────────────

def _iter_writememh_words(path):
    """Yield (addr, word) for each payload in a $writememh dump.

    Honors `@<hex addr>` markers: the next payload word lands at that addr,
    and subsequent words increment by 1. `//` comments and blank lines are
    skipped. Multiple whitespace-separated words on one line are flattened.
    """
    addr = 0
    with open(path, "r", **_HEX_OPEN_KWARGS) as f:
        for raw in f:
            line = raw.split("//", 1)[0].strip()
            if not line:
                continue
            if line.startswith("@"):
                addr = int(line[1:], 16)
                continue
            for tok in line.split():
                yield addr, int(tok, 16)
                addr += 1


def read_writememh_fp32(path):
    """Parse a $writememh dump → flat FP32 array, address-aware.

    If the dump starts at addr 0 with no `@` jumps, this matches the old
    sequential semantics. If there are gaps or non-zero starts, the unwritten
    slots are NaN so gather_banks can detect them. Output length is
    `max(addr)+1` rounded up implicitly by the highest seen address.

    Host endianness note (mirror of `fp32_to_hex_words`): `int(tok, 16)` reads
    the logical 32-bit value and `np.uint32.view(np.float32)` reinterprets the
    host-endian bytes. On LE hosts (x86/ARM-LE) the bit pattern matches the
    writer's hex. On BE hosts this would invert — guard if you ever port.
    """
    items = list(_iter_writememh_words(path))
    if not items:
        return np.empty(0, dtype=np.float32)
    n_words = max(a for a, _ in items) + 1
    buf = np.full(n_words, np.uint32(0), dtype=np.uint32)
    written = np.zeros(n_words, dtype=bool)
    for a, w in items:
        buf[a] = np.uint32(w)
        written[a] = True
    out = buf.view(np.float32).copy()
    out[~written] = np.nan
    return out


def read_writememh_int32(path):
    """Same as read_writememh_fp32 but interpret each word as raw int32.

    Unwritten slots are 0 (since int32 has no NaN). Callers needing
    gap-detection should use read_writememh_fp32 instead.
    """
    items = list(_iter_writememh_words(path))
    if not items:
        return np.empty(0, dtype=np.int32)
    n_words = max(a for a, _ in items) + 1
    buf = np.zeros(n_words, dtype=np.uint32)
    for a, w in items:
        buf[a] = np.uint32(w)
    return buf.view(np.int32).copy()


# ─────────────────────────── writers ───────────────────────────────────────

def write_writememh(path, words_hex, addr_every=0):
    """Write a $readmemh-compatible .mem file.

    Args:
      words_hex : iterable of `str` (already-formatted hex, no 0x prefix)
                  or iterable of `int` (will be %08x-formatted as 32-bit).
      addr_every: if >0, emit `@<addr>` headers every N lines for readability.
    """
    with open(path, "w", **_HEX_OPEN_KWARGS) as f:
        for i, w in enumerate(words_hex):
            if addr_every and i % addr_every == 0:
                f.write(f"@{i:08x}\n")
            if isinstance(w, str):
                f.write(w.lower() + "\n")
            else:
                f.write(f"{int(w) & 0xFFFFFFFF:08x}\n")


def fp32_to_hex_words(arr_fp32):
    """FP32 ndarray → list of 8-char hex strings (IEEE-754 bit pattern).

    Host endianness note: x86/ARM-LE store FP32 in LE, but `np.uint32.view`
    reinterprets the same bytes under host endian. `int(tok, 16)` reads the
    logical 32-bit value, so the resulting bit pattern matches the hex word
    on LE hosts. On BE hosts this would invert — guard if you ever port.
    """
    flat = np.ascontiguousarray(arr_fp32, dtype=np.float32).ravel()
    u32 = flat.view(np.uint32)
    return [f"{int(w):08x}" for w in u32]


# ─────────────────────────── bank gather ───────────────────────────────────

def default_single_bank_row_major(bank_idx, word_offset, M, N):
    """Trivial layout: all of C lives in bank 0, row-major flat."""
    if bank_idx != 0:
        return None
    if word_offset >= M * N:
        return None
    return divmod(word_offset, N)


def default_banks_split_rows(n_banks):
    """Build a layout where M rows are split evenly across `n_banks`, row-major."""
    def mapping(bank_idx, word_offset, M, N):
        rows_per_bank = (M + n_banks - 1) // n_banks
        m0 = bank_idx * rows_per_bank
        if m0 >= M:
            return None
        m = m0 + word_offset // N
        n = word_offset % N
        if m >= min(M, m0 + rows_per_bank):
            return None
        return m, n
    return mapping


def interleaved_row_major_16bank(bank_idx, word_offset, M, N):
    """C[m,n] = SRAM[bank=flat%16, word=flat//16] where flat = m*N+n.

    32 col GEMM 결과를 row-major flat 으로 SRAM 에 쌓고
    sram_1rw_banked 의 INTERLEAVED 매핑 (LSB 4비트가 bank_sel) 으로
    16 bank 에 분산되는 layout 의 역매핑.

    Used by: gemm_sram integration testbench
    Spec   : docs/superpowers/specs/2026-05-14-integration-design.md §3
    """
    flat = word_offset * 16 + bank_idx
    if flat >= M * N:
        return None
    return divmod(flat, N)


def interleaved_row_major_32bank(bank_idx, word_offset, M, N):
    """C[m,n] = SRAM[bank=flat%32, word=flat//32] where flat = m*N+n.

    col-parallel RMW (32 instance) 의 매핑. col j 의 RMW 가 자기 bank j
    만 건드림 — 128 % 32 = 0 이므로 모든 mode (A8/A4/A2) 에서 충돌 0.

    Used by: gemm_sram integration testbench (32-RMW phase)
    Spec   : docs/superpowers/specs/2026-05-15-rmw-32x-design.md §4
    """
    flat = word_offset * 32 + bank_idx
    if flat >= M * N:
        return None
    return divmod(flat, N)


def gather_banks(bank_paths, M, N, mapping, reader=read_writememh_fp32):
    """Reconstruct C_hw (M, N) from per-bank .mem files via `mapping`.

    `mapping(bank_idx, word_offset, M, N) -> (m, n) | None`. Words that map
    to None are silently dropped (padding / unused slots).

    Coverage tracking uses a separate boolean mask, NOT a NaN sentinel, so
    legitimate HW NaN outputs are preserved. Raises if any C[m, n] is not
    written (uncovered layout) or if a (m, n) is written more than once
    (buggy mapping).
    """
    C = np.zeros((M, N), dtype=np.float32)
    written = np.zeros((M, N), dtype=bool)
    for bank_idx, path in enumerate(bank_paths):
        words = reader(path)
        for offset, val in enumerate(words):
            mn = mapping(bank_idx, offset, M, N)
            if mn is None:
                continue
            m, n = mn
            if written[m, n]:
                raise ValueError(
                    f"gather_banks: duplicate write at C[{m}, {n}] from "
                    f"bank {bank_idx} offset {offset}. Mapping is not injective."
                )
            C[m, n] = val
            written[m, n] = True
    if not written.all():
        missing = int((~written).sum())
        raise ValueError(
            f"gather_banks: {missing}/{M*N} elements of C were not covered by "
            f"the bank mapping. Check `mapping` against your SRAM layout."
        )
    return C


# ─────────────────────────── bit-serial / bit-parallel emitters ────────────
# Lifted from MXP/TransformSerial.py for byte-for-byte parity with the
# existing HW input flow. Kept here so the "emit HW input" pipeline is
# self-contained in this package.
#
# All emitters require their matrix dimensions to be exact multiples of
# BLOCK_SIZE. Public callers must pad upstream (cli.cmd_emit does this).

def _require_block_multiple(name, shape):
    for dim_idx, dim in enumerate(shape):
        if dim % BLOCK_SIZE != 0:
            raise ValueError(
                f"{name}: axis {dim_idx} length {dim} is not a multiple of "
                f"{BLOCK_SIZE}. Pad before calling."
            )


def get_bit_slices(int8_row, K, prec):
    """Extract `prec` bit-serial slices (MSB first) from a 32-element int8 row."""
    patterns = np.array(int8_row, dtype=np.int8).view(np.uint8).flatten()
    slices = []
    for bit_pos in range(prec - 1, -1, -1):
        word_bits = 0
        for i in range(K):
            bit = int((patterns[i] >> bit_pos) & 1)
            word_bits |= (bit << i)
        slices.append(word_bits)
    return slices


def emit_a_input_bs(path, a_int, prec):
    """Bit-serial A input. Tile order (K_tile, M_tile), 32 rows × prec slices."""
    _require_block_multiple("emit_a_input_bs(a_int)", a_int.shape)
    M, K = a_int.shape
    M_t, K_t = M // BLOCK_SIZE, K // BLOCK_SIZE
    with open(path, "w", **_HEX_OPEN_KWARGS) as f:
        for k in range(K_t):
            for m in range(M_t):
                block = a_int[m*BLOCK_SIZE:(m+1)*BLOCK_SIZE, k*BLOCK_SIZE:(k+1)*BLOCK_SIZE]
                for row in block:
                    for val in get_bit_slices(row, BLOCK_SIZE, prec):
                        f.write(f"{val:08x}\n")


def emit_a_input_bp(path, a_int):
    """Bit-parallel A input. 256-bit (32 × 8-bit) per row, tile order (K, M)."""
    _require_block_multiple("emit_a_input_bp(a_int)", a_int.shape)
    M, K = a_int.shape
    M_t, K_t = M // BLOCK_SIZE, K // BLOCK_SIZE
    with open(path, "w", **_HEX_OPEN_KWARGS) as f:
        for k in range(K_t):
            for m in range(M_t):
                block = a_int[m*BLOCK_SIZE:(m+1)*BLOCK_SIZE, k*BLOCK_SIZE:(k+1)*BLOCK_SIZE]
                for row in block:
                    hex_str = "".join(f"{v.astype(np.uint8):02x}" for v in reversed(row))
                    f.write(f"{hex_str}\n")


def emit_b_input(path, b_int):
    """Column-packed B input. 256-bit per column, tile order (N, K)."""
    _require_block_multiple("emit_b_input(b_int)", b_int.shape)
    K, N = b_int.shape
    K_t, N_t = K // BLOCK_SIZE, N // BLOCK_SIZE
    with open(path, "w", **_HEX_OPEN_KWARGS) as f:
        for n in range(N_t):
            for k in range(K_t):
                block = b_int[k*BLOCK_SIZE:(k+1)*BLOCK_SIZE, n*BLOCK_SIZE:(n+1)*BLOCK_SIZE]
                for col_idx in range(BLOCK_SIZE):
                    col = block[:, col_idx]
                    hex_str = "".join(f"{v.astype(np.uint8):02x}" for v in reversed(col))
                    f.write(f"{hex_str}\n")


def emit_a_scale(path, a_scale):
    """E8M0 scales for A. Order matches emit_a_input_bs: K_tile → M_tile → row."""
    M_pad, K_t = a_scale.shape
    if M_pad % BLOCK_SIZE != 0:
        raise ValueError(
            f"emit_a_scale: a_scale rows {M_pad} not a multiple of {BLOCK_SIZE}"
        )
    M_t = M_pad // BLOCK_SIZE
    with open(path, "w", **_HEX_OPEN_KWARGS) as f:
        for k in range(K_t):
            for m in range(M_t):
                for row in range(BLOCK_SIZE):
                    f.write(f"{a_scale[m*BLOCK_SIZE+row, k]:02x}\n")


def emit_b_scale(path, b_scale):
    """E8M0 scales for B. Order matches emit_b_input: N_tile → K_tile → col."""
    K_t, N_pad = b_scale.shape
    if N_pad % BLOCK_SIZE != 0:
        raise ValueError(
            f"emit_b_scale: b_scale cols {N_pad} not a multiple of {BLOCK_SIZE}"
        )
    N_t = N_pad // BLOCK_SIZE
    with open(path, "w", **_HEX_OPEN_KWARGS) as f:
        for n in range(N_t):
            for k in range(K_t):
                for col in range(BLOCK_SIZE):
                    f.write(f"{b_scale[k, n*BLOCK_SIZE+col]:02x}\n")
