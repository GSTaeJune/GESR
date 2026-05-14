"""HW I/O — Verilog $writememh / $readmemh bridge.

Input  : SW emits .mem files that the FPGA's $readmemh consumes (quantized
         A/B, scales, bit-serial patterns). Format mirrors what HW expects.
Output : HW produces .mem files via $writememh. Each line is one 32-bit hex
         word; we treat it as an IEEE-754 FP32 bit pattern. `@<addr>` header
         lines and `//` comments are skipped.

Bank layout is plug-in: user supplies a callable
    mapping(bank_idx, word_offset, M, N) -> (m, n) or None
to tell us which C[m, n] each (bank, offset) corresponds to. A `None` return
means "ignored / padding". Two defaults are provided for the simplest cases.
"""
import os

import numpy as np

from .const import BLOCK_SIZE


# ─────────────────────────── readers ───────────────────────────────────────

def _strip_writememh_line(line):
    """Strip $writememh metadata. Returns None if line has no payload."""
    line = line.split("//", 1)[0].strip()       # drop // comments
    if not line:
        return None
    if line.startswith("@"):                     # @<hex addr>
        return None
    return line


def read_writememh_fp32(path):
    """Parse a Verilog $writememh dump as flat FP32 (one word per non-meta line).

    Each payload line is treated as an 8-char IEEE-754 32-bit hex word
    (uppercase/lowercase OK, may be space-separated for multi-word lines).
    """
    words = []
    with open(path, "r") as f:
        for raw in f:
            payload = _strip_writememh_line(raw)
            if payload is None:
                continue
            for tok in payload.split():
                words.append(int(tok, 16))
    arr_u32 = np.array(words, dtype=np.uint32)
    return arr_u32.view(np.float32).copy()


def read_writememh_int32(path):
    """Same as read_writememh_fp32 but interpret each word as raw int32.

    Handy for legacy TransformSerial.py-style c_*.hex outputs (HW PE-array's
    pre-dequant integer accumulator state).
    """
    words = []
    with open(path, "r") as f:
        for raw in f:
            payload = _strip_writememh_line(raw)
            if payload is None:
                continue
            for tok in payload.split():
                words.append(int(tok, 16))
    arr_u32 = np.array(words, dtype=np.uint32)
    return arr_u32.view(np.int32).copy()


# ─────────────────────────── writers ───────────────────────────────────────

def write_writememh(path, words_hex, addr_every=0):
    """Write a $readmemh-compatible .mem file.

    Args:
      words_hex : iterable of `str` (already-formatted hex, no 0x prefix)
                  or iterable of `int` (will be %08x-formatted as 32-bit).
      addr_every: if >0, emit `@<addr>` headers every N lines for readability.
    """
    with open(path, "w") as f:
        for i, w in enumerate(words_hex):
            if addr_every and i % addr_every == 0:
                f.write(f"@{i:08x}\n")
            if isinstance(w, str):
                f.write(w.lower() + "\n")
            else:
                f.write(f"{int(w) & 0xFFFFFFFF:08x}\n")


def fp32_to_hex_words(arr_fp32):
    """FP32 ndarray → list of 8-char hex strings (IEEE-754 bit pattern)."""
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


def gather_banks(bank_paths, M, N, mapping, reader=read_writememh_fp32):
    """Reconstruct C_hw (M, N) from per-bank .mem files via `mapping`.

    `mapping(bank_idx, word_offset, M, N) -> (m, n) | None`. Words that map
    to None are silently dropped (padding / unused slots).

    Returns float32 (M, N). Unwritten slots default to NaN so test failures
    show up loudly instead of as silent zeros.
    """
    C = np.full((M, N), np.nan, dtype=np.float32)
    for bank_idx, path in enumerate(bank_paths):
        words = reader(path)
        for offset, val in enumerate(words):
            mn = mapping(bank_idx, offset, M, N)
            if mn is None:
                continue
            m, n = mn
            C[m, n] = val
    if np.isnan(C).any():
        missing = int(np.isnan(C).sum())
        raise ValueError(
            f"gather_banks: {missing}/{M*N} elements of C were not covered by "
            f"the bank mapping. Check `mapping` against your SRAM layout."
        )
    return C


# ─────────────────────────── bit-serial / bit-parallel emitters ────────────
# Lifted from MXP/TransformSerial.py for byte-for-byte parity with the
# existing HW input flow. Kept here so the "emit HW input" pipeline is
# self-contained in this package.

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
    M, K = a_int.shape
    M_t, K_t = M // BLOCK_SIZE, K // BLOCK_SIZE
    with open(path, "w") as f:
        for k in range(K_t):
            for m in range(M_t):
                block = a_int[m*BLOCK_SIZE:(m+1)*BLOCK_SIZE, k*BLOCK_SIZE:(k+1)*BLOCK_SIZE]
                for row in block:
                    for val in get_bit_slices(row, BLOCK_SIZE, prec):
                        f.write(f"{val:08x}\n")


def emit_a_input_bp(path, a_int):
    """Bit-parallel A input. 256-bit (32 × 8-bit) per row, tile order (K, M)."""
    M, K = a_int.shape
    M_t, K_t = M // BLOCK_SIZE, K // BLOCK_SIZE
    with open(path, "w") as f:
        for k in range(K_t):
            for m in range(M_t):
                block = a_int[m*BLOCK_SIZE:(m+1)*BLOCK_SIZE, k*BLOCK_SIZE:(k+1)*BLOCK_SIZE]
                for row in block:
                    hex_str = "".join(f"{v.astype(np.uint8):02x}" for v in reversed(row))
                    f.write(f"{hex_str}\n")


def emit_b_input(path, b_int):
    """Column-packed B input. 256-bit per column, tile order (N, K)."""
    K, N = b_int.shape
    K_t, N_t = K // BLOCK_SIZE, N // BLOCK_SIZE
    with open(path, "w") as f:
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
    M_t = M_pad // BLOCK_SIZE
    with open(path, "w") as f:
        for k in range(K_t):
            for m in range(M_t):
                for row in range(BLOCK_SIZE):
                    f.write(f"{a_scale[m*BLOCK_SIZE+row, k]:02x}\n")


def emit_b_scale(path, b_scale):
    """E8M0 scales for B. Order matches emit_b_input: N_tile → K_tile → col."""
    K_t, N_pad = b_scale.shape
    N_t = N_pad // BLOCK_SIZE
    with open(path, "w") as f:
        for n in range(N_t):
            for k in range(K_t):
                for col in range(BLOCK_SIZE):
                    f.write(f"{b_scale[k, n*BLOCK_SIZE+col]:02x}\n")
