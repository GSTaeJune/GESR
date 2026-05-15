"""Tests for mxp_tools.hwio — including C1 @addr, L5 dup-write, NEW-4 emit round-trip."""
import os
import struct

import numpy as np
import pytest

from mxp_tools.const import BLOCK_SIZE
from mxp_tools.hwio import (
    default_banks_split_rows,
    default_single_bank_row_major,
    emit_a_input_bp,
    emit_a_input_bs,
    emit_a_scale,
    emit_b_input,
    emit_b_scale,
    fp32_to_hex_words,
    gather_banks,
    get_bit_slices,
    read_writememh_fp32,
    write_writememh,
)


# ─── C1: @addr handling ────────────────────────────────────────────────────

def test_read_writememh_handles_addr_markers(tmp_path):
    """@<addr> jumps should place words at the indicated address, not sequentially."""
    p = tmp_path / "out.mem"
    # Word 0xDEADBEEF at address 0, then jump to addr 5 with word 0xCAFEBABE
    p.write_text("@00000000\ndeadbeef\n@00000005\ncafebabe\n", encoding="ascii")
    arr = read_writememh_fp32(p)
    # Expect length 6: 0=DEADBEEF, 1-4=NaN, 5=CAFEBABE
    assert arr.shape == (6,)
    expected0 = struct.unpack("<f", struct.pack("<I", 0xDEADBEEF))[0]
    expected5 = struct.unpack("<f", struct.pack("<I", 0xCAFEBABE))[0]
    # 0xDEADBEEF is a NaN bit pattern — compare via uint32 view
    assert arr[0:1].view(np.uint32)[0] == 0xDEADBEEF
    assert np.isnan(arr[1]) and np.isnan(arr[2]) and np.isnan(arr[3]) and np.isnan(arr[4])
    assert arr[5] == np.float32(expected5)
    # Suppress unused-name warning
    _ = expected0


def test_read_writememh_nonzero_start(tmp_path):
    """Dump that starts with @ at nonzero address fills earlier slots with NaN."""
    p = tmp_path / "out.mem"
    p.write_text("@00000003\n00000000\n00000000\n", encoding="ascii")
    arr = read_writememh_fp32(p)
    assert arr.shape == (5,)
    assert np.isnan(arr[0]) and np.isnan(arr[1]) and np.isnan(arr[2])
    assert arr[3] == 0.0 and arr[4] == 0.0


def test_read_writememh_skips_comments(tmp_path):
    p = tmp_path / "out.mem"
    p.write_text("// header\n00000000\n  // mid\n3f800000\n", encoding="ascii")
    arr = read_writememh_fp32(p)
    assert arr.tolist() == [0.0, 1.0]


def test_read_writememh_empty_file(tmp_path):
    p = tmp_path / "out.mem"
    p.write_text("// just comments\n", encoding="ascii")
    arr = read_writememh_fp32(p)
    assert arr.shape == (0,)


# ─── fp32 round-trip ───────────────────────────────────────────────────────

def test_fp32_hex_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    src = rng.standard_normal(64).astype(np.float32)
    p = tmp_path / "rt.mem"
    write_writememh(p, fp32_to_hex_words(src), addr_every=16)
    got = read_writememh_fp32(p)
    np.testing.assert_array_equal(got, src)


# ─── gather_banks ──────────────────────────────────────────────────────────

def test_gather_banks_single_layout(tmp_path):
    rng = np.random.default_rng(0)
    M, N = 4, 8
    C_truth = rng.standard_normal((M, N)).astype(np.float32)
    bank = tmp_path / "bank0.mem"
    write_writememh(bank, fp32_to_hex_words(C_truth))
    C = gather_banks([bank], M, N, default_single_bank_row_major)
    np.testing.assert_array_equal(C, C_truth)


def test_gather_banks_preserves_legitimate_nan(tmp_path):
    """C3 fix: HW-emitted NaN must survive (was treated as missing coverage)."""
    M, N = 2, 4
    C_truth = np.array([[1.0, 2.0, 3.0, np.nan],
                        [4.0, 5.0, 6.0, 7.0]], dtype=np.float32)
    bank = tmp_path / "bank0.mem"
    write_writememh(bank, fp32_to_hex_words(C_truth))
    C = gather_banks([bank], M, N, default_single_bank_row_major)
    assert np.isnan(C[0, 3])
    # The rest are equal
    finite_mask = ~np.isnan(C_truth)
    np.testing.assert_array_equal(C[finite_mask], C_truth[finite_mask])


def test_gather_banks_detects_duplicate_writes(tmp_path):
    """L5 fix: mapping that returns the same (m, n) twice should raise."""
    M, N = 2, 2
    bank = tmp_path / "bank0.mem"
    write_writememh(bank, fp32_to_hex_words(np.zeros(8, dtype=np.float32)))

    def buggy_mapping(bank_idx, word_offset, M, N):
        # Always map to (0, 0) — duplicate writes
        if word_offset >= M * N:
            return None
        return 0, 0

    with pytest.raises(ValueError, match="duplicate write"):
        gather_banks([bank], M, N, buggy_mapping)


def test_gather_banks_detects_missing_coverage(tmp_path):
    M, N = 2, 4
    bank = tmp_path / "bank0.mem"
    write_writememh(bank, fp32_to_hex_words(np.zeros(2, dtype=np.float32)))
    # default layout expects M*N=8 words, but file has 2
    with pytest.raises(ValueError, match="not covered"):
        gather_banks([bank], M, N, default_single_bank_row_major)


def test_split_rows_layout(tmp_path):
    M, N = 4, 4
    rng = np.random.default_rng(0)
    C_truth = rng.standard_normal((M, N)).astype(np.float32)
    # 2 banks, 2 rows each
    bank0 = tmp_path / "bank0.mem"
    bank1 = tmp_path / "bank1.mem"
    write_writememh(bank0, fp32_to_hex_words(C_truth[0:2].ravel()))
    write_writememh(bank1, fp32_to_hex_words(C_truth[2:4].ravel()))
    C = gather_banks([bank0, bank1], M, N, default_banks_split_rows(2))
    np.testing.assert_array_equal(C, C_truth)


# ─── C4: emitters reject non-block-multiple shapes ─────────────────────────

def test_emit_a_input_bs_rejects_bad_shape(tmp_path):
    a = np.zeros((33, 32), dtype=np.int8)
    with pytest.raises(ValueError, match="multiple of"):
        emit_a_input_bs(tmp_path / "x.hex", a, prec=8)


def test_emit_b_input_rejects_bad_shape(tmp_path):
    b = np.zeros((32, 33), dtype=np.int8)
    with pytest.raises(ValueError, match="multiple of"):
        emit_b_input(tmp_path / "x.hex", b)


# ─── NEW-4: emit file format / line-count regression ──────────────────────

def test_emit_a_input_bs_line_count(tmp_path):
    """BS format: per (K_tile, M_tile) — 32 rows × prec slices per block."""
    M, K, prec = 32, 64, 8
    a = np.zeros((M, K), dtype=np.int8)
    path = tmp_path / "a_bs.hex"
    emit_a_input_bs(path, a, prec)
    lines = path.read_text(encoding="ascii").splitlines()
    K_t = K // BLOCK_SIZE
    M_t = M // BLOCK_SIZE
    # K_t × M_t × 32 rows × prec slices
    assert len(lines) == K_t * M_t * BLOCK_SIZE * prec
    assert all(len(line) == 8 for line in lines)  # 8 hex chars per line


def test_emit_a_input_bp_line_count(tmp_path):
    M, K = 32, 64
    a = np.zeros((M, K), dtype=np.int8)
    path = tmp_path / "a_bp.hex"
    emit_a_input_bp(path, a)
    lines = path.read_text(encoding="ascii").splitlines()
    K_t = K // BLOCK_SIZE
    M_t = M // BLOCK_SIZE
    assert len(lines) == K_t * M_t * BLOCK_SIZE
    # Each line is 32 bytes × 2 hex chars = 64 chars
    assert all(len(line) == 64 for line in lines)


def test_emit_a_scale_line_count(tmp_path):
    """Scale matrix shape (M_pad, K_t) → K_t × M_pad lines (1 byte each, %02x)."""
    M_pad, K_t = 32, 2
    sc = np.zeros((M_pad, K_t), dtype=np.uint8)
    path = tmp_path / "a_scale.hex"
    emit_a_scale(path, sc)
    lines = path.read_text(encoding="ascii").splitlines()
    assert len(lines) == K_t * M_pad
    assert all(len(line) == 2 for line in lines)


def test_emit_uses_lf_only(tmp_path):
    """L4 fix: emitters must write LF, not CRLF, even on Windows."""
    a = np.zeros((32, 32), dtype=np.int8)
    path = tmp_path / "a_bp.hex"
    emit_a_input_bp(path, a)
    raw = path.read_bytes()
    assert b"\r" not in raw, "emitter must use LF-only line endings"


# ─── get_bit_slices ────────────────────────────────────────────────────────

def test_get_bit_slices_msb_first():
    # int8 row of 0x80 (MSB set, all others 0). With prec=8, slice[0] (MSB)
    # should have bit 0 of each word position set. Going through the encoding:
    # 0x80 = 0b10000000. Bit 7 (MSB) = 1 → goes into slice 0 (first slice).
    row = np.full(32, np.int8(-128))  # 0x80
    slices = get_bit_slices(row, 32, prec=8)
    # MSB slice has all 32 bits set (one bit per row position)
    assert slices[0] == (1 << 32) - 1  # 0xFFFFFFFF
    # Other slices are 0
    for s in slices[1:]:
        assert s == 0
