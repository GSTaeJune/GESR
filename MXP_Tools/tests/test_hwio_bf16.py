"""read_writememh_bf16: 16-bit bf16 dump words -> exact fp32 upcast.

The upcast is bits<<16 (bf16 IS fp32's top 16 bits). A written 0x0000 is a
legitimate +0.0 and must NOT read as a gap -- gap detection uses a separate
written-mask (mirroring read_writememh_fp32), never the value itself.
"""
import numpy as np
import pytest

from mxp_tools import hwio


def _write(tmp_path, text):
    p = tmp_path / "bank0.mem"
    p.write_text(text, newline="\n")
    return str(p)


def test_bf16_roundtrip_basic(tmp_path):
    path = _write(tmp_path, "3f80\n4000\nc040\n0000\n8000\n")
    out = hwio.read_writememh_bf16(path)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[:3], np.float32([1.0, 2.0, -3.0]))
    assert out[3] == 0.0 and not np.signbit(out[3])   # written +0.0, not a gap
    assert out[4] == 0.0 and np.signbit(out[4])       # -0.0 sign preserved


def test_bf16_upcast_is_exact_bits(tmp_path):
    # min subnormal 0x0001 -> fp32 bits 0x00010000
    path = _write(tmp_path, "0001\n7f80\nff80\n")
    out = hwio.read_writememh_bf16(path)
    assert out[0].view(np.uint32) == np.uint32(0x00010000)
    assert np.isinf(out[1]) and out[1] > 0
    assert np.isinf(out[2]) and out[2] < 0


def test_bf16_gap_is_nan(tmp_path):
    path = _write(tmp_path, "@2\n3f80\n")
    out = hwio.read_writememh_bf16(path)
    assert len(out) == 3
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == np.float32(1.0)


def test_bf16_overwidth_word_raises(tmp_path):
    path = _write(tmp_path, "12345\n")
    with pytest.raises(ValueError, match="16 bits"):
        hwio.read_writememh_bf16(path)


def test_gather_banks_with_bf16_reader(tmp_path):
    # 2x2 C over 2 banks, flat = m*N+n, bank=flat%2, word=flat//2
    (tmp_path / "b0.mem").write_text("3f80\n4040\n", newline="\n")  # flat 0, 2
    (tmp_path / "b1.mem").write_text("4000\n4080\n", newline="\n")  # flat 1, 3
    def mapping(bank_idx, word_offset, M, N):
        flat = word_offset * 2 + bank_idx
        return divmod(flat, N) if flat < M * N else None
    C = hwio.gather_banks([str(tmp_path / "b0.mem"), str(tmp_path / "b1.mem")],
                          2, 2, mapping, reader=hwio.read_writememh_bf16)
    np.testing.assert_array_equal(C, np.float32([[1.0, 2.0], [3.0, 4.0]]))
