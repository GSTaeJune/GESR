"""Unit test: interleaved_row_major_16bank 매핑은
sram_1rw_banked 의 INTERLEAVED 주소 분할의 정확한 역연산이어야 한다.
"""
import numpy as np
from mxp_tools import hwio


def test_interleaved_row_major_round_trip():
    M, N = 128, 128
    # forward: C[m,n] → (bank, word)
    flat = lambda m, n: m * N + n
    bank = lambda f: f % 16
    word = lambda f: f // 16

    # inverse via hwio:
    for m, n in [(0, 0), (0, 15), (0, 16), (1, 0), (127, 127), (63, 31)]:
        f = flat(m, n)
        b = bank(f)
        w = word(f)
        result = hwio.interleaved_row_major_16bank(b, w, M, N)
        assert result == (m, n), f"({m},{n})→bank{b}/word{w}→{result}"


def test_interleaved_row_major_out_of_range():
    assert hwio.interleaved_row_major_16bank(0, 9999, 128, 128) is None


def test_interleaved_row_major_32bank_round_trip():
    """col-parallel RMW design: bank = flat % 32, word = flat // 32."""
    M, N = 128, 128
    flat = lambda m, n: m * N + n
    bank = lambda f: f % 32
    word = lambda f: f // 32

    for m, n in [(0, 0), (0, 31), (0, 32), (1, 0), (127, 127), (63, 31), (5, 96)]:
        f = flat(m, n)
        b = bank(f)
        w = word(f)
        result = hwio.interleaved_row_major_32bank(b, w, M, N)
        assert result == (m, n), f"({m},{n})→bank{b}/word{w}→{result}"


def test_interleaved_row_major_32bank_out_of_range():
    assert hwio.interleaved_row_major_32bank(0, 9999, 128, 128) is None
