"""Tests for mxp_tools.quant."""
import numpy as np
import pytest

from mxp_tools.const import BLOCK_SIZE, INT2, INT4, INT8, MAX_INT
from mxp_tools.quant import quantize_block_mx, quantize_matrix_mx


@pytest.mark.parametrize("prec", [INT2, INT4, INT8])
def test_zero_block_returns_zero_int_and_zero_scale(prec):
    block = np.zeros(BLOCK_SIZE, dtype=np.float32)
    raw, scale = quantize_block_mx(block, prec)
    assert raw.dtype == np.int8
    assert scale == np.uint8(0)
    assert np.all(raw == 0)


@pytest.mark.parametrize("prec", [INT2, INT4, INT8])
def test_clip_to_max_int(prec):
    # All-equal large values should saturate to ±MAX_INT after round
    block = np.full(BLOCK_SIZE, 1e30, dtype=np.float32)
    raw, _ = quantize_block_mx(block, prec)
    assert raw.max() <= MAX_INT[prec]
    assert raw.min() >= -MAX_INT[prec]


def test_rejects_wrong_dtype():
    block = np.zeros(BLOCK_SIZE, dtype=np.float64)
    with pytest.raises(TypeError):
        quantize_block_mx(block, INT8)


def test_rejects_wrong_shape():
    block = np.zeros(BLOCK_SIZE + 1, dtype=np.float32)
    with pytest.raises(ValueError):
        quantize_block_mx(block, INT8)


def test_rejects_bad_prec():
    block = np.zeros(BLOCK_SIZE, dtype=np.float32)
    with pytest.raises(ValueError):
        quantize_block_mx(block, 16)


def test_rejects_nan_input():
    # C5 fix: was previously a downstream OverflowError/ValueError
    block = np.zeros(BLOCK_SIZE, dtype=np.float32)
    block[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        quantize_block_mx(block, INT8)


def test_rejects_inf_input():
    block = np.zeros(BLOCK_SIZE, dtype=np.float32)
    block[3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        quantize_block_mx(block, INT8)


def test_matrix_quant_axis1_shape():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 64)).astype(np.float32)
    int_A, scale_A = quantize_matrix_mx(A, INT8, block_axis=1)
    assert int_A.shape == (4, 64)
    assert int_A.dtype == np.int8
    assert scale_A.shape == (4, 64 // BLOCK_SIZE)
    assert scale_A.dtype == np.uint8


def test_matrix_quant_axis0_shape():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((64, 8)).astype(np.float32)
    int_B, scale_B = quantize_matrix_mx(B, INT8, block_axis=0)
    assert int_B.shape == (64, 8)
    assert scale_B.shape == (64 // BLOCK_SIZE, 8)


def test_matrix_quant_rejects_non_block_multiple():
    A = np.zeros((4, 33), dtype=np.float32)
    with pytest.raises(ValueError, match="multiple of"):
        quantize_matrix_mx(A, INT8, block_axis=1)
