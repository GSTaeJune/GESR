"""Tests for mxp_tools.gemm.mxint_gemm_golden."""
import numpy as np
import pytest

from mxp_tools.const import INT8
from mxp_tools.gemm import mxint_gemm_golden
from mxp_tools.quant import quantize_matrix_mx


def test_golden_matches_reference_within_quant_error():
    rng = np.random.default_rng(0)
    M, K, N = 32, 64, 32
    A = rng.standard_normal((M, K)).astype(np.float32) * 0.5
    B = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    C_truth = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)

    int_A, scale_A = quantize_matrix_mx(A, INT8, block_axis=1)
    int_B, scale_B = quantize_matrix_mx(B, INT8, block_axis=0)
    C = mxint_gemm_golden(int_A, scale_A, INT8, int_B, scale_B, INT8)

    # MXINT8 should be within a few % of FP32 on random data. K=64 small-block
    # quant error compounds. 5% is the loose sanity bound; visual viz catches
    # finer regressions.
    rel = np.abs(C - C_truth).max() / np.abs(C_truth).max()
    assert rel < 5e-2, f"MXINT8 should approximate FP32 closely, got rel={rel}"


def test_rejects_wrong_dtype():
    int_A = np.zeros((32, 64), dtype=np.int16)
    scale_A = np.zeros((32, 2), dtype=np.uint8)
    int_B = np.zeros((64, 32), dtype=np.int8)
    scale_B = np.zeros((2, 32), dtype=np.uint8)
    with pytest.raises(TypeError):
        mxint_gemm_golden(int_A, scale_A, INT8, int_B, scale_B, INT8)


def test_rejects_shape_mismatch():
    int_A = np.zeros((32, 64), dtype=np.int8)
    scale_A = np.zeros((32, 2), dtype=np.uint8)
    int_B = np.zeros((96, 32), dtype=np.int8)
    scale_B = np.zeros((3, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        mxint_gemm_golden(int_A, scale_A, INT8, int_B, scale_B, INT8)


def test_zero_inputs_produce_zero_output():
    int_A = np.zeros((32, 64), dtype=np.int8)
    scale_A = np.zeros((32, 2), dtype=np.uint8)
    int_B = np.zeros((64, 32), dtype=np.int8)
    scale_B = np.zeros((2, 32), dtype=np.uint8)
    C = mxint_gemm_golden(int_A, scale_A, INT8, int_B, scale_B, INT8)
    assert np.all(C == 0.0)
