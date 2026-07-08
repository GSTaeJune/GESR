"""BF16 accumulation path in mxint_gemm_golden."""
import numpy as np
import pytest

from mxp_tools.const import BLOCK_SIZE, INT2, INT4, INT8
from mxp_tools.gemm import mxint_gemm_golden
from mxp_tools.quant import quantize_matrix_mx

bfloat16 = pytest.importorskip("ml_dtypes").bfloat16


def _rand_case(seed, M=32, K=64, N=32, prec_a=INT8, prec_b=INT8, scale=0.5):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K)).astype(np.float32) * scale
    B = rng.standard_normal((K, N)).astype(np.float32) * scale
    int_A, scale_A = quantize_matrix_mx(A, prec_a, block_axis=1)
    int_B, scale_B = quantize_matrix_mx(B, prec_b, block_axis=0)
    return int_A, scale_A, prec_a, int_B, scale_B, prec_b


def test_bf16_deterministic():
    args = _rand_case(0)
    c1 = mxint_gemm_golden(*args, accum_dtype="bf16")
    c2 = mxint_gemm_golden(*args, accum_dtype="bf16")
    assert np.array_equal(c1, c2)


def test_bf16_output_is_bf16_representable():
    C = mxint_gemm_golden(*_rand_case(1), accum_dtype="bf16")
    assert np.array_equal(C, C.astype(bfloat16).astype(np.float32))


def test_bf16_uniform_prec_array_matches_scalar():
    rng = np.random.default_rng(2)
    M, K, N = 32, 64, 32
    A = rng.standard_normal((M, K)).astype(np.float32) * 0.5
    B = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    for prec in (INT2, INT4, INT8):
        int_A, scale_A = quantize_matrix_mx(A, prec, block_axis=1)
        int_B, scale_B = quantize_matrix_mx(B, INT8, block_axis=0)
        c_scalar = mxint_gemm_golden(int_A, scale_A, prec, int_B, scale_B, INT8, accum_dtype="bf16")
        prec_arr = np.full((M, K // BLOCK_SIZE), prec, dtype=np.uint8)
        c_array = mxint_gemm_golden(int_A, scale_A, prec_arr, int_B, scale_B, INT8, accum_dtype="bf16")
        assert np.array_equal(c_scalar, c_array), f"prec={prec}"


def test_bf16_zero_inputs_zero_output():
    int_A = np.zeros((32, 64), dtype=np.int8)
    scale_A = np.zeros((32, 2), dtype=np.uint8)
    int_B = np.zeros((64, 32), dtype=np.int8)
    scale_B = np.zeros((2, 32), dtype=np.uint8)
    C = mxint_gemm_golden(int_A, scale_A, INT8, int_B, scale_B, INT8, accum_dtype="bf16")
    assert np.all(C == 0.0)


def test_invalid_accum_dtype_raises():
    with pytest.raises(ValueError, match="accum_dtype"):
        mxint_gemm_golden(*_rand_case(3), accum_dtype="fp16")
