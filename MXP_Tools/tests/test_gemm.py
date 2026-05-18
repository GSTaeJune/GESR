"""Tests for mxp_tools.gemm.mxint_gemm_golden."""
import numpy as np
import pytest

from mxp_tools.const import BLOCK_SIZE, INT2, INT4, INT8
from mxp_tools.gemm import mxint_gemm_golden
from mxp_tools.quant import quantize_block_mx, quantize_matrix_mx


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


# ---------------------------------------------------------------------------
# Mixed-precision prec_A (per-row, per-K-block weight precision)
# ---------------------------------------------------------------------------

def _quantize_weight_mixed(W_fp32, prec_map):
    """Per-(row, K-block) MX quantize. Mirrors sim/gen_mixed.quantize_weight_mixed
    but local-only to avoid the test depending on a sibling project path."""
    M, K = W_fp32.shape
    n_blocks = K // BLOCK_SIZE
    assert prec_map.shape == (M, n_blocks)
    W_int = np.zeros((M, K), dtype=np.int8)
    W_scale = np.zeros((M, n_blocks), dtype=np.uint8)
    for m in range(M):
        for kt in range(n_blocks):
            sl = slice(kt * BLOCK_SIZE, (kt + 1) * BLOCK_SIZE)
            q, s = quantize_block_mx(W_fp32[m, sl], int(prec_map[m, kt]))
            W_int[m, sl] = q
            W_scale[m, kt] = s
    return W_int, W_scale


def test_array_prec_A_matches_scalar_when_uniform():
    """prec_A=ndarray filled with uniform value must equal prec_A=scalar."""
    rng = np.random.default_rng(1)
    M, K, N = 32, 64, 32
    A = rng.standard_normal((M, K)).astype(np.float32) * 0.5
    B = rng.standard_normal((K, N)).astype(np.float32) * 0.5

    for prec in (INT2, INT4, INT8):
        int_A, scale_A = quantize_matrix_mx(A, prec, block_axis=1)
        int_B, scale_B = quantize_matrix_mx(B, INT8, block_axis=0)
        C_scalar = mxint_gemm_golden(int_A, scale_A, prec, int_B, scale_B, INT8)
        prec_arr = np.full((M, K // BLOCK_SIZE), prec, dtype=np.uint8)
        C_array = mxint_gemm_golden(int_A, scale_A, prec_arr, int_B, scale_B, INT8)
        assert np.array_equal(C_scalar, C_array), (
            f"prec={prec}: array path must be bit-exact equal to scalar path"
        )


def test_array_prec_A_actual_mixed():
    """Mixed prec_A: rows 0..15 use INT2, rows 16..31 use INT8 — golden must
    agree with row-block-wise hand computation (each row's K-block has its
    own implicit scale)."""
    rng = np.random.default_rng(2)
    M, K, N = 32, 64, 32
    n_blocks = K // BLOCK_SIZE
    W = rng.standard_normal((M, K)).astype(np.float32) * 0.5
    Aact = rng.standard_normal((K, N)).astype(np.float32) * 0.5

    prec_map = np.zeros((M, n_blocks), dtype=np.uint8)
    prec_map[:M // 2] = INT2     # top half: weight INT2
    prec_map[M // 2:] = INT8     # bottom half: weight INT8

    W_int, W_scale = _quantize_weight_mixed(W, prec_map)
    A_int, A_scale = quantize_matrix_mx(Aact, INT8, block_axis=0)

    C_mixed = mxint_gemm_golden(W_int, W_scale, prec_map, A_int, A_scale, INT8)

    # Hand-computed reference: split rows by prec, run uniform golden on each,
    # then re-stack. Equivalent because prec_A is row-block constant within
    # each subset.
    top_int, top_scale = W_int[:M // 2], W_scale[:M // 2]
    bot_int, bot_scale = W_int[M // 2:], W_scale[M // 2:]
    C_top = mxint_gemm_golden(top_int, top_scale, INT2, A_int, A_scale, INT8)
    C_bot = mxint_gemm_golden(bot_int, bot_scale, INT8, A_int, A_scale, INT8)
    C_ref = np.vstack([C_top, C_bot])

    assert np.array_equal(C_mixed, C_ref)


def test_array_prec_A_rejects_bad_value():
    int_A = np.zeros((32, 64), dtype=np.int8)
    scale_A = np.zeros((32, 2), dtype=np.uint8)
    int_B = np.zeros((64, 32), dtype=np.int8)
    scale_B = np.zeros((2, 32), dtype=np.uint8)
    prec_arr = np.full((32, 2), 3, dtype=np.uint8)   # 3 is not a valid MXINT prec
    with pytest.raises(ValueError, match="prec_A array contains values not in"):
        mxint_gemm_golden(int_A, scale_A, prec_arr, int_B, scale_B, INT8)


def test_array_prec_A_rejects_wrong_shape():
    int_A = np.zeros((32, 64), dtype=np.int8)
    scale_A = np.zeros((32, 2), dtype=np.uint8)
    int_B = np.zeros((64, 32), dtype=np.int8)
    scale_B = np.zeros((2, 32), dtype=np.uint8)
    prec_arr = np.full((32, 3), INT8, dtype=np.uint8)   # 3 blocks not 2
    with pytest.raises(ValueError, match=r"prec_A array shape"):
        mxint_gemm_golden(int_A, scale_A, prec_arr, int_B, scale_B, INT8)
