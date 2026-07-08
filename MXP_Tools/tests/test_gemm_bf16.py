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


from mxp_tools.const import IMPLICIT_SCALE_EXP


def _per_add_bf16_ref(int_A, scale_A, prec_A, int_B, scale_B, prec_B):
    """Independent explicit-loop bf16 accumulation (scalar prec_A only).

    Uses the explicit fp64-intermediate add form `(f64(C)+f64(fp_a))->bf16`;
    the golden uses native `C + fp_a`. These are provably identical (spec 3.3),
    so equality here also cross-checks native-vs-explicit add.
    """
    M, K = int_A.shape
    _, N = int_B.shape
    nb = K // BLOCK_SIZE
    impl = IMPLICIT_SCALE_EXP[prec_A] + IMPLICIT_SCALE_EXP[prec_B]
    C = np.zeros((M, N), dtype=bfloat16)
    for blk in range(nb):
        sl = slice(blk * BLOCK_SIZE, (blk + 1) * BLOCK_SIZE)
        bi = int_A[:, sl].astype(np.int32) @ int_B[sl, :].astype(np.int32)
        E = (scale_A[:, blk:blk + 1].astype(np.int64)
             + scale_B[blk:blk + 1, :].astype(np.int64) - 254 - impl)
        r = bi.astype(np.float32).astype(bfloat16)
        fp_a = np.ldexp(r.astype(np.float64), E).astype(bfloat16)
        C = (C.astype(np.float64) + fp_a.astype(np.float64)).astype(bfloat16)
    return C.astype(np.float32)


def _cast_once_ref(int_A, scale_A, prec_A, int_B, scale_B, prec_B):
    """Wrong-on-purpose: round each block's dequant to bf16 (like HW), but sum
    the partials in float64 and cast to bf16 only ONCE at the end."""
    M, K = int_A.shape
    _, N = int_B.shape
    nb = K // BLOCK_SIZE
    impl = IMPLICIT_SCALE_EXP[prec_A] + IMPLICIT_SCALE_EXP[prec_B]
    acc = np.zeros((M, N), dtype=np.float64)
    for blk in range(nb):
        sl = slice(blk * BLOCK_SIZE, (blk + 1) * BLOCK_SIZE)
        bi = int_A[:, sl].astype(np.int32) @ int_B[sl, :].astype(np.int32)
        E = (scale_A[:, blk:blk + 1].astype(np.int64)
             + scale_B[blk:blk + 1, :].astype(np.int64) - 254 - impl)
        r = bi.astype(np.float32).astype(bfloat16)
        acc += np.ldexp(r.astype(np.float64), E)
    return acc.astype(bfloat16).astype(np.float32)


def test_bf16_rounds_per_k_tile_not_once():
    # Deep-K (16 blocks) so swamping distinguishes per-add rounding from cast-once.
    args = _rand_case(7, M=32, K=32 * 16, N=32)
    golden = mxint_gemm_golden(*args, accum_dtype="bf16")
    assert np.array_equal(golden, _per_add_bf16_ref(*args)), \
        "golden must accumulate per-K-tile in bf16"
    cast_once = _cast_once_ref(*args)
    assert not np.array_equal(golden, cast_once), (
        "test is vacuous: this case does not distinguish per-add from cast-once; "
        "pick a seed/shape where they differ"
    )


def test_bf16_subnormal_dequant_self_consistent():
    # Tiny magnitudes push E8M0 exponents far below 127 so some block partials
    # land in the bf16-subnormal range (|value| < 2^-126), exercising ldexp's
    # subnormal rounding. Golden must match the independent per-add reference there.
    rng = np.random.default_rng(11)
    M, K, N = 32, 64, 32
    A = rng.standard_normal((M, K)).astype(np.float32) * (2.0 ** -65)
    B = rng.standard_normal((K, N)).astype(np.float32) * (2.0 ** -65)
    int_A, scale_A = quantize_matrix_mx(A, INT8, block_axis=1)
    int_B, scale_B = quantize_matrix_mx(B, INT8, block_axis=0)
    golden = mxint_gemm_golden(int_A, scale_A, INT8, int_B, scale_B, INT8, accum_dtype="bf16")
    ref = _per_add_bf16_ref(int_A, scale_A, INT8, int_B, scale_B, INT8)
    assert np.array_equal(golden, ref)


def test_bf16_tracks_fp64_truth():
    """GATE, independent of the E-formula: the bf16 golden must track the true
    fp64 matmul within (MX-quant + bf16-accum) error. This is the one test that
    catches a *systematic* dequant/exponent-formula bug — the ml_dtypes-based
    tests and _per_add_bf16_ref all share the same E formula and cannot. Bound
    (5%) is well above the observed ~1.6% error at K=64 and trips on a formula bug
    (which gives ~100% error). Satisfies spec 5.2 #7 (correctness anchor)."""
    rng = np.random.default_rng(20)
    M, K, N = 32, 64, 32
    A = rng.standard_normal((M, K)).astype(np.float32) * 0.5
    B = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    C_truth = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
    int_A, scale_A = quantize_matrix_mx(A, INT8, block_axis=1)
    int_B, scale_B = quantize_matrix_mx(B, INT8, block_axis=0)
    C = mxint_gemm_golden(int_A, scale_A, INT8, int_B, scale_B, INT8, accum_dtype="bf16")
    rel = np.abs(C - C_truth).max() / np.abs(C_truth).max()
    assert rel < 5e-2, f"bf16 golden must track FP32 truth, got rel={rel}"


def test_bf16_single_block_hand_derived_value():
    """Hand-derived end-to-end value (spec formula + IEEE bf16; NO ml_dtypes in the
    expectation). Single K-block => one exact add to zero.
      32 ones . 32 ones = block_int = 32 (every element).
      e_a = e_b = 127 (2^0); INT8 implicit = 6 each.
      E = 127 + 127 - 254 - 6 - 6 = -12.
      value = 32 * 2^-12 = 2^-7 = 0.0078125, exactly bf16-representable.
    A wrong E-formula (e.g. -253) changes this exact value, so it is a hard
    non-tautological anchor for the dequant/exponent math."""
    int_A = np.ones((32, 32), dtype=np.int8)
    int_B = np.ones((32, 32), dtype=np.int8)
    scale_A = np.full((32, 1), 127, dtype=np.uint8)
    scale_B = np.full((1, 32), 127, dtype=np.uint8)
    C = mxint_gemm_golden(int_A, scale_A, INT8, int_B, scale_B, INT8, accum_dtype="bf16")
    assert np.all(C == np.float32(2.0 ** -7))
