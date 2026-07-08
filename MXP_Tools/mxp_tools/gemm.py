"""Golden-reference MXINT GEMM. Matches MXP_Soft mxint_gemm.

``accum_dtype='fp32'`` (default) is the pure-numpy path described below.
``accum_dtype='bf16'`` accumulates in bfloat16 via ml_dtypes, mirroring the
HW RMW datapath: per K-block (order 0..n-1) it forms the exact int32 block
sum, rounds int->bf16 (RNE), applies a single integer exponent shift 2^E,
and adds into a native bf16 accumulator (RNE). It imports ml_dtypes lazily
(only when accum_dtype='bf16'), so the fp32 path stays numpy-only.

Per OCP MX Spec §6.3.2:

    for each K=32 block:
        int32 block sum = a_int (int8) @ b_int (int8)
        FP32 partial    = int32 block sum
                          * 2^(e_a - 127) * 2^(e_b - 127)
                          * 2^-(implicit_a + implicit_b)
        FP32 accumulator += FP32 partial

Per-block scale application is mandatory — scales differ per block, so an
int32 accumulation across blocks followed by a single scale would be wrong.

Bit-exactness contract:
    `C += block_fp` accumulates blocks in K-tile order (block 0, 1, 2, ...)
    using a single FP32 accumulator. HW that uses a different reduction
    tree (e.g., pairwise), a wider accumulator (FP64 / extended), or fused
    multiply-add ordering will produce different FP32 round-off and may
    show non-zero `C_hw - C_sw` even with bit-correct PE math. If your
    bring-up sees that, suspect accumulation order before chasing PE bugs.
"""
import numpy as np

from .const import BLOCK_SIZE, IMPLICIT_SCALE_EXP


def _e8m0_to_fp32(scale_e8m0):
    """E8M0 (uint8 biased exponent) → FP32 scale factor 2^(e - 127)."""
    return (2.0 ** (scale_e8m0.astype(np.float64) - 127.0)).astype(np.float32)


def mxint_gemm_golden(int_A, scale_A, prec_A, int_B, scale_B, prec_B, accum_dtype="fp32"):
    """Block-wise int32→FP32 accumulator GEMM. Inputs are pre-quantized.

    Args:
      int_A:   int8,  (M, K)
      scale_A: uint8, (M, K // 32)
      prec_A:  int in {2, 4, 8}, OR ndarray (M, K // 32) of per-(row, K-block)
               precisions for mixed-precision weights. Each entry must be in
               {2, 4, 8}.
      int_B:   int8,  (K, N)
      scale_B: uint8, (K // 32, N)
      prec_B:  int in {2, 4, 8} (uniform along K and N).

    Returns:
      C_fp32: float32, (M, N). HW that is bit-correct produces the same matrix.

    Mixed-precision notes:
      When prec_A is an ndarray, each row m and K-block blk applies its own
      implicit_scale = 2^-(IMPLICIT_SCALE_EXP[prec_A[m, blk]] + IMPLICIT_SCALE_EXP[prec_B]).
      Scalar prec_A is a uniform broadcast of the same idea — the two code
      paths must produce identical FP32 results when prec_A is uniform; the
      mixed pytest enforces this.
    """
    if int_A.dtype != np.int8 or int_B.dtype != np.int8:
        raise TypeError(f"int_A, int_B must be int8 (got {int_A.dtype}, {int_B.dtype})")
    if scale_A.dtype != np.uint8 or scale_B.dtype != np.uint8:
        raise TypeError(f"scale_A, scale_B must be uint8 (got {scale_A.dtype}, {scale_B.dtype})")
    if accum_dtype not in ("fp32", "bf16"):
        raise ValueError(f"accum_dtype must be 'fp32' or 'bf16', got {accum_dtype!r}")

    M, K = int_A.shape
    K2, N = int_B.shape
    if K != K2:
        raise ValueError(f"shape mismatch: int_A={int_A.shape}, int_B={int_B.shape}")
    if K % BLOCK_SIZE != 0:
        raise ValueError(f"K={K} must be a multiple of {BLOCK_SIZE}")
    n_blocks = K // BLOCK_SIZE
    if scale_A.shape != (M, n_blocks):
        raise ValueError(f"scale_A shape {scale_A.shape} != ({M}, {n_blocks})")
    if scale_B.shape != (n_blocks, N):
        raise ValueError(f"scale_B shape {scale_B.shape} != ({n_blocks}, {N})")

    # prec_B is always scalar (current scope: activation precision uniform per layer).
    if prec_B not in IMPLICIT_SCALE_EXP:
        raise ValueError(f"prec_B must be in {sorted(IMPLICIT_SCALE_EXP)}, got {prec_B}")
    impl_b = IMPLICIT_SCALE_EXP[prec_B]

    # prec_A: scalar (uniform) or (M, n_blocks) ndarray (mixed-prec weights).
    prec_a_is_array = isinstance(prec_A, np.ndarray)
    if prec_a_is_array:
        if prec_A.shape != (M, n_blocks):
            raise ValueError(
                f"prec_A array shape {prec_A.shape} != ({M}, {n_blocks})"
            )
        # Build per-(row, block) impl_a via LUT. uint8/int dtypes accepted.
        impl_lut = np.full(256, -1, dtype=np.int64)
        for p, v in IMPLICIT_SCALE_EXP.items():
            impl_lut[p] = v
        impl_a_mat = impl_lut[prec_A.astype(np.int64)]
        if (impl_a_mat < 0).any():
            bad = sorted(set(int(p) for p in np.unique(prec_A)
                             if int(p) not in IMPLICIT_SCALE_EXP))
            raise ValueError(
                f"prec_A array contains values not in {sorted(IMPLICIT_SCALE_EXP)}: {bad}"
            )
    else:
        if prec_A not in IMPLICIT_SCALE_EXP:
            raise ValueError(f"prec_A must be in {sorted(IMPLICIT_SCALE_EXP)} or ndarray, got {prec_A}")
        impl_a_scalar = IMPLICIT_SCALE_EXP[prec_A]

    if accum_dtype == "bf16":
        try:
            from ml_dtypes import bfloat16
        except ImportError as e:
            raise ImportError(
                "accum_dtype='bf16' requires ml_dtypes. Install with: "
                "pip install ml_dtypes  (or `pip install -e .[bf16]`)."
            ) from e
        # Mirror the HW RMW datapath, per K-block in order 0..n-1 (see spec D5):
        #   int32 block sum (exact) -> int->bf16 (RNE, = INToRecFN)
        #   -> single integer exponent shift 2^E (exact, = recoded exp-add)
        #   -> bf16 add (RNE, = AddRecFN).
        # E = (e_a - 127) + (e_b - 127) - impl_a - impl_b, per (m, n).
        ea = scale_A.astype(np.int64)          # (M, n_blocks) E8M0 biased exponent
        eb = scale_B.astype(np.int64)          # (n_blocks, N)
        C = np.zeros((M, N), dtype=bfloat16)
        for blk in range(n_blocks):
            sl = slice(blk * BLOCK_SIZE, (blk + 1) * BLOCK_SIZE)
            block_int = int_A[:, sl].astype(np.int32) @ int_B[sl, :].astype(np.int32)
            if prec_a_is_array:
                impl_a_here = impl_a_mat[:, blk][:, np.newaxis]        # (M, 1)
            else:
                impl_a_here = impl_a_scalar                            # scalar
            E = ea[:, blk:blk + 1] + eb[blk:blk + 1, :] - 254 - impl_a_here - impl_b
            # block_int magnitude < 2^20, so float32 is exact and the cast to bf16 is RNE.
            r = block_int.astype(np.float32).astype(bfloat16)
            # ldexp scales by 2^E in float64 (no premature underflow), then one bf16 round.
            fp_a = np.ldexp(r.astype(np.float64), E).astype(bfloat16)
            C = C + fp_a                                              # native bf16 add
            assert C.dtype == bfloat16, "bf16 accumulator was promoted to a wider dtype"
        return C.astype(np.float32)

    a_scale_fp = _e8m0_to_fp32(scale_A)
    b_scale_fp = _e8m0_to_fp32(scale_B)

    C = np.zeros((M, N), dtype=np.float32)
    for blk in range(n_blocks):
        sl = slice(blk * BLOCK_SIZE, (blk + 1) * BLOCK_SIZE)
        block_int = int_A[:, sl].astype(np.int32) @ int_B[sl, :].astype(np.int32)
        if prec_a_is_array:
            # Per-row implicit_scale for this block: (M,) float32 vector.
            # Broadcast over N via [:, np.newaxis] inside the multiply.
            implicit_scale = (
                2.0 ** -(impl_a_mat[:, blk].astype(np.float64) + np.float64(impl_b))
            ).astype(np.float32)
            block_fp = (
                block_int.astype(np.float32)
                * a_scale_fp[:, blk:blk + 1]
                * b_scale_fp[blk:blk + 1, :]
                * implicit_scale[:, np.newaxis]
            )
        else:
            implicit_scale = np.float32(2.0 ** -(impl_a_scalar + impl_b))
            block_fp = (
                block_int.astype(np.float32)
                * a_scale_fp[:, blk:blk + 1]
                * b_scale_fp[blk:blk + 1, :]
                * implicit_scale
            )
        C += block_fp
    return C
