"""FP32 → MX quantization. Bit-exact lift of MXP/TransformSerial.py §6.3.

Stricter than the original: rejects wrong dtype / non-divisible shapes loudly
instead of silently coercing or asserting.
"""
import numpy as np

from .const import BLOCK_SIZE, IMPLICIT_SCALE_EXP, MAX_INT


def quantize_block_mx(block_fp32, prec):
    """Quantize a 32-element FP32 block to (raw_int [int8], e8m0 scale [uint8]).

    Algorithm (OCP MX Spec §6.3):
      X         = floor_pow2(max|V|)
      e8m0      = clip(log2(X) + 127, 0, 254)
      raw_int   = clip(round(V / 2^(e8m0-127) * 2^IMPLICIT), ±MAX_INT)
    """
    if not isinstance(block_fp32, np.ndarray):
        raise TypeError(f"block_fp32 must be np.ndarray, got {type(block_fp32).__name__}")
    if block_fp32.dtype != np.float32:
        raise TypeError(f"block_fp32 must be float32, got {block_fp32.dtype}")
    if block_fp32.shape != (BLOCK_SIZE,):
        raise ValueError(f"block_fp32 must be shape ({BLOCK_SIZE},), got {block_fp32.shape}")
    if prec not in IMPLICIT_SCALE_EXP:
        raise ValueError(f"prec must be in {sorted(IMPLICIT_SCALE_EXP)}, got {prec}")
    if not np.isfinite(block_fp32).all():
        raise ValueError(
            "block_fp32 contains NaN or Inf — MX quant has no defined behavior "
            "for non-finite inputs. Filter or replace upstream."
        )

    max_int = MAX_INT[prec]
    implicit_inv = 1 << IMPLICIT_SCALE_EXP[prec]

    max_abs = float(np.max(np.abs(block_fp32)))
    if max_abs == 0:
        # OCP MX zero-block convention: emit e8m0=0 (biased exp 0 → 2^-127).
        # The int mantissa is also 0, so the product is exactly 0 regardless
        # of how HW dequantizes the scale. If your HW special-cases e8m0=0
        # as a "NaN scale" sentinel, dequant will still yield 0 because the
        # mantissa is 0 — but document the assumption with your HW vendor.
        return np.zeros(BLOCK_SIZE, dtype=np.int8), np.uint8(0)

    floor_log2 = int(np.floor(np.log2(max_abs)))
    e8m0 = int(np.clip(floor_log2 + 127, 0, 254))
    scale = 2.0 ** (e8m0 - 127)

    # Rounding: np.round uses banker's rounding (round-half-to-even, IEEE-754
    # default). OCP MX Spec leaves rounding mode unspecified; if your HW uses
    # round-half-away-from-zero instead, expect ±1 ULP mismatch at the *.5
    # boundary in C_hw - C_sw. Confirm with HW vendor.
    raw = np.round(block_fp32.astype(np.float64) / scale * implicit_inv).astype(np.int32)
    raw = np.clip(raw, -max_int, max_int).astype(np.int8)
    return raw, np.uint8(e8m0)


def quantize_matrix_mx(matrix_fp32, prec, block_axis):
    """Block-quantize along `block_axis` (1 for A's K axis, 0 for B's K axis).

    Matrix size along block_axis must be a multiple of 32 (no auto-padding).

    Returns:
      int_matrix:   int8,  same shape as input
      scale_matrix: uint8, one E8M0 per 32-element block
                    block_axis=1 → shape (rows, n_blocks)
                    block_axis=0 → shape (n_blocks, cols)
    """
    if not isinstance(matrix_fp32, np.ndarray):
        raise TypeError(f"matrix_fp32 must be np.ndarray, got {type(matrix_fp32).__name__}")
    if matrix_fp32.dtype != np.float32:
        raise TypeError(f"matrix_fp32 must be float32, got {matrix_fp32.dtype}")
    if matrix_fp32.ndim != 2:
        raise ValueError(f"matrix_fp32 must be 2-D, got {matrix_fp32.ndim}-D")
    if block_axis not in (0, 1):
        raise ValueError(f"block_axis must be 0 or 1, got {block_axis}")

    block_dim = matrix_fp32.shape[block_axis]
    if block_dim % BLOCK_SIZE != 0:
        raise ValueError(
            f"axis {block_axis} length {block_dim} must be a multiple of {BLOCK_SIZE}"
        )
    n_blocks = block_dim // BLOCK_SIZE
    int_matrix = np.zeros(matrix_fp32.shape, dtype=np.int8)

    if block_axis == 1:
        rows, _ = matrix_fp32.shape
        scale_matrix = np.zeros((rows, n_blocks), dtype=np.uint8)
        for i in range(rows):
            for b in range(n_blocks):
                sl = slice(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)
                q, s = quantize_block_mx(matrix_fp32[i, sl], prec)
                int_matrix[i, sl] = q
                scale_matrix[i, b] = s
    else:
        _, cols = matrix_fp32.shape
        scale_matrix = np.zeros((n_blocks, cols), dtype=np.uint8)
        for j in range(cols):
            for b in range(n_blocks):
                sl = slice(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)
                q, s = quantize_block_mx(matrix_fp32[sl, j], prec)
                int_matrix[sl, j] = q
                scale_matrix[b, j] = s

    return int_matrix, scale_matrix
