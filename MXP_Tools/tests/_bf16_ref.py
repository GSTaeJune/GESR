"""Independent fp32->bf16 rounding reference (pure bit manipulation, no ml_dtypes).

Used only by tests, to anchor ml_dtypes' bf16 rounding to the IEEE round-half-to-
even spec. bf16 shares fp32's 8-bit exponent+bias, so a bf16 value's bit pattern
is exactly the top 16 bits of the corresponding fp32 pattern; rounding is RNE on
the discarded low 16 bits. This also makes bf16 subnormals fp32-representable, so
the same top-16 + RNE rule handles the subnormal range.
"""
import numpy as np


def f32_to_bf16_bits_rne(x_f32):
    """Round one np.float32 scalar to bf16; return the 16-bit pattern as int."""
    b = int(np.float32(x_f32).view(np.uint32))
    exp = (b >> 23) & 0xFF
    mant = b & 0x7FFFFF
    top = b >> 16
    if exp == 0xFF:                       # inf / NaN
        if mant != 0:
            return (top & 0x8000) | 0x7FC0   # NaN -> canonical quiet NaN (matches ml_dtypes)
        return top & 0xFFFF                  # +/- inf
    lsb = (b >> 16) & 1
    round_bit = (b >> 15) & 1
    sticky = (b & 0x7FFF) != 0
    if round_bit and (sticky or lsb):     # round-half-to-even; carry may reach exp/inf
        top += 1
    return top & 0xFFFF


def ml_bf16_bits(x_f32, bfloat16):
    """The bit pattern ml_dtypes produces for the same fp32 scalar (for comparison)."""
    return int(np.float32(x_f32).astype(bfloat16).view(np.uint16))
