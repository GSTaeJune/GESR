"""MX format constants. Mirrors MXP/TransformSerial.py for bit-exact parity."""

INT2 = 2
INT4 = 4
INT8 = 8

PREC_NAMES = {INT8: "mxint8", INT4: "mxint4", INT2: "mxint2"}
ALL_PRECS = (INT8, INT4, INT2)

# OCP MX Spec §6.3 mantissa upscale exponent. quantize_block_mx multiplies by
# 2^IMPLICIT_SCALE_EXP[prec] before rounding; mxint_gemm_golden divides back.
IMPLICIT_SCALE_EXP = {INT8: 6, INT4: 2, INT2: 0}

# Symmetric saturation bound (leaves -2^(N-1) unused per MX spec).
MAX_INT = {INT8: 127, INT4: 7, INT2: 1}

BLOCK_SIZE = 32
