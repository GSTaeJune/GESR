"""mxp_tools — HW verification helper for the MXP FPGA accelerator.

Reads HW $writememh FP32 outputs, computes the SW golden reference + FP32
truth, and visualizes the 3-way diff. Bit-exact-aligned with the MX format
used by MXP/TransformSerial.py.
"""
from .const import ALL_PRECS, BLOCK_SIZE, IMPLICIT_SCALE_EXP, INT2, INT4, INT8, MAX_INT, PREC_NAMES
from .quant import quantize_block_mx, quantize_matrix_mx
from .gemm import mxint_gemm_golden
from . import compare, hwio, viz

__all__ = [
    "INT2", "INT4", "INT8",
    "ALL_PRECS", "BLOCK_SIZE", "PREC_NAMES", "IMPLICIT_SCALE_EXP", "MAX_INT",
    "quantize_block_mx", "quantize_matrix_mx",
    "mxint_gemm_golden",
    "compare", "hwio", "viz",
]
