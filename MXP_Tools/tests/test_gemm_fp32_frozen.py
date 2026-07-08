"""FP32 golden must stay byte-identical across the accum_dtype refactor."""
import os
import numpy as np

from mxp_tools.const import INT8
from mxp_tools.gemm import mxint_gemm_golden

_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "golden_fp32_frozen.npz")


def test_fp32_path_bit_identical_to_frozen():
    with np.load(_FIX) as d:
        C = mxint_gemm_golden(d["iA"], d["sA"], INT8, d["iB"], d["sB"], INT8, accum_dtype="fp32")
        assert np.array_equal(C, d["C"]), "fp32 accum path drifted from frozen reference"
        assert C.dtype == np.float32, f"fp32 path must return float32, got {C.dtype}"


def test_fp32_is_default():
    with np.load(_FIX) as d:
        C_default = mxint_gemm_golden(d["iA"], d["sA"], INT8, d["iB"], d["sB"], INT8)
        assert np.array_equal(C_default, d["C"]), "default accum_dtype must be fp32"
        assert C_default.dtype == np.float32, f"default path must return float32, got {C_default.dtype}"
