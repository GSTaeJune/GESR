"""Tests for mxp_tools.compare — including C2 NaN/Inf handling."""
import numpy as np
import pytest

from mxp_tools.compare import diff_3way


def _zeros(shape=(4, 4)):
    return np.zeros(shape, dtype=np.float32)


def test_identical_matrices_zero_stats():
    C = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = diff_3way(C, C, C)
    for k in ("hw_sw", "sw_fp32", "hw_fp32"):
        s = result["stats"][k]
        assert s["max_abs_err"] == 0.0
        assert s["rmse"] == 0.0
        assert s["n_nonzero_diff"] == 0
        assert s["n_nan_diff"] == 0
        assert s["n_inf_diff"] == 0


def test_nan_in_hw_is_counted_as_mismatch():
    """C2 fix: was previously n_nonzero_diff=0 with NaN inputs."""
    C_ok = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    C_bad = C_ok.copy()
    C_bad[0, 0] = np.nan
    result = diff_3way(C_bad, C_ok, C_ok)
    hw_sw = result["stats"]["hw_sw"]
    assert hw_sw["n_nan_diff"] == 1
    assert hw_sw["n_nonzero_diff"] >= 1, "NaN must count as a mismatch"
    # Finite-only metrics still computable on the rest
    assert np.isfinite(hw_sw["rmse"])  # 3 finite zero diffs → rmse=0
    assert hw_sw["rmse"] == 0.0


def test_inf_in_hw_is_counted_as_mismatch():
    C_ok = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    C_bad = C_ok.copy()
    C_bad[1, 1] = np.inf
    result = diff_3way(C_bad, C_ok, C_ok)
    hw_sw = result["stats"]["hw_sw"]
    assert hw_sw["n_inf_diff"] == 1
    assert hw_sw["n_nonzero_diff"] >= 1


def test_all_nan_yields_nan_metrics_but_correct_count():
    C_ok = _zeros()
    C_bad = np.full_like(C_ok, np.nan)
    result = diff_3way(C_bad, C_ok, C_ok)
    hw_sw = result["stats"]["hw_sw"]
    assert hw_sw["n_nan_diff"] == C_ok.size
    assert hw_sw["n_nonzero_diff"] == C_ok.size
    assert np.isnan(hw_sw["max_abs_err"])  # no finite values to compute on


def test_shape_mismatch_raises():
    a = np.zeros((2, 2), dtype=np.float32)
    b = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        diff_3way(a, b, a)


def test_dtype_strict():
    a = np.zeros((2, 2), dtype=np.float64)
    b = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(TypeError):
        diff_3way(a, b, b)
