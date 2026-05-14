"""3-way FP32 comparison: HW vs SW golden vs FP32 truth."""
import numpy as np


def _pair_stats(x, y):
    """Per-pair stats: max abs err, mean abs err, RMSE, SNR (dB), mismatch count."""
    diff = (x - y).astype(np.float64)
    abs_diff = np.abs(diff)
    sig = np.sum(y.astype(np.float64) ** 2)
    noise = np.sum(diff ** 2)
    if noise == 0:
        snr_db = float("inf")
    elif sig == 0:
        snr_db = float("-inf")
    else:
        snr_db = 10.0 * np.log10(sig / noise)
    return {
        "max_abs_err": float(abs_diff.max()),
        "mean_abs_err": float(abs_diff.mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "snr_db": snr_db,
        "n_nonzero_diff": int((abs_diff > 0).sum()),
        "n_total": int(x.size),
    }


def diff_3way(C_hw, C_sw, C_fp32):
    """Compute pairwise diffs + stats.

    Returns a dict:
      {
        "C_hw", "C_sw", "C_fp32"   : the input matrices,
        "diff_hw_sw"               : C_hw - C_sw,
        "diff_sw_fp32"             : C_sw - C_fp32,
        "diff_hw_fp32"             : C_hw - C_fp32,
        "stats": {
            "hw_sw":    {...},   # HW correctness — should be ~0 if bit-correct
            "sw_fp32":  {...},   # pure quantization error
            "hw_fp32":  {...},   # total observed error
        },
      }
    """
    for name, m in (("C_hw", C_hw), ("C_sw", C_sw), ("C_fp32", C_fp32)):
        if m.dtype != np.float32:
            raise TypeError(f"{name} must be float32, got {m.dtype}")
    if not (C_hw.shape == C_sw.shape == C_fp32.shape):
        raise ValueError(
            f"shape mismatch: C_hw={C_hw.shape}, C_sw={C_sw.shape}, C_fp32={C_fp32.shape}"
        )

    return {
        "C_hw": C_hw, "C_sw": C_sw, "C_fp32": C_fp32,
        "diff_hw_sw": C_hw - C_sw,
        "diff_sw_fp32": C_sw - C_fp32,
        "diff_hw_fp32": C_hw - C_fp32,
        "stats": {
            "hw_sw":   _pair_stats(C_hw, C_sw),
            "sw_fp32": _pair_stats(C_sw, C_fp32),
            "hw_fp32": _pair_stats(C_hw, C_fp32),
        },
    }


def print_stats(result, label=""):
    """Pretty-print the stats dict."""
    s = result["stats"]
    if label:
        print(f"── {label} ──")
    print(f"{'pair':<12}{'max':>14}{'rmse':>14}{'mean_abs':>14}{'snr_dB':>12}{'n_diff':>10}")
    for k in ("hw_sw", "sw_fp32", "hw_fp32"):
        p = s[k]
        snr = f"{p['snr_db']:>11.2f}" if np.isfinite(p["snr_db"]) else f"{str(p['snr_db']):>11}"
        print(
            f"{k:<12}{p['max_abs_err']:>14.3e}{p['rmse']:>14.3e}"
            f"{p['mean_abs_err']:>14.3e}{snr}{p['n_nonzero_diff']:>10d}"
        )
