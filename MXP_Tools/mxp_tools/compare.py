"""3-way FP32 comparison: HW vs SW golden vs FP32 truth.

NaN/Inf policy: any non-finite difference is counted as a mismatch. The
returned `stats` dict carries finite-only summary metrics (max/mean/RMSE/SNR)
plus explicit `n_nan_diff` and `n_inf_diff` counters so downstream auto
pass/fail checks can fail loud rather than reading `nan` and shrugging.
"""
import numpy as np


def _pair_stats(x, y):
    """Per-pair stats: max/mean abs err, RMSE, SNR (dB), mismatch counts.

    `n_nonzero_diff` includes any |diff| > 0 *or* non-finite (NaN/Inf), so a
    fully-NaN HW output cannot look like "0 mismatches". Finite metrics are
    computed on the finite-only mask so they remain meaningful when the
    array contains a few non-finite values.
    """
    diff = (x.astype(np.float64) - y.astype(np.float64))
    abs_diff = np.abs(diff)

    nan_mask = np.isnan(diff)
    inf_mask = np.isinf(diff)
    bad_mask = nan_mask | inf_mask
    finite_mask = ~bad_mask
    n_nan = int(nan_mask.sum())
    n_inf = int(inf_mask.sum())

    if finite_mask.any():
        finite_diff = diff[finite_mask]
        finite_abs = abs_diff[finite_mask]
        finite_y = y.astype(np.float64)[finite_mask]
        max_err = float(finite_abs.max())
        mean_err = float(finite_abs.mean())
        rmse = float(np.sqrt((finite_diff ** 2).mean()))
        sig = float(np.sum(finite_y ** 2))
        noise = float(np.sum(finite_diff ** 2))
        if noise == 0:
            snr_db = float("inf")
        elif sig == 0:
            snr_db = float("-inf")
        else:
            snr_db = 10.0 * np.log10(sig / noise)
    else:
        max_err = float("nan")
        mean_err = float("nan")
        rmse = float("nan")
        snr_db = float("nan")

    # n_nonzero_diff: any finite-nonzero OR any non-finite. The "abs > 0"
    # comparison returns False for NaN, so we OR in the bad_mask explicitly.
    n_nonzero_diff = int(((abs_diff > 0) | bad_mask).sum())

    return {
        "max_abs_err": max_err,
        "mean_abs_err": mean_err,
        "rmse": rmse,
        "snr_db": snr_db,
        "n_nonzero_diff": n_nonzero_diff,
        "n_nan_diff": n_nan,
        "n_inf_diff": n_inf,
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
    """Pretty-print the stats dict. Shows NaN/Inf counters when present."""
    s = result["stats"]
    if label:
        print(f"── {label} ──")
    header = f"{'pair':<12}{'max':>14}{'rmse':>14}{'mean_abs':>14}{'snr_dB':>12}{'n_diff':>10}"
    print(header)
    for k in ("hw_sw", "sw_fp32", "hw_fp32"):
        p = s[k]
        snr = f"{p['snr_db']:>11.2f}" if np.isfinite(p["snr_db"]) else f"{str(p['snr_db']):>11}"
        max_s = f"{p['max_abs_err']:>14.3e}" if np.isfinite(p["max_abs_err"]) else f"{'nan':>14}"
        rmse_s = f"{p['rmse']:>14.3e}" if np.isfinite(p["rmse"]) else f"{'nan':>14}"
        mean_s = f"{p['mean_abs_err']:>14.3e}" if np.isfinite(p["mean_abs_err"]) else f"{'nan':>14}"
        print(f"{k:<12}{max_s}{rmse_s}{mean_s}{snr}{p['n_nonzero_diff']:>10d}")
        if p["n_nan_diff"] or p["n_inf_diff"]:
            print(f"{'':<12}  ⚠ non-finite: {p['n_nan_diff']} NaN, {p['n_inf_diff']} Inf")
    # Greppable single-line summary for sweep logs.
    hw_sw = s["hw_sw"]
    verdict = "PASS" if hw_sw["n_nonzero_diff"] == 0 else "FAIL"
    print(f"summary: hw_sw {verdict}  n_diff={hw_sw['n_nonzero_diff']}/{hw_sw['n_total']}")


def bf16_accuracy_stats(C_sw, C_fp32):
    """Informational: bf16 golden vs FP32 truth. Reuses _pair_stats.

    Returns the _pair_stats dict (snr_db, rmse, ...). NOT a gate - the compare
    pass/fail gate stays hw_sw-only. `catastrophic` flags negative-dB SNR so the
    fp32-fallback decision (spec D4) is actionable rather than buried.
    """
    stats = _pair_stats(C_sw.astype(np.float32), C_fp32.astype(np.float32))
    stats["catastrophic"] = bool(np.isfinite(stats["snr_db"]) and stats["snr_db"] < 0.0)
    return stats
