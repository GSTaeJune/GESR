"""Visualization — heatmap of 3 matrices + 3 diffs, and a per-mode summary."""
import numpy as np


def _lazy_mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "viz needs matplotlib. Install with `pip install matplotlib`."
        ) from e
    return plt


def heatmap_3way(result, savepath=None, title=None, log_diff=True):
    """2×3 grid: top row C_hw / C_sw / C_fp32, bottom row pairwise diffs.

    Args:
      result    : dict from compare.diff_3way
      savepath  : if given, save PNG; else show inline (notebook) or open window.
      title     : suptitle (e.g., "MXINT8 × MXINT4, seed=0").
      log_diff  : if True, plot |diff| on log10 scale (clipped at 1e-12).
    """
    plt = _lazy_mpl()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    mats = [result["C_hw"], result["C_sw"], result["C_fp32"]]
    names = ["C_hw", "C_sw", "C_fp32"]
    vmin = min(m.min() for m in mats)
    vmax = max(m.max() for m in mats)
    for ax, m, name in zip(axes[0], mats, names):
        im = ax.imshow(m, vmin=vmin, vmax=vmax, aspect="auto", cmap="viridis")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)

    diffs = [
        ("|C_hw − C_sw|",   np.abs(result["diff_hw_sw"])),
        ("|C_sw − C_fp32|", np.abs(result["diff_sw_fp32"])),
        ("|C_hw − C_fp32|", np.abs(result["diff_hw_fp32"])),
    ]
    if log_diff:
        diffs = [(n, np.log10(np.maximum(d, 1e-12))) for n, d in diffs]
    dvmin = min(d.min() for _, d in diffs)
    dvmax = max(d.max() for _, d in diffs)
    for ax, (name, d) in zip(axes[1], diffs):
        im = ax.imshow(d, vmin=dvmin, vmax=dvmax, aspect="auto", cmap="magma")
        ax.set_title("log10 " + name if log_diff else name)
        fig.colorbar(im, ax=ax, fraction=0.046)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def summary_dashboard(results_per_mode, savepath=None, metric="rmse"):
    """One table-like figure across modes. Rows = a_prec, cols = b_prec.

    `results_per_mode` keys: "mxint{a}_mxint{b}" → dict from diff_3way.
    `metric` ∈ {"rmse", "max_abs_err", "snr_db"}.
    """
    plt = _lazy_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    precs = (8, 4, 2)
    pair_keys = ("hw_sw", "sw_fp32", "hw_fp32")
    pair_titles = ("HW − SW (correctness)", "SW − FP32 (quant err)", "HW − FP32 (total)")
    for ax, pkey, ptitle in zip(axes, pair_keys, pair_titles):
        # raw grid keeps inf for label display; plot_grid masks inf to NaN
        # so imshow's colormap doesn't saturate the whole panel on one cell.
        raw_grid = np.full((len(precs), len(precs)), np.nan, dtype=np.float64)
        for i, pa in enumerate(precs):
            for j, pb in enumerate(precs):
                key = f"mxint{pa}_mxint{pb}"
                if key in results_per_mode:
                    raw_grid[i, j] = results_per_mode[key]["stats"][pkey][metric]
        plot_grid = raw_grid.copy()
        plot_grid[~np.isfinite(plot_grid)] = np.nan

        im = ax.imshow(plot_grid, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(precs)), [f"B={p}" for p in precs])
        ax.set_yticks(range(len(precs)), [f"A={p}" for p in precs])
        ax.set_title(f"{ptitle}\n{metric}")
        finite_mean = np.nanmean(plot_grid) if np.isfinite(plot_grid).any() else 0.0
        for i in range(len(precs)):
            for j in range(len(precs)):
                v = raw_grid[i, j]
                if np.isnan(v):
                    txt = "—"
                elif np.isposinf(v):
                    txt = "+inf"
                elif np.isneginf(v):
                    txt = "-inf"
                elif abs(v) < 1e-1 or abs(v) >= 1e3:
                    txt = f"{v:.2e}"
                else:
                    txt = f"{v:.3f}"
                pv = plot_grid[i, j]
                color = "white" if np.isfinite(pv) and pv > finite_mean else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"MXP 9-mode summary — {metric}")
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig
