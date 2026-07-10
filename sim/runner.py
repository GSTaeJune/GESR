"""sim/runner.py - VSCode-friendly Python orchestrator.

Run with VSCode "Run Python File" (no args), or with subcommands:

  python sim/runner.py                                    # default mixed-one A=8 + viz
  python sim/runner.py mixed-one --A 8                    # random mixed
  python sim/runner.py mixed-one --A 8 --uniform 8        # uniform W
  python sim/runner.py mixed-one --A 8 --k-tile 8,4,2,8   # per-K-tile W
  python sim/runner.py integration-one --A 8 --B 8
  python sim/runner.py integration-sweep                  # 9-mode

This is a thin Python wrapper that *invokes the existing bash scripts*
(sim/run_mixed_one.sh, sim/run_integration_one.sh, sim/run_integration_sweep.sh)
- those scripts are battle-tested. The wrapper adds:
  - VSCode-button friendly defaults
  - automatic visualization (PNG heatmaps) after each compare
  - per-mode result collection for sweep summary

Why bash dispatch (not pure Python subprocess of xvlog/xelab/xsim):
  Vivado's xsim on Windows has fragile child-process semantics (xsim spawns
  xsimk.exe which loads DLLs from the Vivado bin dir; env propagation through
  Python -> bash -> cmd //c sometimes breaks DLL discovery and leaves
  orphaned xsimk.exe zombies that hold file locks). Calling the original
  .sh scripts directly via `bash` matches the proven invocation pattern.

Required on PATH:
  - `bash`  (Git for Windows: https://git-scm.com/download/win)
  - `python` (current interpreter is used for viz; Vivado tools are called
              from inside the .sh script, so they need to be on bash's PATH)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── shell wrapper ───────────────────────────────────────────────────────────

def _require_bash() -> str:
    bash = shutil.which("bash")
    if bash is None:
        raise FileNotFoundError(
            "`bash` not found on PATH. Install Git for Windows "
            "(https://git-scm.com/download/win) and re-open your terminal."
        )
    return bash


def _bash_run(script_path: str, *args: str, env_extra: dict | None = None,
              check: bool = True) -> int:
    """Invoke a bash script (relative to REPO_ROOT) with the given args.

    Inherits stdin/stdout/stderr and the parent's full env. env_extra values:
      - str           : set/override env var
      - None          : UNSET (remove from env); needed because empty-string
                        env var would still be "set" and may break downstream
                        (e.g. int("") in gen_mixed.py).
    """
    bash = _require_bash()
    argv = [bash, script_path, *args]
    env = os.environ.copy()
    if env_extra:
        for k, v in env_extra.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
    print(f"  $ bash {script_path} {' '.join(args)}", flush=True)
    result = subprocess.run(argv, cwd=str(REPO_ROOT), env=env)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, argv)
    return result.returncode


# ── visualization ───────────────────────────────────────────────────────────

def _viz_one(work_dir: Path, title: str) -> None:
    """Comprehensive 3-row figure: inputs / prec maps + HW-truth diff / stats.

    Loads inputs (W/A/prec_map) from sw_ref/inputs_*.npz (saved by gen_mixed
    or, for integration runs, reconstructed from MXP_Tools' ab_fp32 + quant
    npzs). Reassembles C_hw from bank files. Produces work_dir/result.png.

    Layout (3 rows x 3 cols):
      Row 1 - Inputs+Out : W (fp32), A (fp32), C_hw (HW output)
      Row 2 - Prec+Diff  : W_PREC_map, A_PREC_map, log10|C_hw - C_fp32|
                           <- W prec map (col 0) and HW-truth diff (col 2)
                              share this row's M-axis; the eye can scan a
                              single M-row left -> right to correlate weight
                              precision pattern with total HW error pattern.
      Row 3 - Stats text : PASS/FAIL + hw_sw / hw_fp32 max/rmse/snr summary

    C_sw and the sw_fp32 visualizations are intentionally omitted: when the
    bit-exact gate (hw_sw) passes, C_sw is identical to C_hw and |C_sw -
    C_fp32| identical to |C_hw - C_fp32|, so both would duplicate panels
    elsewhere in the figure.
    """
    sys.path.insert(0, str(REPO_ROOT / "MXP_Tools"))
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mxp_tools import compare as cmp_mod, hwio
    except ImportError as e:
        print(f"[viz] skipped (missing dep): {e}", flush=True)
        return

    sw_ref_dir = work_dir / "sw_ref"
    dump_dir = work_dir / "hw_out"

    # Load golden (C_sw, C_fp32).
    npzs = sorted(sw_ref_dir.glob("C_sw*.npz"),
                  key=lambda p: (0 if p.stem.endswith("_bf16") or p.stem == "C_sw_mixed" else 1))
    if not npzs:
        print(f"[viz] no npz under {sw_ref_dir} - skipping", flush=True)
        return
    ref = np.load(npzs[0])
    C_sw = ref["C_sw"].astype(np.float32)
    C_fp32 = ref["C_fp32"].astype(np.float32)
    M, N = C_sw.shape

    # Load HW dump.
    banks = [str(dump_dir / f"bank{i}.mem") for i in range(32)]
    if not all(Path(b).exists() for b in banks):
        print(f"[viz] missing bank files under {dump_dir} - skipping", flush=True)
        return
    accum = str(ref["accum"]) if "accum" in ref.files else "fp32"
    reader = (hwio.read_writememh_bf16 if accum == "bf16"
              else hwio.read_writememh_fp32)
    C_hw = hwio.gather_banks(banks, M, N, hwio.interleaved_row_major_32bank,
                             reader=reader)
    result = cmp_mod.diff_3way(C_hw, C_sw, C_fp32)

    # Load inputs. Try inputs_mixed.npz (gen_mixed.py), else ab_fp32.npz
    # (MXP_Tools gen for integration).
    inputs_mixed = sw_ref_dir / "inputs_mixed.npz"
    ab_fp32 = work_dir / "ab_fp32.npz"
    W_fp, A_fp = None, None
    W_prec_map, A_prec_scalar = None, None
    if inputs_mixed.exists():
        d = np.load(inputs_mixed)
        W_fp = d["W_fp"]
        A_fp = d["A_fp"]
        W_prec_map = d["prec_map"]                    # (M, K_T) per-block W_PREC
        A_prec_scalar = int(d["A_PREC"])              # layer-wide A_PREC
    elif ab_fp32.exists():
        d = np.load(ab_fp32)
        # In integration, gen names A=activation (analogous to W in mixed)
        # and B=weight. Use the same role split for the figure.
        W_fp = d["A"]   # activation (top operand)
        A_fp = d["B"]   # weight     (bottom operand)
        W_prec_map = None  # uniform precision, no per-block map

    # Cap M for A_prec_map vertical extent (== K dim in input A_fp).
    K_dim = A_fp.shape[0] if A_fp is not None else M
    K_T = K_dim // 32 if K_dim % 32 == 0 else 4  # mixed TB: K=128 -> K_T=4

    # 3-class colormap for precision maps (W=2 red, W=4 orange, W=8 blue).
    prec_cmap = matplotlib.colors.ListedColormap(["#d62728", "#ff7f0e", "#1f77b4"])
    prec_lookup = {2: 0, 4: 1, 8: 2}

    # Build the figure (3 row x 3 col + stats strip).
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.35], hspace=0.4, wspace=0.28)

    # Row 1: W input | A input | C_hw (HW output).
    if W_fp is not None:
        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(W_fp, aspect="auto", cmap="RdBu_r",
                       vmin=-np.abs(W_fp).max(), vmax=np.abs(W_fp).max())
        ax.set_title("Input: Weight matrix W (FP32)")
        ax.set_xlabel("K"); ax.set_ylabel("M")
        fig.colorbar(im, ax=ax, fraction=0.046)
    if A_fp is not None:
        ax = fig.add_subplot(gs[0, 1])
        im = ax.imshow(A_fp, aspect="auto", cmap="RdBu_r",
                       vmin=-np.abs(A_fp).max(), vmax=np.abs(A_fp).max())
        ax.set_title("Input: Activation matrix A (FP32)")
        ax.set_xlabel("N"); ax.set_ylabel("K")
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(C_hw, aspect="auto", cmap="viridis")
    ax.set_title("C_hw  (HW output)")
    ax.set_xlabel("N"); ax.set_ylabel("M")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: W_PREC map | A_PREC map | log10|C_hw - C_fp32|.
    # M-axis aligned across col 0 and col 2 -> visual correlation between
    # low-precision weight rows and high HW-vs-truth error rows.
    if W_prec_map is not None:
        ax = fig.add_subplot(gs[1, 0])
        img_w = np.vectorize(prec_lookup.get)(W_prec_map)
        im = ax.imshow(img_w, aspect="auto", cmap=prec_cmap, vmin=0, vmax=2)
        ax.set_title("W_PREC map (per M-row, per K-block)")
        ax.set_xlabel("K-block (0..3)"); ax.set_ylabel("M-row")
        ax.set_xticks(range(W_prec_map.shape[1]))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["W=2", "W=4", "W=8"])
    else:
        ax = fig.add_subplot(gs[1, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, "(uniform W precision\nno per-block map)",
                ha="center", va="center", fontsize=10, color="gray")

    if A_prec_scalar is not None:
        ax = fig.add_subplot(gs[1, 1])
        a_map = np.full((K_T, N), prec_lookup[A_prec_scalar], dtype=np.uint8)
        im = ax.imshow(a_map, aspect="auto", cmap=prec_cmap, vmin=0, vmax=2)
        ax.set_title(f"A_PREC map (layer-uniform: A={A_prec_scalar})")
        ax.set_xlabel("N"); ax.set_ylabel("K-block")
        ax.set_yticks(range(K_T))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["A=2", "A=4", "A=8"])
    else:
        ax = fig.add_subplot(gs[1, 1])
        ax.axis("off")
        ax.text(0.5, 0.5, "(integration mode\nno mixed A_PREC)",
                ha="center", va="center", fontsize=10, color="gray")

    ax = fig.add_subplot(gs[1, 2])
    d_hw_fp32 = np.abs(result["diff_hw_fp32"])
    im = ax.imshow(np.log10(np.maximum(d_hw_fp32, 1e-12)),
                   aspect="auto", cmap="magma")
    ax.set_title("log10 |C_hw - C_fp32|  (HW vs FP32 truth)")
    ax.set_xlabel("N"); ax.set_ylabel("M")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Row 3: Stats text (compact — sw_fp32 line dropped; redundant when
    # hw_sw passes, since C_sw == C_hw and diffs coincide).
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    s = result["stats"]
    hw_sw_pass = s["hw_sw"]["n_nonzero_diff"] == 0
    verdict = ("PASS  (bit-exact)" if hw_sw_pass
               else f"FAIL  (n_diff={s['hw_sw']['n_nonzero_diff']}/{s['hw_sw']['n_total']})")
    lines = [
        f"hw_sw  (HW vs SW golden)        : {verdict}",
        f"                                   max={s['hw_sw']['max_abs_err']:.3e}   rmse={s['hw_sw']['rmse']:.3e}   snr={s['hw_sw']['snr_db']:.2f} dB",
        f"hw_fp32 (HW vs FP32 truth) [tot] : max={s['hw_fp32']['max_abs_err']:.3e}   rmse={s['hw_fp32']['rmse']:.3e}   snr={s['hw_fp32']['snr_db']:.2f} dB",
    ]
    color = "green" if hw_sw_pass else "red"
    ax.text(0.02, 0.92, lines[0], transform=ax.transAxes, fontsize=12,
            color=color, family="monospace", va="top", weight="bold")
    ax.text(0.02, 0.55, "\n".join(lines[1:]), transform=ax.transAxes,
            fontsize=10, family="monospace", va="top")

    fig.suptitle(title, fontsize=14, y=0.995)
    savepath = work_dir / "result.png"
    fig.savefig(savepath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] full figure -> {savepath}", flush=True)


def _viz_sweep(passed_labels: list[str]) -> None:
    """Render the 9-mode summary grid (rmse). Collects results from
    work/<LABEL>/{sw_ref,hw_out} for each passed mode."""
    sys.path.insert(0, str(REPO_ROOT / "MXP_Tools"))
    try:
        import numpy as np
        from mxp_tools import viz, compare as cmp_mod, hwio
    except ImportError as e:
        print(f"[viz] sweep skipped (missing dep): {e}", flush=True)
        return

    per_mode = {}
    for label in passed_labels:
        # label format A{a}_B{b}; npz format C_sw_mxint{b}_mxint{a}_bf16.npz
        a_p = int(label.split("_")[0][1:])
        b_p = int(label.split("_")[1][1:])
        work = REPO_ROOT / "work" / label
        npz = work / "sw_ref" / f"C_sw_mxint{b_p}_mxint{a_p}_bf16.npz"
        if not npz.exists():
            continue
        ref = np.load(npz)
        C_sw = ref["C_sw"].astype(np.float32)
        C_fp32 = ref["C_fp32"].astype(np.float32)
        M, N = C_sw.shape
        banks = [str(work / "hw_out" / f"bank{i}.mem") for i in range(32)]
        if not all(Path(b).exists() for b in banks):
            continue
        accum = str(ref["accum"]) if "accum" in ref.files else "fp32"
        reader = (hwio.read_writememh_bf16 if accum == "bf16"
                  else hwio.read_writememh_fp32)
        C_hw = hwio.gather_banks(banks, M, N, hwio.interleaved_row_major_32bank,
                                 reader=reader)
        per_mode[f"mxint{a_p}_mxint{b_p}"] = cmp_mod.diff_3way(C_hw, C_sw, C_fp32)

    if not per_mode:
        print("[viz] sweep: no usable results - skipping", flush=True)
        return
    savepath = REPO_ROOT / "work" / "sweep_summary.png"
    viz.summary_dashboard(per_mode, savepath=str(savepath), metric="rmse")
    print(f"[viz] sweep summary -> {savepath}", flush=True)


# ── subcommands ─────────────────────────────────────────────────────────────

def cmd_mixed_one(args: argparse.Namespace) -> int:
    # Build env override. Setting a key to None tells _bash_run to UNSET (pop)
    # that var - NOT set it to empty string. Empty-string env var would still
    # be "set" and trip gen_mixed.py's int("") check.
    env_extra: dict = {}
    if args.uniform is not None:
        env_extra["MIXED_W_UNIFORM"] = str(args.uniform)
        env_extra["MIXED_W_K_TILE"] = None
    elif args.k_tile is not None:
        env_extra["MIXED_W_K_TILE"] = args.k_tile
        env_extra["MIXED_W_UNIFORM"] = None
    else:
        # Random mixed - explicitly unset both so leftover env doesn't taint.
        env_extra["MIXED_W_UNIFORM"] = None
        env_extra["MIXED_W_K_TILE"] = None

    print(f"\n===== mixed-one A_PREC={args.A} =====", flush=True)
    rc = _bash_run("sim/run_mixed_one.sh", str(args.A),
                   env_extra=env_extra, check=False)

    if args.viz:
        title_mode = (f"uniform W={args.uniform}" if args.uniform else
                      f"K-tile {args.k_tile}" if args.k_tile else "random")
        _viz_one(REPO_ROOT / "work" / f"mixed_A{args.A}",
                 title=f"mixed_A{args.A} ({title_mode}) "
                       f"{'PASS' if rc == 0 else 'FAIL'}")
    return rc


def cmd_integration_one(args: argparse.Namespace) -> int:
    label = args.label or f"A{args.A}_B{args.B}"

    # The bash integration-one script doesn't gen/emit/ref - it expects those
    # to be done first. Replicate the sweep's pre-step inline.
    if not args.skip_inputs:
        prep = f"""
set -e
cd MXP_Tools
python -m mxp_tools gen   --out ../work/{label} -M 128 -K 128 -N 128 --seed 0
python -m mxp_tools emit  --out ../work/{label}
python -m mxp_tools ref   --out ../work/{label} --prec-a {args.B} --prec-b {args.A} --accum bf16
"""
        bash = _require_bash()
        rc = subprocess.run([bash, "-c", prep], cwd=str(REPO_ROOT)).returncode
        if rc != 0:
            print(f"{label}: gen/emit/ref FAILED")
            return rc

    print(f"\n===== integration-one {label} =====", flush=True)
    rc = _bash_run("sim/run_integration_one.sh", label, str(args.A), str(args.B),
                   check=False)

    # run_integration_one.sh does NOT include the compare step (sweep does it
    # separately). Run compare here so the gate is uniform with mixed-one.
    if rc == 0:
        bash = _require_bash()
        banks = " ".join(f'../work/{label}/hw_out/bank{i}.mem' for i in range(32))
        compare_script = f"""
cd MXP_Tools
python -m mxp_tools compare \
    --ref ../work/{label}/sw_ref/C_sw_mxint{args.B}_mxint{args.A}_bf16.npz \
    --hw-banks {banks} \
    --layout interleaved_row_major_32bank
"""
        rc = subprocess.run([bash, "-c", compare_script], cwd=str(REPO_ROOT)).returncode

    print(f"\n{label}: {'PASS' if rc == 0 else 'FAIL'}")
    if args.viz:
        _viz_one(REPO_ROOT / "work" / label,
                 title=f"{label} {'PASS' if rc == 0 else 'FAIL'}")
    return rc


def cmd_mixed_sweep(args: argparse.Namespace) -> int:
    """Mixed-precision sweep: random per-(M,K-block) W_PREC × A_PREC ∈ {2,4,8}.

    This is the appropriate regression for the mixed TB - W is random per
    block by design, so the only thing that varies between modes is A_PREC.
    The 9-mode (A,B) ∈ {2,4,8}^2 integration-sweep tests UNIFORM precision
    combinations - a different TB and a different question.
    """
    passed: list[str] = []
    failed: list[str] = []

    for a in (2, 4, 8):
        label = f"mixed_A{a}"
        print(f"\n==================== {label} (random W per block) ====================",
              flush=True)
        try:
            ns = argparse.Namespace(
                A=a, uniform=None, k_tile=None, viz=args.viz,
            )
            rc = cmd_mixed_one(ns)
        except subprocess.CalledProcessError:
            print(f"{label}: pipeline FAILED")
            failed.append(label)
            continue

        (passed if rc == 0 else failed).append(label)

    print("\n====================================")
    print(f"MIXED SWEEP RESULT: {len(passed)}/3 PASS")
    print(f"PASSED: {' '.join(passed)}")
    if failed:
        print(f"FAILED: {' '.join(failed)}")
        return 1
    print("ALL 3 MIXED MODES PASSED")
    return 0


def cmd_integration_sweep(args: argparse.Namespace) -> int:
    """Invoke the existing 9-mode sweep script, then aggregate visuals."""
    print("\n===== integration-sweep (9 modes) =====", flush=True)
    rc = _bash_run("sim/run_integration_sweep.sh", check=False)
    # The sweep script prints PASSED: A2_B2 A2_B4 ... on success. Collect
    # the labels for viz regardless of overall pass/fail.
    passed = [f"A{a}_B{b}" for a in (2, 4, 8) for b in (2, 4, 8)
              if (REPO_ROOT / "work" / f"A{a}_B{b}" / "hw_out" / "bank0.mem").exists()]
    if args.viz and passed:
        _viz_sweep(passed)
    return rc


# ── argparse + VSCode run-button default ────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="sim/runner.py",
        description="VSCode-friendly Python orchestrator for gemm_sram sim.",
    )
    sp = p.add_subparsers(dest="cmd")

    g = sp.add_parser("mixed-one", help="single mixed-precision run")
    g.add_argument("--A", type=int, choices=(2, 4, 8), default=8)
    g.add_argument("--uniform", type=int, choices=(2, 4, 8), default=None,
                   help="force uniform W (overrides random mixed)")
    g.add_argument("--k-tile", type=str, default=None,
                   help="per-K-tile W, e.g. '8,4,2,8'")
    g.add_argument("--no-viz", dest="viz", action="store_false", default=True)
    g.set_defaults(func=cmd_mixed_one)

    g = sp.add_parser("integration-one", help="single uniform-precision run")
    g.add_argument("--label", default=None)
    g.add_argument("--A", type=int, choices=(2, 4, 8), required=True)
    g.add_argument("--B", type=int, choices=(2, 4, 8), required=True)
    g.add_argument("--skip-inputs", action="store_true")
    g.add_argument("--no-viz", dest="viz", action="store_false", default=True)
    g.set_defaults(func=cmd_integration_one)

    g = sp.add_parser("mixed-sweep",
                      help="3-mode mixed sweep: A in {2,4,8} x random W (mixed-prec TB)")
    g.add_argument("--no-viz", dest="viz", action="store_false", default=True)
    g.set_defaults(func=cmd_mixed_sweep)

    g = sp.add_parser("integration-sweep",
                      help="9-mode uniform sweep: (A,B) in {2,4,8}^2 (integration TB)")
    g.add_argument("--no-viz", dest="viz", action="store_false", default=True)
    g.set_defaults(func=cmd_integration_sweep)

    args = p.parse_args(argv)

    # VSCode "Run Python File" support: no subcommand -> mixed-one A=8 default.
    if args.cmd is None:
        print("[runner] no subcommand; defaulting to: mixed-one --A 8 (random mixed)")
        print("[runner] override: python sim/runner.py {mixed-one|integration-one|integration-sweep} ...")
        args = argparse.Namespace(
            cmd="mixed-one", A=8, uniform=None, k_tile=None,
            viz=True, func=cmd_mixed_one,
        )

    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
