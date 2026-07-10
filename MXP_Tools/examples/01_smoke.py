"""End-to-end smoke test — runs the full pipeline using fake HW output.

Usage:  python examples/01_smoke.py [work_dir]

The fake "HW" is just C_sw with a tiny FP32 round-off perturbation, written
out as $writememh. This verifies the SW chain end-to-end without needing a
real FPGA in the loop.

Steps:
  1. gen   → FP32 A, B (small: 64x64x64)
  2. emit  → quantized .hex (just MXINT8 for speed)
  3. ref   → C_sw_mxint8_mxint8.npz
  4. fake  → C_sw → IEEE-754 hex → write_writememh   ← stand-in for FPGA
  5. read  → gather_banks → C_hw
  6. diff  → 3-way stats (HW vs SW should be ~0)
  7. viz   → heatmap PNG + 9-mode summary PNG
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mxp_tools import compare as cmp_mod
from mxp_tools import gemm, hwio, quant, viz
from mxp_tools.cli import main as cli_main


def run_cli(argv):
    """cli.main() exits via sys.exit even on success (exit-code gate for
    scripts). Swallow SystemExit(0) so this in-process step chain continues;
    re-raise on nonzero so a real failure still kills the smoke."""
    try:
        cli_main(argv)
    except SystemExit as e:
        if (e.code or 0) != 0:
            raise


def run(work):
    os.makedirs(work, exist_ok=True)

    print("== step 1/7 gen ==")
    run_cli(["gen", "--out", work, "-M", "64", "-K", "64", "-N", "64", "--seed", "0"])

    print("\n== step 2/7 emit (MXINT8 only) ==")
    run_cli(["emit", "--out", work, "--prec", "8"])

    print("\n== step 3/7 ref (8x8 only) ==")
    run_cli(["ref", "--out", work, "--prec-a", "8", "--prec-b", "8"])

    print("\n== step 4/7 fake HW = C_sw written as $writememh ==")
    ref = np.load(os.path.join(work, "sw_ref", "C_sw_mxint8_mxint8.npz"))
    C_sw = ref["C_sw"].astype(np.float32)
    fake_dir = os.path.join(work, "hw_out")
    os.makedirs(fake_dir, exist_ok=True)
    bank0 = os.path.join(fake_dir, "bank0.mem")
    hwio.write_writememh(bank0, hwio.fp32_to_hex_words(C_sw), addr_every=16)
    print(f"   wrote {bank0}  ({C_sw.size} words)")

    print("\n== step 5/7 compare (HW vs SW vs FP32) ==")
    run_cli([
        "compare",
        "--ref", os.path.join(work, "sw_ref", "C_sw_mxint8_mxint8.npz"),
        "--hw-banks", bank0,
        "--layout", "single",
        "--save", os.path.join(work, "compare_8x8.npz"),
    ])

    print("\n== step 6/7 sanity: HW == SW exactly (round-trip)? ==")
    data = np.load(os.path.join(work, "compare_8x8.npz"))
    max_hw_sw = float(np.abs(data["diff_hw_sw"]).max())
    print(f"   max |HW - SW| = {max_hw_sw}")
    assert max_hw_sw == 0.0, f"round-trip should be exact, got {max_hw_sw}"

    print("\n== step 7a/7 viz heatmap ==")
    run_cli([
        "viz",
        "--npz", os.path.join(work, "compare_8x8.npz"),
        "--save", os.path.join(work, "heatmap_8x8.png"),
        "--title", "MXINT8 × MXINT8, fake-HW smoke",
    ])

    print("\n== step 7b/7 build 9-mode summary using C_sw as fake HW ==")
    results = {}
    data_ab = np.load(os.path.join(work, "ab_fp32.npz"))
    A, B = data_ab["A"], data_ab["B"]
    C_fp32 = data_ab["C_fp32"].astype(np.float32)
    for pa in (8, 4, 2):
        for pb in (8, 4, 2):
            ai, asc = quant.quantize_matrix_mx(A, pa, block_axis=1)
            bi, bsc = quant.quantize_matrix_mx(B, pb, block_axis=0)
            C_sw_full = gemm.mxint_gemm_golden(ai, asc, pa, bi, bsc, pb)
            C_hw_fake = C_sw_full.copy()         # fake HW = SW (bit-correct)
            results[f"mxint{pa}_mxint{pb}"] = cmp_mod.diff_3way(C_hw_fake, C_sw_full, C_fp32)
    fig_path = os.path.join(work, "summary.png")
    viz.summary_dashboard(results, savepath=fig_path, metric="rmse")
    print(f"   summary → {fig_path}")

    print("\n[OK] smoke test PASS")


if __name__ == "__main__":
    work = sys.argv[1] if len(sys.argv) > 1 else "work_smoke"
    run(work)
