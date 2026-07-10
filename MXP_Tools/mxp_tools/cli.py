"""CLI entry point: `python -m mxp_tools <command> ...`

Commands:
  gen      — generate FP32 A, B (and FP32 truth C) → .npz
  emit     — quantize + write HW input files ($readmemh-compatible)
  ref      — compute SW golden C_sw (and FP32 truth) from the saved A, B
  compare  — read HW output + SW golden → 3-way diff stats + .npz dump
  viz      — heatmap + summary dashboard from a compare .npz
  run      — gen → emit → (user runs HW) → compare → viz, end-to-end

Use `python -m mxp_tools <command> -h` for per-command flags.
"""
import argparse
import os
import sys

import numpy as np

from . import compare as cmp_mod
from . import gemm, hwio, quant, viz
from .const import ALL_PRECS, PREC_NAMES


def _add_io(p):
    p.add_argument("--out", required=True, help="output directory")


def _add_shape(p):
    p.add_argument("-M", type=int, default=128)
    p.add_argument("-K", type=int, default=128)
    p.add_argument("-N", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)


def _add_prec_pair(p):
    p.add_argument("--prec-a", type=int, choices=ALL_PRECS, required=True)
    p.add_argument("--prec-b", type=int, choices=ALL_PRECS, required=True)


# ─── commands ─────────────────────────────────────────────────────────────

def cmd_gen(args):
    rng = np.random.default_rng(args.seed)
    A = rng.standard_normal((args.M, args.K)).astype(np.float32)
    B = rng.standard_normal((args.K, args.N)).astype(np.float32)
    C_fp32 = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "ab_fp32.npz"), A=A, B=B, C_fp32=C_fp32, seed=args.seed)
    print(f"gen: wrote ab_fp32.npz  (A={A.shape}, B={B.shape}, seed={args.seed})")


def cmd_emit(args):
    data = np.load(os.path.join(args.out, "ab_fp32.npz"))
    A, B = data["A"], data["B"]
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A, B must be 2-D; got A.ndim={A.ndim}, B.ndim={B.ndim}")
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"shape mismatch in ab_fp32.npz: A.shape[1]={A.shape[1]} != "
            f"B.shape[0]={B.shape[0]}. C_fp32 in this npz won't match the "
            f"emitted multiply."
        )
    M, K = A.shape
    _, N = B.shape

    # pad K to multiple of BLOCK_SIZE
    from .const import BLOCK_SIZE
    K_pad = ((K + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    M_pad = ((M + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    N_pad = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    A_pad = np.zeros((M_pad, K_pad), dtype=np.float32); A_pad[:M, :K] = A
    B_pad = np.zeros((K_pad, N_pad), dtype=np.float32); B_pad[:K, :N] = B

    sub = os.path.join(args.out, "hw_input")
    os.makedirs(sub, exist_ok=True)

    precs = ALL_PRECS if args.prec is None else (args.prec,)
    for prec in precs:
        name = PREC_NAMES[prec]
        a_int, a_sc = quant.quantize_matrix_mx(A_pad, prec, block_axis=1)
        b_int, b_sc = quant.quantize_matrix_mx(B_pad, prec, block_axis=0)

        hwio.emit_a_input_bs(os.path.join(sub, f"a_input_BS_{name}.hex"), a_int, prec)
        hwio.emit_a_input_bp(os.path.join(sub, f"a_input_BP_{name}.hex"), a_int)
        hwio.emit_b_input(os.path.join(sub, f"b_input_{name}.hex"), b_int)
        hwio.emit_a_scale(os.path.join(sub, f"a_scale_{name}.hex"), a_sc)
        hwio.emit_b_scale(os.path.join(sub, f"b_scale_{name}.hex"), b_sc)
        np.savez(
            os.path.join(sub, f"quant_{name}.npz"),
            a_int=a_int, a_scale=a_sc, b_int=b_int, b_scale=b_sc, prec=prec,
        )
        print(f"emit: {name}  ->  {sub}/ (a_input_BS/BP, b_input, a/b_scale, quant_{name}.npz)")


def cmd_ref(args):
    sub = os.path.join(args.out, "hw_input")
    out_ref = os.path.join(args.out, "sw_ref")
    os.makedirs(out_ref, exist_ok=True)
    data = np.load(os.path.join(args.out, "ab_fp32.npz"))
    C_fp32 = data["C_fp32"]
    M, N = C_fp32.shape

    pa_list = ALL_PRECS if args.prec_a is None else (args.prec_a,)
    pb_list = ALL_PRECS if args.prec_b is None else (args.prec_b,)
    for pa in pa_list:
        for pb in pb_list:
            qa = np.load(os.path.join(sub, f"quant_{PREC_NAMES[pa]}.npz"))
            qb = np.load(os.path.join(sub, f"quant_{PREC_NAMES[pb]}.npz"))
            C_sw_pad = gemm.mxint_gemm_golden(
                qa["a_int"], qa["a_scale"], pa,
                qb["b_int"], qb["b_scale"], pb,
                accum_dtype=args.accum,
            )
            C_sw = C_sw_pad[:M, :N]
            suffix = "" if args.accum == "fp32" else f"_{args.accum}"
            path = os.path.join(out_ref, f"C_sw_{PREC_NAMES[pa]}_{PREC_NAMES[pb]}{suffix}.npz")
            np.savez(path, C_sw=C_sw, C_fp32=C_fp32, prec_a=pa, prec_b=pb, accum=args.accum)
            print(f"ref: {PREC_NAMES[pa]} x {PREC_NAMES[pb]} [{args.accum}]  ->  {path}")
            if args.accum == "bf16":
                s = cmp_mod.bf16_accuracy_stats(C_sw, C_fp32)
                print(f"  bf16 vs fp32-truth: SNR={s['snr_db']:.2f} dB  rmse={s['rmse']:.3e}")
                if s["catastrophic"]:
                    print("  WARNING: bf16 SNR < 0 dB - accumulation may be unusable; "
                          "consider the fp32 fallback (spec D4).")


def _resolve_mapping(args, M, N):
    if args.layout == "single":
        return hwio.default_single_bank_row_major
    if args.layout == "split-rows":
        return hwio.default_banks_split_rows(args.n_banks)
    if args.layout == "interleaved_row_major_16bank":
        return hwio.interleaved_row_major_16bank
    if args.layout == "interleaved_row_major_32bank":
        return hwio.interleaved_row_major_32bank
    raise ValueError(f"unknown layout {args.layout}")


def cmd_compare(args):
    ref_npz = np.load(args.ref)
    C_sw = ref_npz["C_sw"].astype(np.float32)
    C_fp32 = ref_npz["C_fp32"].astype(np.float32)
    M, N = C_sw.shape

    mapping = _resolve_mapping(args, M, N)
    # HW dump word format follows the golden's accumulator dtype, recorded in
    # the ref npz by `ref --accum` (absent in pre-bf16 npz files -> fp32).
    accum = str(ref_npz["accum"]) if "accum" in ref_npz.files else "fp32"
    reader = (hwio.read_writememh_bf16 if accum == "bf16"
              else hwio.read_writememh_fp32)
    C_hw = hwio.gather_banks(args.hw_banks, M, N, mapping, reader=reader)
    result = cmp_mod.diff_3way(C_hw, C_sw, C_fp32)
    cmp_mod.print_stats(result, label=os.path.basename(args.ref))

    if args.save:
        np.savez(
            args.save,
            C_hw=C_hw, C_sw=C_sw, C_fp32=C_fp32,
            diff_hw_sw=result["diff_hw_sw"],
            diff_sw_fp32=result["diff_sw_fp32"],
            diff_hw_fp32=result["diff_hw_fp32"],
        )
        print(f"compare: dumped → {args.save}")

    # Bit-exact PASS/FAIL gate.
    # Only hw_sw is required to be zero for "HW is bit-correct": sw_fp32 carries
    # pure quantization error (always nonzero), and hw_fp32 mixes the two. A
    # script-level fail-gate must therefore look at hw_sw.n_nonzero_diff only.
    # NaN/Inf in HW dump are already folded into n_nonzero_diff by _pair_stats.
    n_diff = result["stats"]["hw_sw"]["n_nonzero_diff"]
    if n_diff > 0:
        print(f"compare: FAIL - hw_sw n_nonzero_diff={n_diff} (of "
              f"{result['stats']['hw_sw']['n_total']})")
        return 1
    print(f"compare: PASS - hw_sw bit-exact ({result['stats']['hw_sw']['n_total']} elements)")
    return 0


def cmd_viz(args):
    data = np.load(args.npz)
    result = {
        "C_hw": data["C_hw"], "C_sw": data["C_sw"], "C_fp32": data["C_fp32"],
        "diff_hw_sw": data["diff_hw_sw"],
        "diff_sw_fp32": data["diff_sw_fp32"],
        "diff_hw_fp32": data["diff_hw_fp32"],
        "stats": cmp_mod.diff_3way(data["C_hw"], data["C_sw"], data["C_fp32"])["stats"],
    }
    viz.heatmap_3way(result, savepath=args.save, title=args.title)
    if args.save:
        print(f"viz: heatmap → {args.save}")


# ─── argparse wiring ───────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(prog="mxp_tools")
    sp = p.add_subparsers(dest="cmd", required=True)

    g = sp.add_parser("gen", help="generate FP32 A, B + truth C")
    _add_io(g); _add_shape(g)
    g.set_defaults(func=cmd_gen)

    g = sp.add_parser("emit", help="quantize + write HW input .hex files")
    _add_io(g)
    g.add_argument("--prec", type=int, choices=ALL_PRECS, default=None,
                   help="single prec to emit; default: all of 2,4,8")
    g.set_defaults(func=cmd_emit)

    g = sp.add_parser("ref", help="compute SW golden C_sw for each (prec_a, prec_b)")
    _add_io(g)
    g.add_argument("--prec-a", type=int, choices=ALL_PRECS, default=None)
    g.add_argument("--prec-b", type=int, choices=ALL_PRECS, default=None)
    g.add_argument("--accum", choices=("fp32", "bf16"), default="fp32",
                   help="golden accumulator precision (default fp32)")
    g.set_defaults(func=cmd_ref)

    g = sp.add_parser("compare", help="3-way diff: HW vs SW vs FP32")
    g.add_argument("--ref", required=True, help="path to C_sw_*.npz from `ref`")
    g.add_argument("--hw-banks", nargs="+", required=True,
                   help="one or more $writememh files (one per bank)")
    g.add_argument("--layout", choices=("single", "split-rows",
                                        "interleaved_row_major_16bank",
                                        "interleaved_row_major_32bank"),
                   default="single")
    g.add_argument("--n-banks", type=int, default=1)
    g.add_argument("--save", default=None, help="dump .npz of result")
    g.set_defaults(func=cmd_compare)

    g = sp.add_parser("viz", help="heatmap from a compare .npz")
    g.add_argument("--npz", required=True)
    g.add_argument("--save", default=None, help="PNG path (else show)")
    g.add_argument("--title", default=None)
    g.set_defaults(func=cmd_viz)

    from . import rmw_gen
    g = sp.add_parser("rmw-gen", help="generate RMW unit-test vectors")
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=64)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=lambda a: rmw_gen.main([
        "--out", a.out, "--n", str(a.n), "--seed", str(a.seed)
    ]))

    args = p.parse_args(argv)
    rc = args.func(args)
    # Commands that don't gate (gen/emit/ref/viz/rmw-gen) return None → exit 0.
    # cmd_compare returns 1 on bit-exact mismatch.
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
