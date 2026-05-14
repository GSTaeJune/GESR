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
        print(f"emit: {name}  →  {sub}/ (a_input_BS/BP, b_input, a/b_scale, quant_{name}.npz)")


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
            )
            C_sw = C_sw_pad[:M, :N]
            path = os.path.join(out_ref, f"C_sw_{PREC_NAMES[pa]}_{PREC_NAMES[pb]}.npz")
            np.savez(path, C_sw=C_sw, C_fp32=C_fp32, prec_a=pa, prec_b=pb)
            print(f"ref: {PREC_NAMES[pa]} × {PREC_NAMES[pb]}  →  {path}")


def _resolve_mapping(args, M, N):
    if args.layout == "single":
        return hwio.default_single_bank_row_major
    if args.layout == "split-rows":
        return hwio.default_banks_split_rows(args.n_banks)
    if args.layout == "interleaved_row_major_16bank":
        return hwio.interleaved_row_major_16bank
    raise ValueError(f"unknown layout {args.layout}")


def cmd_compare(args):
    ref_npz = np.load(args.ref)
    C_sw = ref_npz["C_sw"].astype(np.float32)
    C_fp32 = ref_npz["C_fp32"].astype(np.float32)
    M, N = C_sw.shape

    mapping = _resolve_mapping(args, M, N)
    C_hw = hwio.gather_banks(args.hw_banks, M, N, mapping)
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
    g.set_defaults(func=cmd_ref)

    g = sp.add_parser("compare", help="3-way diff: HW vs SW vs FP32")
    g.add_argument("--ref", required=True, help="path to C_sw_*.npz from `ref`")
    g.add_argument("--hw-banks", nargs="+", required=True,
                   help="one or more $writememh files (one per bank)")
    g.add_argument("--layout", choices=("single", "split-rows", "interleaved_row_major_16bank"), default="single")
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
    args.func(args)


if __name__ == "__main__":
    main()
