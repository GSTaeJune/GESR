# MXP_scheduler/demo_qwen.py
"""Pull a Qwen2.5 Q/K/V projection GEMM shape, assign an arbitrary per-tile avg-bit map,
run the band-serpentine (B,d) scheduler, and emit the schedule (order + per-region (B,d) +
honest gap). The 'arbitrary per-tile avg bits' model the MXINT block-32 mixed precision
(each 32x32 W tile = 32 blocks; the per-tile average lies on a lattice in [2,8]).

Run:  cd MXP_scheduler && python demo_qwen.py --model qwen2.5-0.5b --proj q --seq 128
"""
import argparse
import os
import mxp_scheduler as s
import band_sched as b

# Qwen2.5 (hidden H, kv projection out dim for GQA). Out dim for q = H.
QWEN = {
    "qwen2.5-0.5b": {"H": 896,  "kv": 128},
    "qwen2.5-7b":   {"H": 3584, "kv": 512},
    "qwen2.5-72b":  {"H": 8192, "kv": 1024},
}


def _round32(x):
    return max(32, (x // 32) * 32)


def _wbits_random_2_4_8(MT, KT, seed):
    """Per-tile average of 32 per-block draws from {2,4,8} -> a fractional avg in [2,8]
    (deterministic LCG; no Math.random/Date dependency)."""
    out = []
    state = (seed * 2654435761 + 12345) & 0xFFFFFFFF
    for i in range(MT):
        row = []
        for j in range(KT):
            tot = 0
            for _ in range(32):                      # 32 blocks per tile
                state = (state * 1103515245 + 12345) & 0x7FFFFFFF
                tot += (2, 4, 8)[(state >> 16) % 3]
            row.append(tot / 32.0)                    # per-tile average bits in [2,8]
        out.append(row)
    return out


def build_work(model, proj, seq, act):
    cfg = QWEN[model]
    H = _round32(cfg["H"])
    out = H if proj == "q" else _round32(cfg["kv"])
    M, K, N = _round32(seq), H, out                  # activation[S,H] x weight[H,out]
    MT, KT = M // s.TILE, K // s.TILE
    wbits = _wbits_random_2_4_8(MT, KT, seed=0)
    return s.Work(M=M, K=K, N=N, wbits=wbits, act_bits=act)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(QWEN), default="qwen2.5-0.5b")
    p.add_argument("--proj", choices=["q", "k", "v"], default="q")
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--act", type=int, default=8)
    p.add_argument("--bank-size", type=int, default=4096)
    p.add_argument("--banks", type=int, default=32)
    p.add_argument("--dram-bw", type=float, default=1e12)
    p.add_argument("--out", default=None, help="path to write the full cube order")
    args = p.parse_args(argv)

    w = build_work(args.model, args.proj, args.seq, args.act)
    hw = s.HW(bank_size=args.bank_size, banks=args.banks, dram_bw=args.dram_bw)
    T = w.MT * w.KT * w.NT
    res = b.optimize_band(w, hw)

    print("model=%s proj=%s  M=%d K=%d N=%d (MT=%d KT=%d NT=%d, T=%d)  act=%d cap=%d"
          % (args.model, args.proj, w.M, w.K, w.N, w.MT, w.KT, w.NT, T, w.act_bits, hw.cap_bits))
    if not res["feasible"]:
        print("NO FEASIBLE SCHEDULE: %s" % res["reason"]); return 1
    print("energy=%.0f  honest gap=%.1f%% (lb=%.0f)  (B,d) evaluated=%d"
          % (res["energy"], res["gap"] * 100, res["lower_bound"], res["nodes_expanded"]))
    order = res["order"]
    print("order length=%d  head=%s  tail=%s" % (len(order), order[:4], order[-4:]))
    out = args.out or os.path.join("work", "%s_%s" % (args.model, args.proj), "band_order.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for i, c in enumerate(order):
            f.write("%d %d,%d,%d\n" % (i, c[0], c[1], c[2]))
    print("full order written to %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
