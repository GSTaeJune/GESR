# MXP_scheduler/measure_gap.py
"""Measure the optimality gap the structural baseline leaves vs the PROVEN-optimal schedule.

gap = (warmstart_energy - opt_energy) / opt_energy, recorded where the engine proves optimal;
on timeout the CP-SAT backend prints the HONEST (lower-bound-relative) gap, marked with '*'.

Backends:
  --backend astar  : M1 A* (default; OOMs at T>=32 under pressure -- keep to T<=12 shapes).
  --backend cpsat  : CP-SAT engine; pushes to the T=32/64 shapes A* cannot reach.
  --backend band   : band-serpentine (B,d) heuristic; large-T honest-gap rows (never proven).

Run from MXP_scheduler/:
  python measure_gap.py --backend cpsat
  python measure_gap.py --backend cpsat --max-time 60
  python measure_gap.py --backend band
"""
import argparse
import mxp_scheduler as s
import warmstart as ws
import eval_sched as es

BANKS, WB = 32, 32

ASTAR_SHAPES = [(64, 64, 32), (64, 64, 64), (96, 64, 64)]            # T = 4, 8, 12
CPSAT_SHAPES = [(64, 64, 32), (96, 64, 64), (128, 128, 64), (128, 128, 128)]  # T = 4, 12, 32, 64
BAND_SHAPES = [(256, 256, 256), (512, 512, 256), (512, 512, 512)]    # T = 512, 2048, 4096
PRECS = {
    "unif8": lambda MT, KT: [[8] * KT for _ in range(MT)],
    "unif2": lambda MT, KT: [[2] * KT for _ in range(MT)],
    "mixed": lambda MT, KT: [[2 if (i + j) % 2 == 0 else 8 for j in range(KT)] for i in range(MT)],
}
MULTS = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def min_working_set(w):
    mws = 0
    for c in es.all_cubes(w):
        ws_c = (es.tile_size(es.a_tile(c), w) + es.tile_size(es.w_tile(c), w)
                + es.tile_size(es.c_tile(c), w))
        mws = max(mws, ws_c)
    return mws


def _solve(backend, w, hw, max_time):
    if backend == "astar":
        import astar
        r = astar.optimize_exact(w, hw)
        return (r["energy"] if r["feasible"] else None, r["proven_optimal"],
                r["nodes_expanded"], r["gap"])
    if backend == "band":
        import band_sched
        r = band_sched.optimize_band(w, hw)
        return (r["energy"] if r["feasible"] else None, r["proven_optimal"],
                r["nodes_expanded"], r["gap"])
    import cpsat_sched
    r = cpsat_sched.optimize_exact(w, hw, max_time=max_time)
    return (r["energy"] if r["feasible"] else None, r["proven_optimal"],
            r["nodes_expanded"], r["gap"])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["astar", "cpsat", "band"], default="astar")
    p.add_argument("--max-time", type=float, default=30.0, help="per-instance CP-SAT budget (s)")
    p.add_argument("--quick", action="store_true",
                   help="smallest shape, one prec, two capacities -- fast CI smoke (not the full sweep)")
    args = p.parse_args(argv)
    shapes = (ASTAR_SHAPES if args.backend == "astar"
              else CPSAT_SHAPES if args.backend == "cpsat"
              else BAND_SHAPES)
    precs = PRECS
    mults = MULTS
    if args.quick:
        shapes = shapes[:1]
        precs = {"mixed": PRECS["mixed"]}
        mults = MULTS[:2]

    print("backend=%s   ('*' = NOT proven optimal)" % args.backend)
    print("  gap% column: proven -> (warm-opt)/opt vs the proven optimum;")
    print("               not proven -> (warm-incumbent)/incumbent* AND 'lbN' = honest")
    print("               lower-bound-relative gap (energy-lb)/lb. Negative* = incumbent worse than warm.")
    print("shape         T  prec  capxWS  warmE(e6) optE(e6)  proven  nodes   gap%")
    print("-" * 80)
    for (M, K, N) in shapes:
        MT, KT, NT = M // 32, K // 32, N // 32
        T = MT * KT * NT
        for pname, pf in precs.items():
            w = s.Work(M=M, K=K, N=N, wbits=pf(MT, KT), act_bits=8)
            mws = min_working_set(w)
            for mult in mults:
                bank_size = max(1, int(mws * mult) // (BANKS * WB))
                hw = s.HW(bank_size=bank_size, banks=BANKS, dram_bw=1e12, word_bits=WB)
                capx = hw.cap_bits / mws
                if args.backend == "band" and T > 4096:
                    warm = None                                # large T: skip structural warmstart
                else:
                    warm = ws.structural_incumbent(w, hw)
                optE, proven, nodes, gap = _solve(args.backend, w, hw, args.max_time)
                warmE = warm[0] if warm else None
                if warmE and optE and optE > 0:
                    sg = (warmE - optE) / optE * 100          # structural opportunity vs the answer
                    if proven:
                        gaps = "%6.2f" % sg                   # gap-1 vs the PROVEN optimum
                    else:
                        gaps = "%+6.1f* lb%.0f" % (sg, gap)   # not proven: struct-rel* AND honest lb gap
                elif optE and not proven:
                    gaps = "  -  * lb%.0f" % gap              # no warm baseline; honest lb gap only
                else:
                    gaps = "   -  "
                if warmE:
                    we = "%8.2f" % (warmE / 1e6)
                elif warm is None:
                    we = "  warm=-"                           # band large-T: warmstart skipped
                else:
                    we = "   inf  "
                oe = "%8.2f" % (optE / 1e6) if optE else "   inf  "
                print("%dx%dx%-4d %3d %6s %5.2fx  %s %s  %5s  %6d  %s"
                      % (M, K, N, T, pname, capx, we, oe, str(proven), nodes, gaps))
        print("-" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
