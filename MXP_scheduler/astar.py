# MXP_scheduler/astar.py
"""astar — exact joint (order + eviction) GEMM-mapping optimizer for the MXP scheduler (M1).

Best-first search over SchedState = (done, resident, last_compute). g = cumulative VARIABLE
DRAM energy (read+spill)*(dram+onchip). h = admissible lower bound = remaining needed-and-
non-resident A/W tiles * coef (C reloads/spills lower-bound to 0; final C writes are a
schedule-invariant constant, excluded from g and h alike -- an exact reformulation of spec
§7's final-write floor, not an approximation). stall=0 is a hard prune. Determinism via
(f, g, order-signature) keys. Exact when the frontier empties; otherwise incumbent + min open f
= honest gap. Seeded with the best structural schedule (spec §D6 floor). stdlib only.
"""
import heapq
import itertools
import eval_sched as es
import warmstart as ws

DEFAULT_NODE_BUDGET = 200_000


def _heuristic(state, cubes, w, coef):
    """Admissible lower bound on remaining g: every remaining cube's A and W tile, if not
    resident, needs >= 1 load; C reload/spill lower-bounds to 0 (final C writes are the
    excluded invariant constant). ADMISSIBLE but NOT consistent (see the plan's heuristic
    note) -- optimality relies on the exact-state best_g_to_state re-expansion below."""
    needed_a, needed_w = set(), set()
    for c in cubes:
        if c not in state.done:
            needed_a.add(es.a_tile(c))
            needed_w.add(es.w_tile(c))
    bits = 0.0
    for t in needed_a | needed_w:
        if t not in state.resident:
            bits += es.tile_size(t, w)
    return bits * coef


def astar(w, hw, node_budget=DEFAULT_NODE_BUDGET, warm=None):
    """Joint (order+eviction) search. Returns a dict:
    {energy, order, evictions, feasible, proven_optimal, lower_bound, gap, nodes_expanded,
     source, min_steady_stall, reason}.

    Move set = es.eviction_choices (the full exact set, shared with the oracle; invariant #10).
    Determinism = (f, g, sig, seq) heap key with a monotonic seq so SchedState is never compared
    (invariant #7). Termination = exact when the frontier empties without hitting the node
    budget; hitting the budget forces proven_optimal=False with an honest gap (invariant #8).
    Infeasible (no stall=0 schedule) -> feasible=False + diagnostic (invariant #11)."""
    cubes = es.all_cubes(w)
    T = len(cubes)
    cube_id = {c: i for i, c in enumerate(cubes)}
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    seq = itertools.count()                          # monotonic tie-breaker (never compares state)

    start = es.SchedState(done=frozenset(), resident=frozenset(), last_compute=-1.0)

    # incumbent (spec §D6 floor): best structural schedule, if any
    if warm is None:
        warm = ws.structural_incumbent(w, hw)
    inc_energy = warm[0] if warm is not None else float("inf")
    inc_order = list(warm[1]) if warm is not None else None
    inc_evict = list(warm[2]) if warm is not None else None
    inc_source = "structural" if warm is not None else None

    h0 = _heuristic(start, cubes, w, coef)
    open_heap = [(h0, 0.0, (), next(seq), start, [], [])]
    best_g_to_state = {start: 0.0}
    nodes = 0
    min_open_f = h0
    budget_hit = False

    while open_heap:
        f, g, sig, _s, state, order_list, evict_list = heapq.heappop(open_heap)
        min_open_f = f                               # popped in nondecreasing f order
        if g > best_g_to_state.get(state, float("inf")):
            continue                                 # stale (a cheaper path to state was found)
        if f >= inc_energy:                          # cannot beat the incumbent -> done
            break                                    # (heap is f-ordered: nothing better remains)
        if len(state.done) == T:                     # goal
            if g < inc_energy:
                inc_energy, inc_order, inc_evict, inc_source = g, order_list, evict_list, "astar"
            continue
        if nodes >= node_budget:                     # budget exhausted BEFORE expanding -> stop
            budget_hit = True
            break
        nodes += 1

        for c in cubes:                              # deterministic: canonical cube order
            if c in state.done:
                continue
            for ev in es.eviction_choices(state, c, w, hw):   # full exact move set
                ns, read, spill, unhidden, cap_ok = es.apply_cube(state, c, ev, w, hw)
                if not cap_ok:
                    continue                         # eviction_choices guarantees this; defensive
                if state.last_compute >= 0 and unhidden > 0:
                    continue                         # stall=0 HARD prune (mid-stream)
                ng = g + (read + spill) * coef
                if ng >= best_g_to_state.get(ns, float("inf")):
                    continue
                nf = ng + _heuristic(ns, cubes, w, coef)
                if nf >= inc_energy:                 # bound: cannot beat incumbent
                    continue
                best_g_to_state[ns] = ng
                nsig = sig + (cube_id[c],)
                heapq.heappush(open_heap,
                               (nf, ng, nsig, next(seq), ns, order_list + [c], evict_list + [ev]))

    feasible = inc_order is not None
    if budget_hit:
        proven = False
    else:
        proven = (not open_heap) or (inc_energy <= min_open_f + 1e-9)
    lower_bound = inc_energy if (proven and feasible) else min_open_f
    gap = (0.0 if (proven and feasible)
           else (inc_energy - lower_bound) / lower_bound if (lower_bound > 0 and feasible) else 0.0)

    # invariant #11: never return a bare inf -- diagnose the empty feasible region
    min_ss, reason = None, None
    if not feasible:
        min_ss = ws.min_structural_steady_stall(w, hw)
        if min_ss is None:
            reason = ("no capacity-feasible schedule exists (footprint exceeds on-chip capacity "
                      "for every mapping); increase banks / bank_size / word_bits.")
        else:
            reason = (f"no stall=0-feasible schedule; closest structural schedule still stalls "
                      f"{min_ss:.1f} cycles mid-stream. Raise dram_bw (eff_bw={hw.eff_bw:g}) or "
                      f"shrink the resident window so each fetch hides under the prior cube's compute.")

    return {"energy": inc_energy, "order": inc_order, "evictions": inc_evict,
            "feasible": feasible, "proven_optimal": (feasible and proven),
            "lower_bound": lower_bound, "gap": gap, "nodes_expanded": nodes,
            "source": inc_source, "min_steady_stall": min_ss, "reason": reason}


def optimize_exact(w, hw, node_budget=DEFAULT_NODE_BUDGET):
    """Public entry point. Joint (order+eviction) exact optimizer with D6 structural floor.
    Returns the astar() result dict: energy, order, evictions, feasible, proven_optimal,
    lower_bound, gap, nodes_expanded, source, min_steady_stall, reason. When feasible is False,
    energy is inf, order/evictions are None, and reason/min_steady_stall carry the diagnostic."""
    return astar(w, hw, node_budget=node_budget)


def selftest():
    import mxp_scheduler as s
    import oracle as o
    # G1: all-resident -> no reloads/spills; energy = first-touch A/W floor (12288 bits * coef),
    # NOT zero. Proven at the root.
    w1 = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw1 = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)
    coef1 = hw1.coeffs["dram"] + hw1.coeffs["onchip"]
    r1 = optimize_exact(w1, hw1)
    assert r1["energy"] == 12288 * coef1 and r1["proven_optimal"], r1
    # G2: A* == oracle under capacity pressure (mixed-precision W sizes)
    w2 = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    hw2 = s.HW(bank_size=2, banks=32, dram_bw=10 ** 12, word_bits=1024)   # cap 65536
    r2 = optimize_exact(w2, hw2)
    ref2 = o.dp_optimal(w2, hw2)
    assert r2["proven_optimal"] and abs(r2["energy"] - ref2["energy"]) < 1e-6, (r2, ref2)
    # G3: determinism
    assert optimize_exact(w2, hw2)["order"] == r2["order"]
    print("astar selftest: OK")


def main(argv=None):
    import argparse
    import mxp_scheduler as s
    p = argparse.ArgumentParser(description="MXP_scheduler M1 - exact joint (order+eviction) optimizer")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--M", type=int); p.add_argument("--K", type=int); p.add_argument("--N", type=int)
    p.add_argument("--act", type=int, default=8)
    p.add_argument("--bank-size", type=int, default=1024)
    p.add_argument("--banks", type=int, default=32)
    p.add_argument("--dram-bw", type=float, default=64.0)
    p.add_argument("--node-budget", type=int, default=DEFAULT_NODE_BUDGET)
    a = p.parse_args(argv)
    if a.selftest:
        selftest(); return 0
    if not (a.M and a.K and a.N):
        p.error("provide --M --K --N (or --selftest)")
    MT, KT = a.M // s.TILE, a.K // s.TILE
    w = s.Work(M=a.M, K=a.K, N=a.N, wbits=[[a.act] * KT for _ in range(MT)], act_bits=a.act)
    hw = s.HW(bank_size=a.bank_size, banks=a.banks, dram_bw=a.dram_bw)
    res = optimize_exact(w, hw, node_budget=a.node_budget)
    if not res["feasible"]:
        print(f"NO FEASIBLE SCHEDULE: {res['reason']}")
        return 1                                      # CLI error, never a silent inf (invariant #11)
    status = "PROVEN OPTIMAL" if res["proven_optimal"] else f"gap={res['gap']*100:.1f}% (lb={res['lower_bound']:.0f})"
    print(f"energy(variable DRAM) = {res['energy']:.0f}   {status}   nodes={res['nodes_expanded']}   source={res['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
