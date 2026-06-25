# MXP_scheduler/cpsat_sched.py
"""cpsat_sched -- CP-SAT joint (cube-order + tile-eviction) EXACT optimizer for the MXP scheduler.

Offline tool (depends on ortools); NEVER imported by the stdlib runtime. Step-indexed CP-SAT
model with COSMA-style C/P/S/R residency actions, demand-driven loads, exact-rational integer
scaling, M1-bit-exact cost (reproduces eval_sched.apply_cube), stall=0 hard constraint, honest
gap on timeout. Returns a dict shaped like astar.optimize_exact.

Spec: docs/superpowers/specs/2026-06-25-mxp-scheduler-cpsat-design.md
"""
import math
from fractions import Fraction
from ortools.sat.python import cp_model
import mxp_scheduler as s
import eval_sched as es

INT64_MARGIN = 2 ** 62


def _cubes_tiles(w):
    """Canonical cube list and the sorted set of distinct A/W/C tiles they touch."""
    cubes = es.all_cubes(w)
    tiles = sorted({t for c in cubes
                    for t in (es.a_tile(c), es.w_tile(c), es.c_tile(c))})
    return cubes, tiles


def _global_scale(w, hw):
    """Single positive-integer scale G that integerizes EVERY rational coefficient in the model:
    tile sizes, cap_bits, the stall LHS (size*freq_ratio) and the stall RHS (dram_bw*cube_compute).
    Exact via fractions.Fraction (NO round()). G == 1 for integer-wbits / integral-BW configs."""
    cubes, tiles = _cubes_tiles(w)
    fr = Fraction(hw.freq_ratio).limit_denominator(10 ** 12)
    bw = Fraction(hw.dram_bw).limit_denominator(10 ** 12)
    rats = []
    for t in tiles:
        sz = Fraction(es.tile_size(t, w)).limit_denominator(10 ** 12)
        rats.append(sz)            # capacity + objective
        rats.append(sz * fr)       # stall LHS
    rats.append(Fraction(hw.cap_bits).limit_denominator(10 ** 12))
    for c in cubes:
        rats.append(bw * Fraction(es.cube_compute(c, w, hw)).limit_denominator(10 ** 12))  # stall RHS
    g = 1
    for r in rats:
        g = g * r.denominator // math.gcd(g, r.denominator)
    return g


def _build(w, hw, enforce_stall=True):
    """Build the step-indexed CP-SAT model. Returns a handle dict. enforce_stall=False drops the
    stall=0 constraint (used by the infeasibility diagnostic to tell capacity- from stall-infeasible)."""
    cubes, tiles = _cubes_tiles(w)
    T = len(cubes)
    KT = w.KT
    G = _global_scale(w, hw)
    fr = Fraction(hw.freq_ratio).limit_denominator(10 ** 12)
    bw = Fraction(hw.dram_bw).limit_denominator(10 ** 12)
    size_G = {t: int(Fraction(es.tile_size(t, w)) * G) for t in tiles}          # objective + capacity
    cap_G = int(Fraction(hw.cap_bits) * G)
    lhs_G = {t: int(Fraction(es.tile_size(t, w)) * fr * G) for t in tiles}      # stall LHS coeff
    rhs_G = {c: int(bw * Fraction(es.cube_compute(c, w, hw)) * G) for c in cubes}  # stall RHS coeff
    maxc = max([cap_G, 1] + list(size_G.values()) + list(lhs_G.values()) + list(rhs_G.values()))
    if maxc >= INT64_MARGIN:
        raise ValueError("cpsat_sched: scaled coefficient %d exceeds int64 margin; "
                         "reduce dram_bw or precision of inputs" % maxc)

    uses = {t: [] for t in tiles}
    for c in cubes:
        for t in (es.a_tile(c), es.w_tile(c), es.c_tile(c)):
            uses[t].append(c)
    awtiles = [t for t in tiles if t[0] != "C"]
    ctiles = [t for t in tiles if t[0] == "C"]
    cgroup = {ct: [c for c in cubes if es.c_tile(c) == ct] for ct in ctiles}

    m = cp_model.CpModel()
    x = {(c, t): m.NewBoolVar("x_%d_%d" % (ci, t))
         for ci, c in enumerate(cubes) for t in range(T)}
    for c in cubes:                                  # each cube runs exactly once
        m.Add(sum(x[(c, t)] for t in range(T)) == 1)
    for t in range(T):                               # exactly one cube per step
        m.Add(sum(x[(c, t)] for c in cubes) == 1)

    res = {(tau, t): m.NewBoolVar("res_%d_%d" % (ti, t))
           for ti, tau in enumerate(tiles) for t in range(T)}

    def rprev(tau, t):
        return res[(tau, t - 1)] if t > 0 else 0

    for c in cubes:                                  # co-residency: a cube's 3 tiles resident at its step
        for tau in (es.a_tile(c), es.w_tile(c), es.c_tile(c)):
            for t in range(T):
                m.Add(res[(tau, t)] >= x[(c, t)])

    load = {}
    for tau in tiles:
        for t in range(T):
            lv = m.NewBoolVar("load_%s_%d" % (tau, t))
            m.Add(lv >= res[(tau, t)] - rprev(tau, t))             # fires on non-resident->resident
            m.Add(lv <= sum(x[(c, t)] for c in uses[tau]))         # DEMAND-DRIVEN (closes vacuous stall)
            load[(tau, t)] = lv

    for t in range(T):                               # capacity (M1 semantics: post-load resident <= cap)
        m.Add(sum(size_G[tau] * res[(tau, t)] for tau in tiles) <= cap_G)

    rl, sp = {}, {}                                  # C reload / C partial-spill charged terms
    for ct in ctiles:
        for t in range(T):
            cnt = sum(x[(c, u)] for c in cgroup[ct] for u in range(t))   # cnt_before == c_counter(done)
            cge1 = m.NewBoolVar("cge1_%s_%d" % (ct, t))
            m.Add(cnt >= 1).OnlyEnforceIf(cge1)
            m.Add(cnt <= 0).OnlyEnforceIf(cge1.Not())
            ev = m.NewBoolVar("evict_%s_%d" % (ct, t))
            m.Add(ev >= rprev(ct, t) - res[(ct, t)])
            r = m.NewBoolVar("rl_%s_%d" % (ct, t))                  # reload = load AND cnt>=1
            m.Add(r >= load[(ct, t)] + cge1 - 1)
            rl[(ct, t)] = r
            clek = m.NewBoolVar("clek_%s_%d" % (ct, t))
            m.Add(cnt <= KT - 1).OnlyEnforceIf(clek)
            m.Add(cnt >= KT).OnlyEnforceIf(clek.Not())
            sv = m.NewBoolVar("sp_%s_%d" % (ct, t))                 # spill = evict AND 1<=cnt<=KT-1
            m.Add(sv >= ev + cge1 + clek - 2)
            sp[(ct, t)] = sv

    def traffic_terms(t, coeff):
        terms = [coeff[tau] * load[(tau, t)] for tau in awtiles]
        terms += [coeff[ct] * rl[(ct, t)] for ct in ctiles]
        terms += [coeff[ct] * sp[(ct, t)] for ct in ctiles]
        return terms

    obj = sum(sum(traffic_terms(t, size_G)) for t in range(T))
    m.Minimize(obj)

    if enforce_stall:                                # stall=0 hard: step i traffic hides under step i-1 compute
        for i in range(1, T):
            m.Add(sum(traffic_terms(i, lhs_G)) <= sum(rhs_G[c] * x[(c, i - 1)] for c in cubes))

    return dict(model=m, x=x, res=res, obj=obj, cubes=cubes, tiles=tiles, T=T, G=G)


def _reconstruct(solver, h):
    """Read (order, evictions) out of a solved model. evictions[t] = tiles resident at t-1 but not t."""
    x, res, cubes, tiles, T = h["x"], h["res"], h["cubes"], h["tiles"], h["T"]
    order = []
    for t in range(T):
        order.append(next(c for c in cubes if solver.Value(x[(c, t)]) == 1))
    evictions = []
    for t in range(T):
        ev = set()
        for tau in tiles:
            prev = solver.Value(res[(tau, t - 1)]) if t > 0 else 0
            if prev == 1 and solver.Value(res[(tau, t)]) == 0:
                ev.add(tau)
        evictions.append(frozenset(ev))
    return order, evictions


def _new_solver(max_time, random_seed):
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1            # determinism (in-process, fixed ortools version)
    solver.parameters.random_seed = random_seed
    if max_time is not None:
        solver.parameters.max_time_in_seconds = max_time
    return solver


def _infeasible_result(w, hw, nodes):
    """No stall=0 solution for the full model. Re-solve dropping stall=0 to tell capacity-infeasible
    from stall0-infeasible. Never returns a bare inf without a human reason (invariant: no silent inf)."""
    import warmstart as ws
    h2 = _build(w, hw, enforce_stall=False)
    solver2 = _new_solver(10.0, 0)
    st2 = solver2.Solve(h2["model"])
    base = {"energy": float("inf"), "order": None, "evictions": None,
            "feasible": False, "proven_optimal": False, "lower_bound": float("inf"),
            "gap": 0.0, "nodes_expanded": int(nodes), "source": "cpsat"}
    if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        base["min_steady_stall"] = ws.min_structural_steady_stall(w, hw)
        base["reason"] = ("no stall=0-feasible schedule; raise dram_bw (eff_bw=%g) or shrink the "
                          "resident window so each fetch hides under the prior cube's compute."
                          % hw.eff_bw)
    else:
        base["min_steady_stall"] = None
        base["reason"] = ("no capacity-feasible schedule exists (footprint exceeds on-chip capacity "
                          "for every mapping); increase banks / bank_size / word_bits.")
    return base


def optimize_exact(w, hw, max_time=None, random_seed=0):
    """Joint (order+eviction) exact optimizer. Returns a dict with the same keys as
    astar.optimize_exact: energy, order, evictions, feasible, proven_optimal, lower_bound, gap,
    nodes_expanded, source, min_steady_stall, reason. proven_optimal=True only when the solver
    proves OPTIMAL; on timeout it returns the incumbent with an honest gap; infeasible -> diagnostic."""
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    h = _build(w, hw, enforce_stall=True)
    solver = _new_solver(max_time, random_seed)
    status = solver.Solve(h["model"])
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        order, evictions = _reconstruct(solver, h)
        bits = solver.Value(h["obj"]) / h["G"]
        energy = bits * coef
        proven = status == cp_model.OPTIMAL
        if proven:
            lower_bound, gap = energy, 0.0
        else:
            lower_bound = (solver.BestObjectiveBound() / h["G"]) * coef
            gap = (energy - lower_bound) / lower_bound if lower_bound > 0 else 0.0
        return {"energy": energy, "order": order, "evictions": evictions, "feasible": True,
                "proven_optimal": proven, "lower_bound": lower_bound, "gap": gap,
                "nodes_expanded": int(solver.NumBranches()), "source": "cpsat",
                "min_steady_stall": None, "reason": None}
    return _infeasible_result(w, hw, solver.NumBranches())
