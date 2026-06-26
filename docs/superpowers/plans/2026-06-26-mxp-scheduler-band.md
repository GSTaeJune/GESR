# MXP_scheduler band-serpentine (B,d) scheduler -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (or executing-plans). Steps use `- [ ]` checkboxes.

**Goal:** Build `MXP_scheduler/band_sched.py` (a band-serpentine (B,d)-per-region heuristic scheduler scaling to any T, scored by the unchanged `eval_sched` cost model, with a first-touch-floor honest lower bound), wire it into `measure_gap.py --backend band`, and ship `demo_qwen.py` that pulls Qwen Q/K/V projection dims, assigns arbitrary per-tile avg bits, runs the scheduler, and emits the order.

**Architecture:** Per region (= one m-row) pick two knobs B (open C tiles) and d (K-depth). Order = band-serpentine over (m,n), k swept inside each band (k-outer/n-inner so W is amortized across B columns). Eviction = exact set-difference of the resident set across steps. The module computes NO cost itself — it folds `eval_sched.apply_cube` / calls `eval_sched.eval_sched` (single source of truth). Heuristic: `proven_optimal=False`, honest gap vs the first-touch A/W floor.

**Tech stack:** Python 3, stdlib only (no ortools), pytest. Imports `mxp_scheduler` (TILE/FP32_BITS/Work/HW) and `eval_sched` (all_cubes, a/w/c_tile, tile_size, apply_cube, eval_sched, SchedState). Spec: `docs/superpowers/specs/2026-06-26-mxp-scheduler-band-bd-design.md`.

**Key facts (zero repo context assumed):**
- Tests live in `MXP_scheduler/*.py`, run `cd MXP_scheduler && python -m pytest -q`. Imports are bare. No conftest.
- `eval_sched.eval_sched(w,hw,order,evictions)` REQUIRES `order` to be a full permutation of `all_cubes(w)` (raises otherwise) and returns dict incl. `feasible, stall0_feasible, capacity_feasible, dram_read_bits, dram_spill_bits, steady_stall, energy`. `energy = (dram_read_bits+dram_spill_bits)*coef`, `coef = hw.coeffs["dram"]+hw.coeffs["onchip"]`.
- `eval_sched.apply_cube(state, c, evict_frozenset, w, hw)` -> `(new_state, read, spill, unhidden, capacity_ok)`. `SchedState(frozenset(), frozenset(), -1.0)` is the empty start.
- `es.a_tile((m,k,n))=("A",k,n)`, `es.w_tile=("W",m,k)`, `es.c_tile=("C",m,n)`, `es.tile_size(tile,w)` bits.
- `mxp_scheduler`: `TILE=32`, `FP32_BITS=32`, `Work(M,K,N,wbits,act_bits)` with `w.MT/KT/NT`; `HW(...).cap_bits`, `.eff_bw`. `Work.wbits` entries must be in [2,8] (fractional ok).
- Result dict must match `astar.optimize_exact` keys: energy, order, evictions, feasible, proven_optimal, lower_bound, gap, nodes_expanded, source, min_steady_stall, reason.

**v1 note (discovered in design):** spec §7.2's C-spill fallback is UNREACHABLE in v1 — `B=1,d=1` is the minimal no-spill footprint (`C + max_k W(m,k) + A`); if that exceeds cap a cube genuinely doesn't fit (capacity-infeasible, no schedule), and spilling cannot go below co-residency. So v1 implements only the capacity/stall diagnostic; C-spill is documented as not-needed.

---

## Task 1: band_sched.py — geometry, constructor, footprint

**Files:** Create `MXP_scheduler/band_sched.py`; Test `MXP_scheduler/test_band_sched.py`.

- [ ] **Step 1: failing test**
```python
# MXP_scheduler/test_band_sched.py
import pytest
import mxp_scheduler as s
import eval_sched as es
import band_sched as b


def test_band_schedule_is_full_permutation_and_feasible():
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 4], [2, 2]], act_bits=2)   # MT=KT=NT=2, T=8
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)             # roomy
    knobs = [(w.NT, w.KT)] * w.MT                                     # B=NT, d=KT per region
    order, evictions = b.band_schedule(w, hw, knobs)
    assert sorted(order) == sorted(es.all_cubes(w))                   # full permutation
    assert len(evictions) == len(order)
    r = es.eval_sched(w, hw, order, evictions)
    assert r["feasible"] is True                                      # feasible by construction


def test_footprint_matches_resident_peak():
    # footprint(B,d) must equal the max per-step resident bits eval_sched sees for that region.
    w = s.Work(M=32, K=64, N=64, wbits=[[2, 8]], act_bits=2)          # MT=1,KT=2,NT=2
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)
    B, d = 2, 2
    order, evictions = b.band_schedule(w, hw, [(B, d)])
    # replay, tracking peak resident bits
    st = es.SchedState(frozenset(), frozenset(), -1.0)
    peak = 0
    for i, c in enumerate(order):
        st, *_ = es.apply_cube(st, c, evictions[i], w, hw)
        peak = max(peak, sum(es.tile_size(t, w) for t in st.resident))
    assert peak == b.footprint_bits(w, hw, 0, B, d)
```

- [ ] **Step 2: run, expect fail** — `cd MXP_scheduler && python -m pytest test_band_sched.py -q` -> `ModuleNotFoundError`.

- [ ] **Step 3: implement**
```python
# MXP_scheduler/band_sched.py
"""band_sched -- band-serpentine (B,d)-per-region heuristic scheduler for large T.

Offline, stdlib only. Two per-region knobs: B = open C tiles (band width), d = K-depth
(k-chunk size). Order = band-serpentine over (m,n), k swept inside each band (k-outer/n-inner
so W is amortized across the band's B columns). Eviction = exact set-difference of the resident
set across steps. Cost is scored by eval_sched (single source of truth) -- this module computes
no cost itself. Honest lower bound = first-touch A/W floor. Returns an astar.optimize_exact-shaped
dict. NOT imported by the stdlib runtime path that avoids heavy deps -- but it IS pure stdlib.

Spec: docs/superpowers/specs/2026-06-26-mxp-scheduler-band-bd-design.md
"""
import math
import mxp_scheduler as s
import eval_sched as es

TILE = s.TILE
FP32_BITS = s.FP32_BITS


def _groups(NT, B, reverse):
    g = [list(range(c, min(c + B, NT))) for c in range(0, NT, B)]
    return list(reversed(g)) if reverse else g


def _chunks(KT, d):
    return [list(range(c, min(c + d, KT))) for c in range(0, KT, d)]


def _emit_region(order, evictions, w, m, B, d, last_chunk_tiles, last_band_c):
    """Append region m's band-serpentine cubes + per-step evictions. Eviction = set-difference
    of the resident set across steps, realized as: at the first cube of each chunk drop the prior
    chunk's W/A; at the first cube of each band (also chunk 0) additionally drop the prior band's
    (completed) C tiles. Carried state (last_chunk_tiles, last_band_c) makes region boundaries a
    correct union too. Returns updated (last_chunk_tiles, last_band_c)."""
    for group in _groups(w.NT, B, reverse=(m % 2 == 1)):
        band_c = {es.c_tile((m, group[0], n)) for n in group}      # ("C",m,n) for n in group
        for ci, chunk in enumerate(_chunks(w.KT, d)):
            ev = set(last_chunk_tiles)                             # prior chunk's W/A
            if ci == 0:
                ev |= last_band_c                                  # band boundary: prior band's C
            this_chunk = set()
            first = True
            for k in chunk:                                        # k-outer
                for n in group:                                    # n-inner -> W(m,k) amortized
                    c = (m, k, n)
                    order.append(c)
                    evictions.append(frozenset(ev) if first else frozenset())
                    first = False
                    this_chunk.add(es.w_tile(c))
                    this_chunk.add(es.a_tile(c))
            last_chunk_tiles = this_chunk
        last_band_c = band_c
    return last_chunk_tiles, last_band_c


def band_schedule(w, hw, knobs):
    """knobs: list of (B,d) per region m (len == w.MT). Returns (order, evictions): order is a full
    permutation of all_cubes; evictions[i] applied by eval_sched before loading cube i."""
    if len(knobs) != w.MT:
        raise ValueError("knobs must have one (B,d) per region (w.MT=%d)" % w.MT)
    order, evictions = [], []
    lct, lbc = set(), set()
    for m in range(w.MT):
        B, d = knobs[m]
        if not (1 <= B <= w.NT and 1 <= d <= w.KT):
            raise ValueError("region %d: B in [1,NT], d in [1,KT]; got B=%s d=%s" % (m, B, d))
        lct, lbc = _emit_region(order, evictions, w, m, B, d, lct, lbc)
    return order, evictions


def footprint_bits(w, hw, m, B, d):
    """Resident peak for region m with (B,d): C(B) + W(max k-chunk) + A(d*B). Sizes via eval_sched."""
    c_region = B * TILE * TILE * FP32_BITS
    a_region = d * B * TILE * TILE * w.act_bits
    w_region = 0
    for chunk in _chunks(w.KT, d):
        cw = sum(es.tile_size(("W", m, k), w) for k in chunk)
        if cw > w_region:
            w_region = cw
    return c_region + w_region + a_region
```

- [ ] **Step 4: run, expect pass** — both tests green.
- [ ] **Step 5: commit**
```bash
git add MXP_scheduler/band_sched.py MXP_scheduler/test_band_sched.py
git commit -m "feat(scheduler): band_sched geometry + constructor + footprint (feasible-by-construction)"
```

---

## Task 2: per-region scorer, (B,d) search, optimize_band, honest LB

**Files:** Modify `MXP_scheduler/band_sched.py`; Test `MXP_scheduler/test_band_sched.py`.

- [ ] **Step 1: failing tests**
```python
def test_optimize_band_all_resident_floor():
    # Roomy cap -> band keeps everything resident -> energy == first-touch A/W floor, gap == 0.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)
    res = b.optimize_band(w, hw)
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    floor = (2 * 2048 + 4 * 2048) * coef                              # 2 A + 4 W tiles, 2048 bits each
    assert res["feasible"] is True and res["source"] == "band"
    assert res["proven_optimal"] is False
    assert res["energy"] == pytest.approx(floor)
    assert res["lower_bound"] == pytest.approx(floor)
    assert res["gap"] == pytest.approx(0.0, abs=1e-9)


def test_optimize_band_parity_and_not_below_optimum():
    # band energy == eval_sched(its order); and band cannot beat the free optimum (oracle).
    import oracle as o
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=2, banks=32, dram_bw=10 ** 12, word_bits=1024)  # cap 65536, pressure
    res = b.optimize_band(w, hw)
    r = es.eval_sched(w, hw, res["order"], res["evictions"])
    assert r["energy"] == pytest.approx(res["energy"])                # module never self-computes cost
    opt = o.dp_optimal(w, hw)["energy"]
    assert res["energy"] >= opt - 1e-6                                # restriction cannot beat optimum
```

- [ ] **Step 2: run, expect fail** (`optimize_band` missing).

- [ ] **Step 3: implement** (append to `band_sched.py`)
```python
def _score_region(w, hw, m, B, d):
    """Partial apply_cube fold over region m alone (fresh resident). Returns
    (bits, capacity_ok, steady_stall): bits = read+spill, steady_stall = sum of mid-stream
    un-hidden transfer. Region-isolated (its first cube treated as fill) -- used only to SELECT
    (B,d); the whole-schedule eval_sched pass is authoritative."""
    order, evictions = [], []
    _emit_region(order, evictions, w, m, B, d, set(), set())
    st = es.SchedState(frozenset(), frozenset(), -1.0)
    bits = 0.0
    steady = 0.0
    for i, c in enumerate(order):
        st, read, spill, unhidden, cap_ok = es.apply_cube(st, c, evictions[i], w, hw)
        if not cap_ok:
            return float("inf"), False, float("inf")
        if i > 0:
            steady += unhidden
        bits += read + spill
    return bits, True, steady


def _candidates(n):
    """(B,d) candidate values in [1,n]. Full range for small n; for large n a logged subset
    (powers of two + n). Ragged tails are allowed, so divisibility is NOT required."""
    if n <= 16:
        return list(range(1, n + 1))
    cand = {1, n}
    p = 1
    while p <= n:
        cand.add(p)
        p *= 2
    return sorted(cand)


def first_touch_floor(w, hw):
    """Valid lower bound on the objective: every distinct A and W tile is loaded at least once."""
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    cubes = es.all_cubes(w)
    a_tiles = {es.a_tile(c) for c in cubes}
    w_tiles = {es.w_tile(c) for c in cubes}
    bits = (sum(es.tile_size(t, w) for t in a_tiles)
            + sum(es.tile_size(t, w) for t in w_tiles))
    return bits * coef


def optimize_band(w, hw, skiplog=None):
    """Per-region (B,d) search + band-serpentine schedule, scored by eval_sched. Returns the
    astar.optimize_exact-shaped dict. Heuristic: proven_optimal=False; gap vs first-touch floor.
    skiplog: optional list; appended with a note when the (B,d) candidate set is capped (no silent
    truncation)."""
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    Bcand, dcand = _candidates(w.NT), _candidates(w.KT)
    if skiplog is not None and (len(Bcand) < w.NT or len(dcand) < w.KT):
        skiplog.append("(B,d) candidates capped: B in %s of 1..%d, d in %s of 1..%d"
                       % (Bcand, w.NT, dcand, w.KT))
    knobs = []
    n_eval = 0
    bad = None                                          # (m, reason, min_stall)
    for m in range(w.MT):
        best = None                                     # (bits, B, d)
        for B in Bcand:
            for d in dcand:
                if footprint_bits(w, hw, m, B, d) > hw.cap_bits:
                    continue
                n_eval += 1
                bits, cap_ok, steady = _score_region(w, hw, m, B, d)
                if not cap_ok or steady > 0:            # capacity already filtered; enforce stall=0
                    continue
                if best is None or bits < best[0]:
                    best = (bits, B, d)
        if best is None and bad is None:                # region infeasible -> diagnose once
            if footprint_bits(w, hw, m, 1, 1) > hw.cap_bits:
                bad = (m, "capacity", None)
            else:
                _, _, steady = _score_region(w, hw, m, 1, 1)
                bad = (m, "stall", steady)
        knobs.append((best[1], best[2]) if best else (1, 1))

    lb = first_touch_floor(w, hw)
    if bad is not None:
        m, reason, stall = bad
        if reason == "capacity":
            msg = ("region %d: no capacity-feasible (B,d); even one cube's A+W+C exceeds cap "
                   "(%d bits). Increase banks/bank_size/word_bits." % (m, hw.cap_bits))
        else:
            msg = ("region %d: capacity-fits at (B=1,d=1) but no stall=0 schedule "
                   "(min steady stall ~%.1f). Raise dram_bw (eff_bw=%g)." % (m, stall, hw.eff_bw))
        return {"energy": float("inf"), "order": None, "evictions": None, "feasible": False,
                "proven_optimal": False, "lower_bound": lb, "gap": float("inf"),
                "nodes_expanded": n_eval, "source": "band", "min_steady_stall": stall, "reason": msg}

    order, evictions = band_schedule(w, hw, knobs)
    r = es.eval_sched(w, hw, order, evictions)
    energy = r["energy"]
    gap = (energy - lb) / lb if (r["feasible"] and lb > 0) else float("inf")
    return {"energy": energy if r["feasible"] else float("inf"),
            "order": order if r["feasible"] else None,
            "evictions": evictions if r["feasible"] else None,
            "feasible": bool(r["feasible"]), "proven_optimal": False,
            "lower_bound": lb, "gap": gap if r["feasible"] else float("inf"),
            "nodes_expanded": n_eval, "source": "band",
            "min_steady_stall": None if r["feasible"] else r["steady_stall"],
            "reason": None if r["feasible"] else "constructed schedule infeasible under eval_sched"}
```

- [ ] **Step 4: run, expect pass.**
- [ ] **Step 5: commit**
```bash
git add MXP_scheduler/band_sched.py MXP_scheduler/test_band_sched.py
git commit -m "feat(scheduler): band per-region (B,d) search + optimize_band + first-touch-floor honest gap"
```

---

## Task 3: infeasibility diagnostics + restriction-gap + uniform/selftest

**Files:** Modify `band_sched.py` (add `selftest`, `main`); Test `test_band_sched.py`.

- [ ] **Step 1: failing tests**
```python
def test_capacity_infeasible_reported():
    w = s.Work(M=32, K=32, N=32, wbits=[[8]], act_bits=8)             # one cube A+W+C=49152
    hw = s.HW(bank_size=1, banks=1, dram_bw=10 ** 12, word_bits=32)   # cap 32 << one cube
    res = b.optimize_band(w, hw)
    assert res["feasible"] is False and res["energy"] == float("inf")
    assert res["gap"] == float("inf") and "capacity" in res["reason"]


def test_stall_infeasible_reported():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=1)                    # roomy cap, tiny BW
    res = b.optimize_band(w, hw)
    assert res["feasible"] is False and res["energy"] == float("inf")
    assert "stall" in res["reason"] and res["min_steady_stall"] is not None


@pytest.mark.parametrize("M,K,N,wb,act,bank_size,word_bits", [
    (64, 64, 32, [[2, 2], [2, 2]], 2, 1024, 32),
    (32, 64, 64, [[2, 8]], 2, 1024, 32),
    (64, 64, 32, [[2, 4], [2, 2]], 2, 2, 1024),
])
def test_band_never_below_optimum(M, K, N, wb, act, bank_size, word_bits):
    import oracle as o
    w = s.Work(M=M, K=K, N=N, wbits=wb, act_bits=act)
    hw = s.HW(bank_size=bank_size, banks=32, dram_bw=10 ** 12, word_bits=word_bits)
    res = b.optimize_band(w, hw)
    if res["feasible"]:
        assert res["energy"] >= o.dp_optimal(w, hw)["energy"] - 1e-6


def test_band_selftest_runs():
    b.selftest()
```

- [ ] **Step 2: run, expect fail** (`selftest` missing; others should pass against Task-2 code — if a diagnostic test fails, fix `optimize_band`, do not weaken the test).

- [ ] **Step 3: implement** `selftest` + `main` (append):
```python
def selftest():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    r = optimize_band(w, hw)
    assert r["feasible"] and r["source"] == "band" and not r["proven_optimal"]
    assert abs(r["energy"] - 12288 * coef) < 1e-6, r          # all-resident floor
    chk = es.eval_sched(w, hw, r["order"], r["evictions"])
    assert chk["feasible"] and abs(chk["energy"] - r["energy"]) < 1e-6   # parity
    print("band_sched selftest: OK")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="MXP_scheduler band-serpentine (B,d) heuristic scheduler")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--M", type=int); p.add_argument("--K", type=int); p.add_argument("--N", type=int)
    p.add_argument("--act", type=int, default=8)
    p.add_argument("--bank-size", type=int, default=1024)
    p.add_argument("--banks", type=int, default=32)
    p.add_argument("--dram-bw", type=float, default=64.0)
    args = p.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    if not (args.M and args.K and args.N):
        p.error("provide --M --K --N (or --selftest)")
    mt, kt = args.M // s.TILE, args.K // s.TILE
    w = s.Work(M=args.M, K=args.K, N=args.N,
               wbits=[[args.act] * kt for _ in range(mt)], act_bits=args.act)
    hw = s.HW(bank_size=args.bank_size, banks=args.banks, dram_bw=args.dram_bw)
    res = optimize_band(w, hw)
    if not res["feasible"]:
        print("NO FEASIBLE SCHEDULE: %s" % res["reason"]); return 1
    print("energy(variable DRAM) = %.0f   honest gap=%.1f%% (lb=%.0f)   (B,d) evaluated=%d   source=%s"
          % (res["energy"], res["gap"] * 100, res["lower_bound"], res["nodes_expanded"], res["source"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: run, expect pass**; also `cd MXP_scheduler && python band_sched.py --selftest` -> `band_sched selftest: OK`.
- [ ] **Step 5: commit**
```bash
git add MXP_scheduler/band_sched.py MXP_scheduler/test_band_sched.py
git commit -m "feat(scheduler): band diagnostics + selftest + CLI; restriction-gap vs oracle tests"
```

---

## Task 4: measure_gap.py --backend band (large T)

**Files:** Modify `MXP_scheduler/measure_gap.py`; Test `test_band_sched.py`.

- [ ] **Step 1: edits** to `measure_gap.py`:
  - `--backend` choices: add `"band"` (the `add_argument` for `--backend`).
  - Add `BAND_SHAPES = [(256, 256, 256), (512, 512, 256), (512, 512, 512)]` (T = 512, 2048, 4096).
  - Backend shape select: `shapes = ASTAR_SHAPES if args.backend=="astar" else CPSAT_SHAPES if args.backend=="cpsat" else BAND_SHAPES`.
  - `_solve` band branch:
    ```python
    if backend == "band":
        import band_sched
        r = band_sched.optimize_band(w, hw)
        return (r["energy"] if r["feasible"] else None, r["proven_optimal"],
                r["nodes_expanded"], r["gap"])
    ```
  - In `main`, for `backend=="band"` SKIP `warmstart.structural_incumbent` when `T` is large (e.g. `T > 4096`): set `warm=None` and print `warm=-`; otherwise compute it as today. (Honest gap is `r["gap"]` printed via the existing `lb%.0f` path; band is never proven so the not-proven branch always fires.)

- [ ] **Step 2: smoke test**
```python
def test_measure_gap_band_backend_smoke():
    import measure_gap
    rc = measure_gap.main(["--backend", "band", "--quick"])
    assert rc == 0
```
(If `--quick` isn't honored by the band path, mirror the existing `--quick` handling so the smoke runs only the first BAND_SHAPE + 2 capacities.)

- [ ] **Step 3: run** `cd MXP_scheduler && python -m pytest test_band_sched.py -k measure_gap -q` (pass) and visually `python measure_gap.py --backend band --quick`.
- [ ] **Step 4: commit**
```bash
git add MXP_scheduler/measure_gap.py MXP_scheduler/test_band_sched.py
git commit -m "feat(scheduler): measure_gap --backend band (large-T honest-gap rows)"
```

---

## Task 5: demo_qwen.py — Qwen Q/K/V dims -> arbitrary per-tile avg bits -> order (the deliverable)

**Files:** Create `MXP_scheduler/demo_qwen.py`; Test `test_band_sched.py`.

- [ ] **Step 1: implement**
```python
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
```

- [ ] **Step 2: smoke test**
```python
def test_demo_qwen_smoke(tmp_path):
    import demo_qwen
    out = str(tmp_path / "order.txt")
    rc = demo_qwen.main(["--model", "qwen2.5-0.5b", "--proj", "q", "--seq", "64",
                         "--bank-size", "4096", "--out", out])
    assert rc == 0
    import os
    assert os.path.exists(out) and os.path.getsize(out) > 0
```

- [ ] **Step 3: run** the smoke test (pass) and manually `cd MXP_scheduler && python demo_qwen.py --model qwen2.5-0.5b --proj q --seq 128` -> prints shape, energy, honest gap, order head/tail, writes order file.
- [ ] **Step 4: commit**
```bash
git add MXP_scheduler/demo_qwen.py MXP_scheduler/test_band_sched.py
git commit -m "feat(scheduler): demo_qwen -- Qwen Q/K/V dims + arbitrary per-tile avg bits -> band order"
```

---

## Task 6: full regression + verification

- [ ] **Step 1:** `cd MXP_scheduler && python -m pytest -q` (all prior + new tests pass; no regressions in M1/CP-SAT modules — they are untouched).
- [ ] **Step 2:** `cd MXP_scheduler && python band_sched.py --selftest` and `python demo_qwen.py --model qwen2.5-0.5b --proj q --seq 128` (both succeed).
- [ ] **Step 3:** runtime-clean check (band_sched is stdlib-only, imports no ortools): `cd MXP_scheduler && python -c "import sys, band_sched; assert 'ortools' not in sys.modules; print('band stdlib-clean: OK')"`.
- [ ] **Step 4:** `git commit -m "test(scheduler): full regression green for band scheduler" --allow-empty`.

---

## Self-review notes (author)
- Spec coverage: geometry/constructor/footprint (T1); per-region search + optimize_band + honest LB (T2); diagnostics + restriction-gap + selftest/CLI (T3); measure_gap backend (T4); Qwen demo (T5); regression (T6).
- C-spill fallback (spec §7.2) deliberately NOT implemented — unreachable in v1 (B=1,d=1 is the no-spill floor); §10's C-spill test is replaced by the capacity/stall diagnostic tests (T3). This is a documented refinement; if a reviewer wants spill exercised, it requires a forced-B>=2 constraint that v1 does not have.
- Eviction = set-difference realized via carried (last_chunk_tiles, last_band_c); `test_footprint_matches_resident_peak` (T1) is the guard that the realized peak equals the footprint formula (the C1 invariant).
- Module computes NO cost itself: scoring via `apply_cube`, authoritative energy via `eval_sched` — `test_optimize_band_parity` guards it.
- Result keys match `astar.optimize_exact` exactly; `proven_optimal=False`, honest gap; consistent with measure_gap consumption.
