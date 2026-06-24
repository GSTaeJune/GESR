# MXP_scheduler M0 (existing-model improvements) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the low-risk, immediately-useful scheduler improvements that the exact-scheduler spec depends on: separate steady-state stall from fill/drain, parameterize the cycle model for later HW calibration, and lock down the DRAM energy presets — all on the existing closed-form model with no new optimizer.

**Architecture:** Pure refactor + parameterization of `MXP_scheduler/mxp_scheduler.py`'s cycle/stall path and `hwconfig.py`'s config resolution. No new search/solver. The `mxp_scheduler_annotated.py` twin must stay numerically identical (`--crosscheck`). All changes preserve existing energy/DRAM numbers; only the *stall return shape* and *new optional cycle params* change behavior, and defaults reproduce today's results.

**Tech Stack:** Python 3.13, stdlib only, pytest. Run tests from `MXP_scheduler/`.

**Scope note:** This is M0. The `(order+eviction)` evaluator, the A* joint optimizer, and the DP-exact oracle are M1 (separate plan) — they form one coherent new-optimizer effort and are best planned once M0's stall/cycle foundation is in place. Spec: `docs/superpowers/specs/2026-06-23-mxp-scheduler-precision-adaptive-design.md`.

---

## File Structure

- `MXP_scheduler/mxp_scheduler.py` — modify `_stall_of_order`, `stall_fill`, `actual_cycle`, `evaluate`, `report`, `selftest`, `compute_work`, `_blocks`, `HW`. Source of truth.
- `MXP_scheduler/mxp_scheduler_annotated.py` — mirror every logic change identically (the `--crosscheck` twin).
- `MXP_scheduler/hwconfig.py` — modify `resolve` to pass new cycle params; modify `_TOP_KEYS`/validation for an optional `cycles` config block.
- `MXP_scheduler/test_mxp_scheduler.py` — reset stall goldens, add new contract tests.
- `MXP_scheduler/test_hwconfig.py` — add cycle-param + DRAM-preset validation tests.

**Twin rule:** Any edit to a numeric function in `mxp_scheduler.py` MUST be mirrored in `mxp_scheduler_annotated.py`. Run `python mxp_scheduler.py --crosscheck` after each task; it must print `crosscheck: OK`.

---

## Task 1: Separate steady-state stall from fill/drain

**Why:** The current `stall` lumps mid-stream (steady-state) stall with the unavoidable trailing C drain, so a `steady_stall == 0` feasibility gate (the spec's stall=0 constraint, enforced later in M1) is impossible to express. This task splits them and surfaces `steady_stall` in `evaluate`/`report`. No enforcement yet — observability only.

**Files:**
- Modify: `MXP_scheduler/mxp_scheduler.py` (`_stall_of_order:286`, `stall_fill:328`, `actual_cycle:342`, `evaluate:347`, `report:399`, `lpt_headroom:387`, `crosscheck:504`, `selftest:425`)
- Modify: `MXP_scheduler/mxp_scheduler_annotated.py` (same functions + its `explain()` and `lpt_headroom`; note: the annotated twin has **no** `selftest`)
- Test: `MXP_scheduler/test_mxp_scheduler.py`

> **Caller inventory (every place that unpacks `stall_fill` or calls `_stall_of_order` — ALL must change):** `mxp_scheduler.py`: `actual_cycle:343`, `evaluate:355`, `lpt_headroom:394-395` (calls `_stall_of_order` and subtracts → must `sum()` the new 2-tuple), `selftest:425`. `mxp_scheduler_annotated.py`: same + `explain()`. Tests: `test_stall_fill_g3`, `test_stall_zero_when_bw_huge`, `test_stall_includes_c_drain`, **`test_no_double_buffer_exposes_fetch:456`**, **`test_freq_ratio_scales_dram_time:496`**, **`test_evaluate_shared_walk_matches_public_functions:610`**. Missing any of these lands a failing test.

- [ ] **Step 1: Update the golden tests to the new 3-value stall contract**

`stall_fill` will return `(fill, steady_stall, drain)` instead of `(stall, fill)`. Edit these existing tests in `test_mxp_scheduler.py`:

Replace `test_stall_fill_g3` body (currently lines ~186-201):

```python
def test_stall_fill_g3():
    # G3: 2-bit weights -> tiny compute (256/block) but FP32 C psum dominates the DRAM stall.
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 2]], act_bits=8)
    m = s.Mapping(perm=("N", "M", "K"), m_in=2, k_in=2, n_in=1)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    fill, steady, drain = s.stall_fill(m, w, hw)
    # fill = (a0+w0)/bw = (16384+8192)/32 = 768
    # boundary 0->1: A reload(16384)+C write(65536); demand=81920; steady = 81920/32 - 256 = 2304
    # trailing drain = c_blk(65536)/32 = 2048
    assert fill == 768.0
    assert steady == 2304.0
    assert drain == 2048.0
    # actual = compute_work + fill + steady + drain = 512 + 768 + 2304 + 2048 = 5632
    assert s.actual_cycle(m, w, hw) == 5632.0
```

Replace `test_stall_zero_when_bw_huge` body (currently ~204-210):

```python
def test_stall_zero_when_bw_huge():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("M", "K", "N"), m_in=2, k_in=2, n_in=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)   # effectively infinite BW
    fill, steady, drain = s.stall_fill(m, w, hw)
    assert steady < 1.0
    assert drain < 1.0
    assert fill < 1.0
```

Replace `test_stall_includes_c_drain` body (currently ~407-414):

```python
def test_stall_includes_c_drain():
    # single all-resident block: no mid-stream stall, only the unavoidable trailing C drain
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("M", "K", "N"), m_in=2, k_in=2, n_in=2)   # single block
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    fill, steady, drain = s.stall_fill(m, w, hw)
    c_blk = 2 * 2 * 1024 * 32          # 131072
    assert steady == 0.0               # one block -> no boundary -> no mid-stream stall
    assert drain == c_blk / 32         # trailing C drain only (4096.0)
```

Also fix the THREE existing tests that unpack the old 2-tuple (they break otherwise).

Replace `test_no_double_buffer_exposes_fetch` body (currently ~448-458) — its last 3 lines:

```python
    assert s.feasible(m, w, nodb)                                # still feasible (footprint=90112 <= 102400)
    _f1, steady_db, _d1 = s.stall_fill(m, w, db)
    _f2, steady_no, _d2 = s.stall_fill(m, w, nodb)
    assert steady_no > steady_db                                 # no overlap -> strictly more mid-stream stall
```

Replace `test_freq_ratio_scales_dram_time` body (currently ~489-502) from the `sb, fb = ...` line down:

```python
    fb, steady_b, drain_b = s.stall_fill(m, w, base)
    fs, steady_s, drain_s = s.stall_fill(m, w, slow)
    sb, ss = steady_b + drain_b, steady_s + drain_s
    assert fs == pytest.approx(2 * fb)               # fill is pure DRAM transfer -> exactly 2x
    assert ss > sb                                   # more un-hidden DRAM time -> more stall
    # compute term is freq-independent: actual_cycle minus the DRAM (fill+stall) part is equal
    assert s.actual_cycle(m, w, slow) - (ss + fs) == pytest.approx(s.actual_cycle(m, w, base) - (sb + fb))
    assert s.energy_breakdown(m, w, base) == s.energy_breakdown(m, w, slow)  # energy freq-independent
```

Replace `test_evaluate_shared_walk_matches_public_functions` body (currently ~600-611) from the `stall, fill = ...` line down:

```python
        fill, steady, drain = s.stall_fill(m, w, hw)
        assert (r["fill"], r["steady_stall"], r["drain"]) == (fill, steady, drain)
        assert r["stall"] == steady + drain
```

- [ ] **Step 2: Add a new test for the steady_stall observability in evaluate**

Append to `test_mxp_scheduler.py`:

```python
def test_evaluate_exposes_steady_stall_and_drain():
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 2]], act_bits=8)
    m = s.Mapping(perm=("N", "M", "K"), m_in=2, k_in=2, n_in=1)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    r = s.evaluate(m, w, hw)
    assert r["steady_stall"] == 2304.0
    assert r["drain"] == 2048.0
    assert r["fill"] == 768.0
    # back-compat: r["stall"] stays the combined steady+drain
    assert r["stall"] == 2304.0 + 2048.0
    assert r["actual_cycle"] == float(s.compute_work(w)) + 768.0 + 2304.0 + 2048.0
```

- [ ] **Step 3: Run the updated tests to verify they fail**

Run: `python -m pytest test_mxp_scheduler.py -k "stall or steady" -v`
Expected: FAIL — `stall_fill` still returns a 2-tuple, so unpacking `fill, steady, drain` raises `ValueError: not enough values to unpack`, and `evaluate` has no `steady_stall` key.

- [ ] **Step 4: Implement the split in `mxp_scheduler.py`**

Replace `_stall_of_order` (`:286-325`) return logic — accumulate into `steady`, return `(steady, drain)`:

```python
def _stall_of_order(blocks, bw, double_buffered=True):
    """Return (steady_stall, drain) for a given block ORDER. steady_stall = mid-stream
    transfer time not hidden by neighbouring compute (the part a stall=0 schedule must
    drive to zero). drain = the last tile's C write, which has no following compute to hide
    it (always unavoidable). Splitting them lets a steady_stall==0 feasibility gate exist."""
    n = len(blocks)
    if n == 0:
        return 0.0, 0.0
    seen = set()
    prev = None
    prev_compute = 0.0
    steady = 0.0
    for idx, compute_blk, a_blk, w_blk, c_blk in blocks:
        mn = (idx["M"], idx["N"])
        if prev is not None:
            prev_mn = (prev["M"], prev["N"])
            fetch = 0.0
            if (idx["K"], idx["N"]) != (prev["K"], prev["N"]):
                fetch += a_blk
            if (idx["M"], idx["K"]) != (prev["M"], prev["K"]):
                fetch += w_blk
            if mn != prev_mn:
                fetch += c_blk
                if mn in seen:
                    fetch += c_blk
            hide = prev_compute if double_buffered else 0.0
            steady += max(0.0, fetch / bw - hide)
        seen.add(mn)
        prev = idx
        prev_compute = compute_blk
    drain = blocks[-1][4] / bw
    return steady, drain
```

Replace `stall_fill` (`:328-339`):

```python
def stall_fill(m, w, hw, blocks=None):
    """Return (fill, steady_stall, drain). fill = first block's input fetch (unhidable
    startup). steady_stall = mid-stream un-hidden transfer (stall=0 target). drain =
    trailing C write. blocks: optional precomputed list(_blocks(m, w))."""
    if blocks is None:
        blocks = list(_blocks(m, w))
    bw = hw.eff_bw
    _, _, a0, w0, _c0 = blocks[0]
    fill = (a0 + w0) / bw
    steady, drain = _stall_of_order(blocks, bw, _double_buffered(m, w, hw, blocks))
    return fill, steady, drain
```

Replace `actual_cycle` (`:342-344`):

```python
def actual_cycle(m, w, hw):
    fill, steady, drain = stall_fill(m, w, hw)
    return float(compute_work(w)) + fill + steady + drain
```

In `evaluate` (`:347-367`), replace the `stall, fill = stall_fill(...)` line and the returned dict's stall fields:

```python
    fill, steady, drain = stall_fill(m, w, hw, blocks)
    cw = compute_work(w)
    return {
        "mapping": m,
        "feasible": feas,
        "energy": eb["total"],
        "energy_breakdown": eb,
        "dram": d,
        "compute_work": cw,
        "fill": fill,
        "steady_stall": steady,
        "drain": drain,
        "stall": steady + drain,          # back-compat combined value
        "actual_cycle": float(cw) + fill + steady + drain,
    }
```

In `report` (`:399-407`), add a `steady` column. Replace the header and row f-strings:

```python
    lines = [f"# GEMM ({w.M}x{w.K}x{w.N})  cap_bits={hw.cap_bits}  dram_bw={hw.dram_bw}"
             f"  freq_ratio={hw.freq_ratio}  eff_bw={hw.eff_bw}",
             f"{'perm':<10}{'m_in':>5}{'k_in':>5}{'n_in':>5}{'energy':>16}"
             f"{'actual_cycle':>16}{'steady':>10}{'drain':>10}"]
    for r in ranked[:top]:
        m = r["mapping"]
        lines.append(f"{''.join(m.perm):<10}{m.m_in:>5}{m.k_in:>5}{m.n_in:>5}"
                     f"{r['energy']:>16.0f}{r['actual_cycle']:>16.1f}"
                     f"{r['steady_stall']:>10.1f}{r['drain']:>10.1f}")
```

In `selftest` (`:425-426`), update the G3 assertion to the new shape:

```python
    assert stall_fill(m3, w3, hw3) == (768.0, 2304.0, 2048.0)   # (fill, steady, drain)
    assert actual_cycle(m3, w3, hw3) == 5632.0
```

Fix `lpt_headroom` (`:387-396`) — it calls `_stall_of_order` directly and subtracts, so it breaks on the new 2-tuple. Wrap each call in `sum(...)`:

```python
def lpt_headroom(m, w, hw):
    """Headroom indicator: natural perm-order stall vs LPT-reordered stall (both = steady+drain)."""
    blocks = list(_blocks(m, w))
    db = _double_buffered(m, w, hw, blocks)
    natural = sum(_stall_of_order(blocks, hw.eff_bw, db))
    lpt = sum(_stall_of_order(sorted(blocks, key=lambda b: -b[1]), hw.eff_bw, db))  # b[1]=compute_blk desc
    return {"natural_stall": natural, "lpt_stall": lpt, "headroom": natural - lpt}
```

Extend `crosscheck`'s compared-keys tuple (`:504`) so the new evaluate keys are covered by twin parity:

```python
            for key in ("feasible", "energy", "actual_cycle", "stall", "fill", "steady_stall", "drain"):
```

- [ ] **Step 5: Mirror every change in `mxp_scheduler_annotated.py`**

Apply the identical edits (same function bodies; keep the annotated file's 한국어 주석 around them) to `_stall_of_order`, `stall_fill`, `actual_cycle`, `evaluate`, `report`, and `lpt_headroom`. The annotated twin has **no `selftest`** (skip that), but it DOES have an `explain()` function that unpacks `stall_fill` — fix it:

```python
    # in explain(): was `stall, fill = stall_fill(m, w, hw)`
    fill, steady, drain = stall_fill(m, w, hw)
    stall = steady + drain
```

The numbers must match `mxp_scheduler.py` exactly. (The annotated `crosscheck` lives only in `mxp_scheduler.py`, not the twin — no mirror needed for that.)

- [ ] **Step 6: Run tests + selftest + crosscheck to verify they pass**

Run: `python -m pytest test_mxp_scheduler.py -v`
Expected: PASS (all, including the 4 edited/added stall tests).

Run: `python mxp_scheduler.py --selftest`
Expected: `selftest: OK`

Run: `python mxp_scheduler.py --crosscheck`
Expected: `crosscheck: OK`

> Note: `crosscheck` compares a fixed tuple of `evaluate()` keys between twins (`mxp_scheduler.py:504`). Step 4 extends that tuple to include `steady_stall`/`drain`, so the twin's `evaluate` must produce identical values for them — that's why Step 5 must mirror exactly. `crosscheck` also calls `lpt_headroom` (`:508`), so the `sum(...)` fix must be in both files.

- [ ] **Step 7: Commit**

```bash
git add MXP_scheduler/mxp_scheduler.py MXP_scheduler/mxp_scheduler_annotated.py MXP_scheduler/test_mxp_scheduler.py
git commit -m "feat(scheduler): split steady_stall from fill/drain (stall=0 observability)"
```

---

## Task 2: Parameterize the cycle model (cycles_per_bit, SA fill/drain)

**Why:** stall=0 feasibility (enforced in M1) must eventually be grounded in HW-measured cycles, not just the analytic formula. This adds injectable parameters with defaults that reproduce today's numbers, so HW `[CYC]` measurements plug in later via config with zero behavior change now.

**Files:**
- Modify: `MXP_scheduler/mxp_scheduler.py` (`HW:18` + its `__post_init__` validation, `compute_work:218` signature, `actual_cycle:342`, `evaluate:347`, `main:567`). NOTE: `_blocks`/`stall_fill` are **not** changed here — M0 leaves the per-block hide budget unscaled (see the hide-budget note in Step 3); `compute_work` is the only compute formula that gains the `cycles_per_bit` factor.
- Modify: `MXP_scheduler/mxp_scheduler_annotated.py` (mirror `HW`, `compute_work`, `actual_cycle`, `evaluate`, `main`)
- Modify: `MXP_scheduler/hwconfig.py` (`resolve:228`, `_TOP_KEYS:17`, `load_config:24`)
- Test: `MXP_scheduler/test_mxp_scheduler.py`, `MXP_scheduler/test_hwconfig.py`

- [ ] **Step 1: Write failing tests for the new cycle params**

Append to `test_mxp_scheduler.py`:

```python
def test_cycle_params_default_unchanged():
    # defaults (cycles_per_bit=1, sa_fill=sa_drain=0) reproduce today's actual_cycle
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 2]], act_bits=8)
    m = s.Mapping(perm=("N", "M", "K"), m_in=2, k_in=2, n_in=1)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    assert hw.cycles_per_bit == 1.0
    assert hw.sa_fill_cycles == 0
    assert hw.sa_drain_cycles == 0
    assert s.actual_cycle(m, w, hw) == 5632.0   # same as Task 1 golden

def test_cycle_params_scale_compute_and_add_latency():
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 2]], act_bits=8)
    m = s.Mapping(perm=("N", "M", "K"), m_in=2, k_in=2, n_in=1)
    base = s.HW(bank_size=1024, banks=32, dram_bw=32)
    # compute_work doubles with cycles_per_bit=2; SA fill/drain add flat cycles
    hw2 = s.HW(bank_size=1024, banks=32, dram_bw=32,
               cycles_per_bit=2.0, sa_fill_cycles=10, sa_drain_cycles=20)
    assert s.compute_work(w, hw2.cycles_per_bit) == 2 * s.compute_work(w, base.cycles_per_bit)
    # actual_cycle gains 2x compute over base-compute, plus 30 SA latency
    base_cycle = s.actual_cycle(w=w, m=m, hw=base)
    assert s.actual_cycle(m, w, hw2) == (
        base_cycle + s.compute_work(w) + 30)   # +1x compute (2x-1x) + 10 + 20

def test_cycle_params_reject_negative():
    import pytest
    with pytest.raises(ValueError):
        s.HW(1024, 32, 64, cycles_per_bit=0)
    with pytest.raises(ValueError):
        s.HW(1024, 32, 64, sa_fill_cycles=-1)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest test_mxp_scheduler.py -k cycle -v`
Expected: FAIL — `HW.__init__` has no `cycles_per_bit`/`sa_fill_cycles`/`sa_drain_cycles`.

- [ ] **Step 3: Add the params to `HW` and thread `cycles_per_bit` through compute**

In `HW` (`mxp_scheduler.py:18`), add fields after `coeffs`:

```python
    cycles_per_bit: float = 1.0   # on-chip cycles per weight-bit-plane (bit-serial); HW-calibrated
    sa_fill_cycles: int = 0       # SA pipeline fill latency (flat); 0 = current model
    sa_drain_cycles: int = 0      # SA pipeline drain latency (flat); 0 = current model
```

In `HW.__post_init__`, add validation (after the existing checks):

```python
        if self.cycles_per_bit <= 0:
            raise ValueError(f"cycles_per_bit must be positive, got {self.cycles_per_bit}")
        for name, v in (("sa_fill_cycles", self.sa_fill_cycles),
                        ("sa_drain_cycles", self.sa_drain_cycles)):
            if v < 0:
                raise ValueError(f"{name} must be non-negative, got {v}")
```

Change `compute_work` (`:218`) to accept the scale:

```python
def compute_work(w, cycles_per_bit=1.0):
    # order-independent ideal: cycles_per_bit * TILE * NT * Sum wbits. Exact (no int()).
    return cycles_per_bit * TILE * w.NT * sum(sum(row) for row in w.wbits)
```

Change `actual_cycle` (`:342`, edited in Task 1) to scale compute and add SA latency:

```python
def actual_cycle(m, w, hw):
    fill, steady, drain = stall_fill(m, w, hw)
    return (float(compute_work(w, hw.cycles_per_bit)) + hw.sa_fill_cycles
            + fill + steady + drain + hw.sa_drain_cycles)
```

In `evaluate` (`:347`, edited in Task 1), update the compute + actual_cycle lines to use the scaled compute:

```python
    cw = compute_work(w, hw.cycles_per_bit)
    ...
        "compute_work": cw,
        ...
        "actual_cycle": float(cw) + hw.sa_fill_cycles + fill + steady + drain + hw.sa_drain_cycles,
```

> **Hide-budget scaling — read this, the reasoning matters.** M0 scales the order-independent `compute_work` TOTAL by `cycles_per_bit` but leaves the per-block `compute_blk` *hide budget* inside `_stall_of_order` at `cycles_per_bit=1`. This is a deliberate **conservative approximation, not a unit-cancellation identity**: the hide budget `prev_compute` and the transfer time `fetch/eff_bw` are already in the same unit (on-chip cycles), so if compute genuinely takes `cycles_per_bit×` longer, the latency-hiding window also grows and `steady_stall` *should decrease*. M0 does NOT grow the hide budget, so when `cycles_per_bit>1` the reported `steady_stall` is **pessimistic (over-stated)** — safe for an observability milestone, and inert at the default `cycles_per_bit=1.0` (today's numbers unchanged). Add exactly this comment at `actual_cycle`:
> ```python
> # NOTE: cycles_per_bit scales the compute_work TOTAL but NOT the per-block hide budget in
> # _stall_of_order, so steady_stall is conservative (over-stated) when cycles_per_bit>1.
> # M1's stall=0 feasibility gate must scale the per-block hide budget consistently (else it
> # would wrongly mark feasible schedules infeasible). Inert at the default cycles_per_bit=1.0.
> ```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest test_mxp_scheduler.py -k cycle -v`
Expected: PASS.

- [ ] **Step 5: Thread cycle params through hwconfig**

In `hwconfig.py`, add `"cycles"` to `_TOP_KEYS` (`:17`):

```python
_TOP_KEYS = {"sram", "dram", "chip_freq_mhz", "coeffs", "cacti_bin", "cycles"}
```

In `load_config` (`:24`), after the coeffs block, accept an optional `cycles` dict with known keys:

```python
    cfg.setdefault("cycles", {})
    _CYCLE_KEYS = {"cycles_per_bit", "sa_fill_cycles", "sa_drain_cycles"}
    unknown = set(cfg["cycles"]) - _CYCLE_KEYS
    if unknown:
        raise ValueError(f"config cycles has unknown key(s) {sorted(unknown)}; valid: {sorted(_CYCLE_KEYS)}")
```

In `resolve` (`:228`), add the cycle params to the returned kwargs (defaults if absent):

```python
    cyc = cfg["cycles"]
    return {"bank_size": sram["bank_size"], "banks": sram["banks"],
            "word_bits": sram["word_bits"], "dram_bw": float(d["dram_bw"]),
            "freq_ratio": chip / d["dram_freq_mhz"], "coeffs": coeffs,
            "cycles_per_bit": cyc.get("cycles_per_bit", 1.0),
            "sa_fill_cycles": cyc.get("sa_fill_cycles", 0),
            "sa_drain_cycles": cyc.get("sa_drain_cycles", 0)}
```

In `mxp_scheduler.py` `main()` (`:567-573`), the `HW(...)` call currently ENDS with `coeffs=coeffs)`. Change that closing `)` to a comma and append the three kwargs — replace the whole call:

```python
        hw = HW(bank_size=pick(a.bank_size, "bank_size", 1024),
                banks=pick(a.banks, "banks", 32),
                dram_bw=pick(a.dram_bw, "dram_bw", 64.0),
                word_bits=cfg_kw["word_bits"] if cfg_kw is not None else 32,
                freq_ratio=pick(a.freq_ratio, "freq_ratio", 1.0),
                coeffs=coeffs,
                cycles_per_bit=cfg_kw["cycles_per_bit"] if cfg_kw is not None else 1.0,
                sa_fill_cycles=cfg_kw["sa_fill_cycles"] if cfg_kw is not None else 0,
                sa_drain_cycles=cfg_kw["sa_drain_cycles"] if cfg_kw is not None else 0)
```

Apply the **same** `HW(...)` replacement in `mxp_scheduler_annotated.py` `main()` (its identical call, ~`:573`) so a `cycles` config block is honored by both CLIs (twin parity).

- [ ] **Step 6: Write + run hwconfig test**

Append to `test_hwconfig.py`:

```python
def test_resolve_passes_cycle_params(tmp_path, monkeypatch):
    import hwconfig as hc
    cfg = {"sram": {"bank_size": 1024, "banks": 32}, "dram": "LPDDR5-6400_x16",
           "chip_freq_mhz": 250.0, "cycles": {"cycles_per_bit": 2.0, "sa_fill_cycles": 5}}
    p = tmp_path / "hw.json"
    p.write_text(__import__("json").dumps(cfg))
    loaded = hc.load_config(str(p))
    kw = hc.resolve(loaded, runner=lambda *a, **k: {"onchip_pj_per_bit": 0.85, "sram_max_freq_mhz": 900.0})
    assert kw["cycles_per_bit"] == 2.0
    assert kw["sa_fill_cycles"] == 5
    assert kw["sa_drain_cycles"] == 0   # default

def test_load_config_rejects_unknown_cycle_key(tmp_path):
    import hwconfig as hc, pytest
    cfg = {"sram": {"bank_size": 1, "banks": 1}, "dram": "X", "chip_freq_mhz": 1,
           "cycles": {"cyclez_per_bit": 2.0}}
    p = tmp_path / "hw.json"
    p.write_text(__import__("json").dumps(cfg))
    with pytest.raises(ValueError):
        hc.load_config(str(p))
```

Run: `python -m pytest test_hwconfig.py -k cycle -v`
Expected: PASS.

- [ ] **Step 7: Mirror in annotated twin + verify selftest/crosscheck**

Apply the `HW` fields + validation, `compute_work` signature, `actual_cycle`, and `evaluate` changes to `mxp_scheduler_annotated.py`.

Run: `python mxp_scheduler.py --selftest && python mxp_scheduler.py --crosscheck`
Expected: `selftest: OK` then `crosscheck: OK`.

- [ ] **Step 8: Commit**

```bash
git add MXP_scheduler/mxp_scheduler.py MXP_scheduler/mxp_scheduler_annotated.py MXP_scheduler/hwconfig.py MXP_scheduler/test_mxp_scheduler.py MXP_scheduler/test_hwconfig.py
git commit -m "feat(scheduler): parameterize cycle model (cycles_per_bit, SA fill/drain) for HW calibration"
```

---

## Task 3: Lock down DRAM presets (validation + provenance)

**Why:** The `dram_presets.json` `pj_per_bit` values are energy-ranking-critical and must each carry a sourced provenance. Add tests that enforce the schema (required fields, positive values, non-empty `source`) so a future preset edit can't silently ship an unsourced or malformed coefficient.

**Files:**
- Test: `MXP_scheduler/test_hwconfig.py`
- Modify (only if a preset fails the new schema test): `MXP_scheduler/dram_presets.json`

- [ ] **Step 1: Write the failing schema-enforcement test**

Append to `test_hwconfig.py`:

```python
def test_dram_presets_schema_and_provenance():
    import json, hwconfig as hc
    presets = json.load(open(hc.DEFAULT_PRESETS))
    assert presets, "dram_presets.json must be non-empty"
    for name, p in presets.items():
        assert set(p) >= {"data_rate_mts", "bus_bits", "pj_per_bit", "source"}, \
            f"{name} missing required field(s)"
        assert p["data_rate_mts"] > 0 and p["bus_bits"] > 0 and p["pj_per_bit"] > 0, \
            f"{name} has non-positive numeric field"
        assert isinstance(p["source"], str) and len(p["source"].strip()) >= 20, \
            f"{name} source provenance too short/missing"

def test_dram_params_derivation_matches_convention():
    import hwconfig as hc
    d = hc.dram_params("LPDDR5-6400_x16")
    # spec convention: f_dram = data_rate/2 (DDR bus clock); dram_bw = 2*bus_bits
    assert d["dram_bw"] == 2 * 16
    assert d["dram_freq_mhz"] == 6400 / 2.0
    assert d["pj_per_bit"] == 9.0
```

- [ ] **Step 2: Run to verify the derivation test FAILS (stale presets)**

Run: `python -m pytest test_hwconfig.py -k dram -v`
Expected: `test_dram_params_derivation_matches_convention` FAILS with `assert 5.0 == 9.0`. The shipped `dram_presets.json` still carries the **2026-06-10 seed values** (device-internal boundary: LPDDR5 5.0 / LPDDR5X 4.5 / DDR4 15.0 / DDR5 10.0), but `docs/dram-energy/README.md` (확정일 2026-06-23) re-derived all four on a **unified full-system boundary** and lands on 9.0 / 7.5 / 20.0 / 14.0. The JSON was never updated to match its own finalized provenance — this task fixes that. (`test_dram_presets_schema_and_provenance` may pass already; the derivation test is the red one.)

- [ ] **Step 3: Update `dram_presets.json` to the finalized full-system values**

Replace the entire file with the README-aligned values + full-system provenance (do NOT weaken the test; the test encodes the authoritative value):

```json
{
  "LPDDR5-6400_x16":  {"data_rate_mts": 6400, "bus_bits": 16, "pj_per_bit": 9.0,
                       "source": "full-system boundary (device core + on-die I/O + SoC PHY/controller + refresh). Anchor: Ha 2018 (Stanford PhD, refs/Ha2018) Fig 4.8/4.9 LPDDR4 ~8.5-13 pJ/b device-incl-I/O + ~2-2.5 SoC PHY/controller. Cross-check: O'Connor MICRO 2017 HBM2 3.97 pJ/b (LPDDR5 ~2.3x). See docs/dram-energy/README.md (2026-06-23)."},
  "LPDDR5X-8533_x16": {"data_rate_mts": 8533, "bus_bits": 16, "pj_per_bit": 7.5,
                       "source": "full-system. LPDDR5 9.0 x ~0.83 LPDDR5X efficiency gain (higher data rate at lower energy/bit). See docs/dram-energy/README.md (2026-06-23)."},
  "DDR4-3200_x64":    {"data_rate_mts": 3200, "bus_bits": 64, "pj_per_bit": 20.0,
                       "source": "full-system, secondary confidence (LPDDR5/5X are the validated focus). Wide terminated x64 bus + memory controller dominate: device-incl-I/O ~15 (O'Connor GDDR5 14.0) + SoC controller. See docs/dram-energy/README.md (2026-06-23)."},
  "DDR5-4800_x64":    {"data_rate_mts": 4800, "bus_bits": 64, "pj_per_bit": 14.0,
                       "source": "full-system, secondary confidence. DDR4 20 x VDD 1.2->1.1V (~0.84 dynamic) + on-die ECC / DFE overhead. See docs/dram-energy/README.md (2026-06-23)."}
}
```

- [ ] **Step 4: Run to verify the dram tests pass**

Run: `python -m pytest test_hwconfig.py -k dram -v`
Expected: PASS (both schema + derivation). The `pj_per_bit` are now 9.0/7.5/20.0/14.0 and every `source` is full-system provenance (≥20 chars).

- [ ] **Step 5: Add a docs cross-reference comment**

In `hwconfig.py` at `DEFAULT_PRESETS` (`:14`), append a comment line:

```python
DEFAULT_PRESETS = _HERE / "dram_presets.json"   # provenance + derivation: see docs/dram-energy/ ; schema locked by test_dram_presets_schema_and_provenance
```

- [ ] **Step 6: Commit**

```bash
git add MXP_scheduler/test_hwconfig.py MXP_scheduler/hwconfig.py MXP_scheduler/dram_presets.json
git commit -m "fix(scheduler): update DRAM presets to finalized full-system pj/bit + lock schema"
```

---

## Final verification (run after all tasks)

- [ ] Run full suites + embedded checks:

Run: `python -m pytest test_mxp_scheduler.py test_hwconfig.py -q`
Expected: all PASS.

Run: `python mxp_scheduler.py --selftest && python mxp_scheduler.py --crosscheck`
Expected: `selftest: OK` then `crosscheck: OK`.

Run: `python mxp_scheduler.py --M 128 --K 128 --N 128 --bank-size 1024 --banks 32 --dram-bw 64 --act 8`
Expected: a ranked table that now includes `steady` and `drain` columns; energy numbers unchanged from before M0.

---

## Self-review notes (author)

- **Spec coverage:** M0 scope = stall split (spec §1.3/§6 fill/steady/drain), cycle parameterization (spec §9, §1 item 4 / D7), DRAM coeffs finalization (spec §9, D8). The `(order+eviction)` evaluator, A*, and DP oracle (spec §5-§7, M1) are deliberately deferred to the M1 plan — flagged in the Scope note.
- **M3 observability:** Task 1 surfaces `steady_stall` in `evaluate`/`report`, making the spec §8 "feasible region non-empty" numbers (12/42/119) computable directly (count mappings with `steady_stall == 0`), which becomes M1's regression gate.
- **Twin parity:** every numeric edit is mirrored and gated by `--crosscheck`.
- **No enforcement in M0:** stall=0 is only *observed* here; enforcement (feasibility gate) is M1 — this is the spec's M3 fix (avoid a broken intermediate).
