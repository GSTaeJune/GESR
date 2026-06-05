# MXP_scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a software cost-model + optimizer that, given SRAM bank size/count and a per-32×32-tile average-weight-bit map, outputs the lowest-energy GEMM mapping (outer loop permutation + on-chip blocking) for the MXP bit-serial accelerator, with a sequence-aware stall (cycle) model.

**Architecture:** Pure-Python, stdlib-only. A standard file `mxp_scheduler.py` built function-by-function via TDD (pytest dev tests), then mirrored into a heavily-commented twin `mxp_scheduler_annotated.py` that produces byte-identical results (guarded by `--crosscheck`). The evaluator is a pure event counter (no energy weighting); energy = counts·coeffs is a separate final step; cycles = deterministic compute-work + sequence-aware stall.

**Tech Stack:** Python 3 (stdlib: dataclasses, itertools, argparse, json, math). pytest for dev-time TDD only (not a runtime dependency). matplotlib optional (guarded import) for Pareto plot.

**Spec:** `docs/superpowers/specs/2026-06-04-mxp-scheduler-design.md`. **Prior-art rationale:** `docs/research/2026-06-04-mapping-scheduler-prior-art.md`.

**Key modeling decisions locked from the spec:**
- Tensor reuse: A=[K,N] (reused over M), W=[M,K] (reused over N), C=[M,N] (accumulated over K). (spec §4.2)
- DRAM reload factors: `reload(A)=M_out`, `reload(W)=N_out`, `reload(C)=K_out`. (spec §6.2)
- `compute_work = Σ_cube 32·b = 32·N_T·Σ_{m,k} wbits[m][k]` — order-independent **ideal lower bound**. (spec §6.4)
- **Actual cycle = compute_work + fill + Σ stall**; stall is order-dependent. The stall model is **sequence-aware (inputs A,W only for latency-hiding)**, NOT a roofline. C psum traffic lives in the energy/DRAM model only (C-in-stall is a Phase-2 refinement). (spec §8)
- "32×32" = the physical array (spatial mapping), hardware-fixed, NOT searched. Search space = (≤6 perms) × (divisor blocking) — thousands even for huge matrices. (spec §9)

---

## File Structure

```
MXP_scheduler/
    mxp_scheduler.py            # [표준본] clean implementation (built in Tasks 1–9)
    mxp_scheduler_annotated.py  # [주석 상세본] mirror w/ line-by-line 한국어 comments + --explain (Task 10)
    test_mxp_scheduler.py       # pytest dev tests (TDD driver; not shipped runtime dep)
```

- `mxp_scheduler.py` is **importable** (all logic at module level) AND has a CLI (`if __name__ == "__main__"`).
- Embedded `--selftest` re-runs the golden asserts with plain `assert` (no pytest) so the shipped program self-validates.
- `--crosscheck` imports both files and asserts identical `evaluate()` output across a sweep (drift guard).
- Both `.py` files are self-contained (read top-to-bottom; the annotated one does NOT import the standard one).

**Conventions used in every task (do not rename later):**
- `TILE = 32`
- `DEFAULT_COEFFS = {"dram": 200.0, "onchip": 6.0, "mac": 1.0, "rmw": 5.0}`
- Classes: `HW(bank_size, banks, dram_bw, word_bits=32, coeffs=DEFAULT_COEFFS)`, `Work(M, K, N, wbits, act_bits)`, `Mapping(perm, m_in, k_in, n_in)`.
- Functions: `divisors`, `gen_mappings`, `footprint_bits`, `feasible`, `dram_bits`, `onchip_bits`, `mac_ops`, `rmw_ops`, `compute_work`, `stall_fill`, `actual_cycle`, `energy_breakdown`, `evaluate`, `optimize`, `lpt_headroom`, `report`, `selftest`, `crosscheck`, `main`.

---

## Task 1: Scaffold + HW/Work dataclasses + divisors + validation

**Files:**
- Create: `MXP_scheduler/mxp_scheduler.py`
- Test: `MXP_scheduler/test_mxp_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
# MXP_scheduler/test_mxp_scheduler.py
import math
import pytest
import mxp_scheduler as s


def test_tile_counts_and_total_w_bits():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    assert (w.MT, w.KT, w.NT) == (2, 2, 2)
    # total_w_bits = TILE*TILE * sum(all wbits) = 1024 * 32 = 32768
    assert w.total_w_bits == 32768


def test_divisors():
    assert s.divisors(1) == [1]
    assert s.divisors(4) == [1, 2, 4]
    assert s.divisors(32) == [1, 2, 4, 8, 16, 32]


def test_work_validates_wbits_shape():
    with pytest.raises(ValueError):
        s.Work(M=64, K=64, N=64, wbits=[[8, 8]], act_bits=8)  # wrong rows (MT=2)
    with pytest.raises(ValueError):
        s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=3)  # act not in {2,4,8}


def test_hw_capacity_bits():
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)
    assert hw.cap_bits == 1024 * 32 * 32  # bank_size*banks*word_bits = 1048576
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mxp_scheduler'` (file not created yet).

- [ ] **Step 3: Write minimal implementation**

```python
# MXP_scheduler/mxp_scheduler.py
"""MXP_scheduler — GEMM mapping cost-model + optimizer for the MXP bit-serial SA.
Spec: docs/superpowers/specs/2026-06-04-mxp-scheduler-design.md
Standard (clean) build. stdlib only.
"""
from dataclasses import dataclass, field
import itertools

TILE = 32
DEFAULT_COEFFS = {"dram": 200.0, "onchip": 6.0, "mac": 1.0, "rmw": 5.0}


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


@dataclass
class HW:
    bank_size: int          # words per bank
    banks: int
    dram_bw: float          # bits per cycle
    word_bits: int = 32     # FP32 psum word
    coeffs: dict = field(default_factory=lambda: dict(DEFAULT_COEFFS))

    @property
    def cap_bits(self):
        return self.bank_size * self.banks * self.word_bits


@dataclass
class Work:
    M: int
    K: int
    N: int
    wbits: list             # MT x KT, average weight bits per 32x32 tile, each in [2,8]
    act_bits: int           # layer-uniform activation precision (2/4/8)

    def __post_init__(self):
        if self.act_bits not in (2, 4, 8):
            raise ValueError(f"act_bits must be 2/4/8, got {self.act_bits}")
        if len(self.wbits) != self.MT or any(len(r) != self.KT for r in self.wbits):
            raise ValueError(f"wbits must be {self.MT}x{self.KT}")

    @property
    def MT(self):
        return -(-self.M // TILE)   # ceil

    @property
    def KT(self):
        return -(-self.K // TILE)

    @property
    def NT(self):
        return -(-self.N // TILE)

    @property
    def total_w_bits(self):
        return TILE * TILE * sum(sum(row) for row in self.wbits)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/mxp_scheduler.py MXP_scheduler/test_mxp_scheduler.py
git commit -m "feat(scheduler): HW/Work dataclasses + divisors + validation"
```

---

## Task 2: Mapping enumeration + footprint + feasibility

**Files:**
- Modify: `MXP_scheduler/mxp_scheduler.py`
- Test: `MXP_scheduler/test_mxp_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
def test_gen_mappings_count():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    ms = s.gen_mappings(w)
    # 6 perms x divisors(2)^3 = 6 x 2^3 = 48
    assert len(ms) == 48
    # every blocking factor divides its tile count
    for m in ms:
        assert w.MT % m.m_in == 0 and w.KT % m.k_in == 0 and w.NT % m.n_in == 0
        assert set(m.perm) == {"M", "K", "N"}


def test_footprint_and_feasible():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    # all-resident mapping: m_in=k_in=n_in=2
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    # foot(A)=k_in*n_in*1024*act = 2*2*1024*8 = 32768
    # foot(W)=m_in*k_in*1024*max_wbits = 2*2*1024*8 = 65536
    # foot(C)=m_in*n_in*1024*32 = 2*2*1024*32 = 131072
    assert s.footprint_bits(m, w) == 32768 + 65536 + 131072  # 229376
    big = s.HW(bank_size=1024, banks=32, dram_bw=64)   # cap=1048576
    tiny = s.HW(bank_size=1, banks=2, dram_bw=64)       # cap=64
    assert s.feasible(m, w, big) is True
    assert s.feasible(m, w, tiny) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: FAIL — `AttributeError: module 'mxp_scheduler' has no attribute 'Mapping'`.

- [ ] **Step 3: Write minimal implementation**

Append to `mxp_scheduler.py`:

```python
@dataclass(frozen=True)
class Mapping:
    perm: tuple             # permutation of ("M","K","N"), perm[0] = outermost
    m_in: int               # resident (inner) tile count per dim
    k_in: int
    n_in: int


def _out_in(m, w):
    """Return (out, inn) dicts of outer/inner factors per dimension."""
    inn = {"M": m.m_in, "K": m.k_in, "N": m.n_in}
    out = {"M": w.MT // m.m_in, "K": w.KT // m.k_in, "N": w.NT // m.n_in}
    return out, inn


def gen_mappings(w):
    ms = []
    for perm in itertools.permutations(("M", "K", "N")):
        for mi in divisors(w.MT):
            for ki in divisors(w.KT):
                for ni in divisors(w.NT):
                    ms.append(Mapping(perm=perm, m_in=mi, k_in=ki, n_in=ni))
    return ms


def footprint_bits(m, w):
    max_wbits = max(max(row) for row in w.wbits)   # conservative resident W storage (A7)
    foot_a = m.k_in * m.n_in * TILE * TILE * w.act_bits
    foot_w = m.m_in * m.k_in * TILE * TILE * max_wbits
    foot_c = m.m_in * m.n_in * TILE * TILE * 32
    return int(foot_a + foot_w + foot_c)


def feasible(m, w, hw):
    return footprint_bits(m, w) <= hw.cap_bits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/mxp_scheduler.py MXP_scheduler/test_mxp_scheduler.py
git commit -m "feat(scheduler): Mapping enumeration + footprint + feasibility"
```

---

## Task 3: DRAM traffic counts (reload factors + partial psum spill)

**Files:**
- Modify: `MXP_scheduler/mxp_scheduler.py`
- Test: `MXP_scheduler/test_mxp_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
def test_dram_bits_all_resident():
    # G1: 64^3, all-resident (K_out=1 -> no spill), uniform wbits=8, act=8
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    d = s.dram_bits(m, w)
    assert d["A"] == 64 * 64 * 8 * 1        # K*N*act * M_out(=1) = 32768
    assert d["W"] == 32768 * 1              # total_w_bits * N_out(=1)
    assert d["Cw"] == 64 * 64 * 32 * 1      # M*N*32 * K_out(=1) = 131072
    assert d["Cr"] == 0                     # K_out-1 = 0 (first touch zero-init)
    assert d["total"] == 32768 + 32768 + 131072 + 0  # 196608


def test_dram_bits_kspill():
    # G2: force K_out=2 (k_in=1) -> psum spill doubles C write, adds C read
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("K", "M", "N"), m_in=2, k_in=1, n_in=2)
    d = s.dram_bits(m, w)
    assert d["Cw"] == 64 * 64 * 32 * 2     # K_out=2 -> 262144
    assert d["Cr"] == 64 * 64 * 32 * 1     # (K_out-1)=1 -> 131072
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'dram_bits'`.

- [ ] **Step 3: Write minimal implementation**

Append to `mxp_scheduler.py`:

```python
def dram_bits(m, w):
    out, _ = _out_in(m, w)
    a = (w.K * w.N * w.act_bits) * out["M"]            # reload(A) = M_out
    wt = w.total_w_bits * out["N"]                      # reload(W) = N_out
    cw = (w.M * w.N * 32) * out["K"]                    # each outer-K writes partial
    cr = (w.M * w.N * 32) * (out["K"] - 1)              # reload for accumulate; first touch zero-init
    return {"A": int(a), "W": int(wt), "Cw": int(cw), "Cr": int(cr),
            "total": int(a + wt + cw + cr)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/mxp_scheduler.py MXP_scheduler/test_mxp_scheduler.py
git commit -m "feat(scheduler): DRAM traffic counts with reload factors + psum spill"
```

---

## Task 4: on-chip / MAC / RMW counts + compute_work

**Files:**
- Modify: `MXP_scheduler/mxp_scheduler.py`
- Test: `MXP_scheduler/test_mxp_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
def test_onchip_mac_rmw_and_compute_work():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    # MAC = M*K*N
    assert s.mac_ops(w) == 64 * 64 * 64                   # 262144
    # RMW = T * 1024 * disp; T=8, disp(act=8)=1
    assert s.rmw_ops(w) == 8 * 1024 * 1                   # 8192
    # compute_work = 32 * NT * sum(all wbits) = 32 * 2 * 32 = 2048
    assert s.compute_work(w) == 2048
    # onchip (G1 all-resident): A_rd=T*1024*act=8*1024*8=65536 ; W_rd=NT*total_w=2*32768=65536 ;
    # refill = DRAM(A)+DRAM(W)+DRAM(Cr) = 32768+32768+0 = 65536 ; total=196608
    assert s.onchip_bits(m, w) == 65536 + 65536 + 65536   # 196608


def test_rmw_disp_low_precision():
    w2 = s.Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 2]], act_bits=2)
    assert s.rmw_ops(w2) == 8 * 1024 * 4                   # disp(act=2)=4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'mac_ops'`.

- [ ] **Step 3: Write minimal implementation**

Append to `mxp_scheduler.py`:

```python
_DISP = {8: 1, 4: 2, 2: 4}   # RMW dispatches per col fire by activation mode


def mac_ops(w):
    return w.M * w.K * w.N


def rmw_ops(w):
    return w.MT * w.KT * w.NT * TILE * TILE * _DISP[w.act_bits]


def compute_work(w):
    # Σ_cube 32·b = 32 · NT · Σ_{m,k} wbits  (order-independent ideal lower bound)
    return int(TILE * w.NT * sum(sum(row) for row in w.wbits))


def onchip_bits(m, w):
    a_rd = (w.MT * w.KT * w.NT) * TILE * TILE * w.act_bits   # A read once per cube
    w_rd = w.NT * w.total_w_bits                              # W tile read once per n_t
    d = dram_bits(m, w)
    refill = d["A"] + d["W"] + d["Cr"]                        # DRAM -> on-chip loads (mapping-variable)
    return int(a_rd + w_rd + refill)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/mxp_scheduler.py MXP_scheduler/test_mxp_scheduler.py
git commit -m "feat(scheduler): on-chip/MAC/RMW counts + compute_work"
```

---

## Task 5: Energy combiner (counts · coeffs — separate final step)

**Files:**
- Modify: `MXP_scheduler/mxp_scheduler.py`
- Test: `MXP_scheduler/test_mxp_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
def test_energy_breakdown_g1():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)   # default coeffs 200/6/1/5
    e = s.energy_breakdown(m, w, hw)
    assert e["dram"] == 196608 * 200      # 39321600
    assert e["onchip"] == 196608 * 6      # 1179648
    assert e["mac"] == 262144 * 1         # 262144
    assert e["rmw"] == 8192 * 5           # 40960
    assert e["total"] == 39321600 + 1179648 + 262144 + 40960   # 40804352
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'energy_breakdown'`.

- [ ] **Step 3: Write minimal implementation**

Append to `mxp_scheduler.py`:

```python
def energy_breakdown(m, w, hw):
    c = hw.coeffs
    e_dram = dram_bits(m, w)["total"] * c["dram"]
    e_onchip = onchip_bits(m, w) * c["onchip"]
    e_mac = mac_ops(w) * c["mac"]
    e_rmw = rmw_ops(w) * c["rmw"]
    return {"dram": e_dram, "onchip": e_onchip, "mac": e_mac, "rmw": e_rmw,
            "total": e_dram + e_onchip + e_mac + e_rmw}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/mxp_scheduler.py MXP_scheduler/test_mxp_scheduler.py
git commit -m "feat(scheduler): energy = counts*coeffs (separate weighting step)"
```

---

## Task 6: Sequence-aware stall + actual cycle

**Files:**
- Modify: `MXP_scheduler/mxp_scheduler.py`
- Test: `MXP_scheduler/test_mxp_scheduler.py`

**Model (spec §8):** walk the outer-iteration sequence in `perm` order (perm[0] outermost). For each block: `compute_blk = 32·N_in·Σ wbits(resident M_in×K_in tiles)`. Fetch counts **inputs only** (A,W): A reloads when `(K_out_idx, N_out_idx)` changes, W reloads when `(M_out_idx, K_out_idx)` changes. `fill` = first block's input fetch time (unhidable). `stall_j = max(0, fetch_j/bw − compute_{j-1})` (double-buffer hides fetch behind previous block's compute).

- [ ] **Step 1: Write the failing test**

```python
def test_stall_fill_g3():
    # G3: low-bit blocks cannot hide an A-reload -> stall>0
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 2]], act_bits=8)
    # perm N outer, n_in=1 (N_out=2); m_in=2,k_in=2 (M_out=K_out=1); bw=32
    m = s.Mapping(perm=("N", "M", "K"), m_in=2, k_in=2, n_in=1)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    stall, fill = s.stall_fill(m, w, hw)
    # block compute = 32*N_in(1)*sum(2,2,2,2=8) = 256 each
    # j0 fetch = A_blk + W_blk = (2*1*1024*8=16384) + (8*1024=8192) = 24576 ; fill=24576/32=768
    # j1: N changed -> A reload 16384 ; W same -> 0 ; stall=max(0,16384/32 - 256)=max(0,512-256)=256
    assert fill == 768.0
    assert stall == 256.0
    # actual = compute_work + fill + stall = 512 + 768 + 256 = 1536
    assert s.actual_cycle(m, w, hw) == 1536.0


def test_stall_zero_when_bw_huge():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("M", "K", "N"), m_in=2, k_in=2, n_in=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)   # effectively infinite BW
    stall, fill = s.stall_fill(m, w, hw)
    assert stall == 0.0
    assert fill < 1.0   # fetch/huge-bw ~ 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'stall_fill'`.

- [ ] **Step 3: Write minimal implementation**

Append to `mxp_scheduler.py`:

```python
def stall_fill(m, w, hw):
    """Sequence-aware stall over outer-iteration blocks. Inputs (A,W) only (spec §8)."""
    out, inn = _out_in(m, w)
    bw = hw.dram_bw
    a_blk = inn["K"] * inn["N"] * TILE * TILE * w.act_bits
    # iterate outer index tuples in perm order (perm[0] outermost)
    ranges = [range(out[d]) for d in m.perm]
    prev = None
    prev_compute = 0.0
    fill = 0.0
    total_stall = 0.0
    for combo in itertools.product(*ranges):
        idx = dict(zip(m.perm, combo))                     # outer indices per dim
        mo, ko = idx["M"], idx["K"]
        # resident W tiles for this block, summed avg bits
        w_sum = sum(w.wbits[mt][kt]
                    for mt in range(mo * inn["M"], (mo + 1) * inn["M"])
                    for kt in range(ko * inn["K"], (ko + 1) * inn["K"]))
        w_blk = w_sum * TILE * TILE
        compute_blk = TILE * inn["N"] * w_sum              # 32 * N_in * Σ wbits(block)
        if prev is None:
            fill = (a_blk + w_blk) / bw
        else:
            fetch = 0
            if (idx["K"], idx["N"]) != (prev["K"], prev["N"]):
                fetch += a_blk                              # A reloads (A indexed K,N)
            if (idx["M"], idx["K"]) != (prev["M"], prev["K"]):
                fetch += w_blk                              # W reloads (W indexed M,K)
            total_stall += max(0.0, fetch / bw - prev_compute)
        prev = idx
        prev_compute = compute_blk
    return total_stall, fill


def actual_cycle(m, w, hw):
    stall, fill = stall_fill(m, w, hw)
    return float(compute_work(w)) + fill + stall
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: PASS (13 passed).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/mxp_scheduler.py MXP_scheduler/test_mxp_scheduler.py
git commit -m "feat(scheduler): sequence-aware stall + actual cycle (not roofline)"
```

---

## Task 7: evaluate() + optimize() (exhaustive, feasible, constraint, Pareto)

**Files:**
- Modify: `MXP_scheduler/mxp_scheduler.py`
- Test: `MXP_scheduler/test_mxp_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
def test_evaluate_keys():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)
    r = s.evaluate(m, w, hw)
    assert r["feasible"] is True
    assert r["energy"] == 40804352.0
    assert r["actual_cycle"] == s.actual_cycle(m, w, hw)
    assert r["mapping"] is m


def test_optimize_picks_min_energy_feasible():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)
    ranked = s.optimize(w, hw)
    assert all(r["feasible"] for r in ranked)
    # sorted ascending by energy
    assert ranked == sorted(ranked, key=lambda r: r["energy"])
    # min-energy mapping must be all-resident (no spill, no input reload): out factors all 1
    best = ranked[0]["mapping"]
    assert (best.m_in, best.k_in, best.n_in) == (w.MT, w.KT, w.NT)


def test_optimize_cycle_constraint_filters():
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 2]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    tight = min(s.actual_cycle(m, w, hw)
                for m in s.gen_mappings(w) if s.feasible(m, w, hw))
    ranked = s.optimize(w, hw, max_cycle=tight)
    assert len(ranked) >= 1
    assert all(r["actual_cycle"] <= tight + 1e-9 for r in ranked)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'evaluate'`.

- [ ] **Step 3: Write minimal implementation**

Append to `mxp_scheduler.py`:

```python
def evaluate(m, w, hw):
    feas = feasible(m, w, hw)
    eb = energy_breakdown(m, w, hw)
    stall, fill = stall_fill(m, w, hw)
    return {
        "mapping": m,
        "feasible": feas,
        "energy": eb["total"],
        "energy_breakdown": eb,
        "dram": dram_bits(m, w),
        "compute_work": compute_work(w),
        "stall": stall,
        "fill": fill,
        "actual_cycle": float(compute_work(w)) + fill + stall,
    }


def optimize(w, hw, max_cycle=None):
    """Exhaustive over (perm x blocking). Return feasible results sorted by energy asc.
    If max_cycle given, keep only results with actual_cycle <= max_cycle."""
    results = [evaluate(m, w, hw) for m in gen_mappings(w)]
    results = [r for r in results if r["feasible"]]
    if max_cycle is not None:
        results = [r for r in results if r["actual_cycle"] <= max_cycle + 1e-9]
    results.sort(key=lambda r: r["energy"])
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: PASS (16 passed).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/mxp_scheduler.py MXP_scheduler/test_mxp_scheduler.py
git commit -m "feat(scheduler): evaluate + exhaustive optimize with cycle constraint"
```

---

## Task 8: LPT-greedy headroom baseline

**Files:**
- Modify: `MXP_scheduler/mxp_scheduler.py`
- Test: `MXP_scheduler/test_mxp_scheduler.py`

**Note (spec §9.2):** LPT is a *headroom* indicator only (independent-jobs/makespan model; our tiles are reuse-coupled), not a drop-in. It reorders the outer blocks of a given mapping by descending block-compute and recomputes stall under the same input-reload diff logic.

- [ ] **Step 1: Write the failing test**

```python
def test_lpt_headroom_runs_and_reports_both():
    w = s.Work(M=128, K=128, N=128, wbits=[[2, 8, 2, 8]] * 4, act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    best = s.optimize(w, hw)[0]["mapping"]
    h = s.lpt_headroom(best, w, hw)
    assert set(h) == {"natural_stall", "lpt_stall", "headroom"}
    assert h["headroom"] == h["natural_stall"] - h["lpt_stall"]
    assert h["natural_stall"] >= 0 and h["lpt_stall"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'lpt_headroom'`.

- [ ] **Step 3: Write minimal implementation**

Append to `mxp_scheduler.py`:

```python
def _blocks(m, w):
    """Yield (idx_dict, compute_blk, a_blk, w_blk) per outer block in perm order."""
    out, inn = _out_in(m, w)
    a_blk = inn["K"] * inn["N"] * TILE * TILE * w.act_bits
    ranges = [range(out[d]) for d in m.perm]
    for combo in itertools.product(*ranges):
        idx = dict(zip(m.perm, combo))
        mo, ko = idx["M"], idx["K"]
        w_sum = sum(w.wbits[mt][kt]
                    for mt in range(mo * inn["M"], (mo + 1) * inn["M"])
                    for kt in range(ko * inn["K"], (ko + 1) * inn["K"]))
        yield idx, TILE * inn["N"] * w_sum, a_blk, w_sum * TILE * TILE


def _stall_of_order(blocks, bw):
    prev = None
    prev_compute = 0.0
    total = 0.0
    for idx, compute_blk, a_blk, w_blk in blocks:
        if prev is not None:
            fetch = 0
            if (idx["K"], idx["N"]) != (prev["K"], prev["N"]):
                fetch += a_blk
            if (idx["M"], idx["K"]) != (prev["M"], prev["K"]):
                fetch += w_blk
            total += max(0.0, fetch / bw - prev_compute)
        prev = idx
        prev_compute = compute_blk
    return total


def lpt_headroom(m, w, hw):
    blocks = list(_blocks(m, w))
    natural = _stall_of_order(blocks, hw.dram_bw)
    lpt = _stall_of_order(sorted(blocks, key=lambda b: -b[1]), hw.dram_bw)
    return {"natural_stall": natural, "lpt_stall": lpt, "headroom": natural - lpt}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: PASS (17 passed).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/mxp_scheduler.py MXP_scheduler/test_mxp_scheduler.py
git commit -m "feat(scheduler): LPT-greedy headroom baseline"
```

---

## Task 9: report + CLI + --selftest

**Files:**
- Modify: `MXP_scheduler/mxp_scheduler.py`
- Test: `MXP_scheduler/test_mxp_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
def test_selftest_passes():
    s.selftest()   # raises AssertionError on any golden mismatch; returns None on success


def test_report_contains_mapping_and_energy():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)
    ranked = s.optimize(w, hw)
    txt = s.report(ranked[:3], w, hw)
    assert "perm" in txt and "energy" in txt and "actual_cycle" in txt


def test_cli_selftest_exit0():
    import subprocess, sys, pathlib
    here = pathlib.Path(s.__file__).parent
    r = subprocess.run([sys.executable, "mxp_scheduler.py", "--selftest"],
                       cwd=here, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'selftest'`.

- [ ] **Step 3: Write minimal implementation**

Append to `mxp_scheduler.py`:

```python
def report(ranked, w, hw, top=10):
    lines = [f"# GEMM ({w.M}x{w.K}x{w.N})  cap_bits={hw.cap_bits}  dram_bw={hw.dram_bw}",
             f"{'perm':<10}{'m_in':>5}{'k_in':>5}{'n_in':>5}{'energy':>16}{'actual_cycle':>16}{'stall':>12}"]
    for r in ranked[:top]:
        m = r["mapping"]
        lines.append(f"{''.join(m.perm):<10}{m.m_in:>5}{m.k_in:>5}{m.n_in:>5}"
                     f"{r['energy']:>16.0f}{r['actual_cycle']:>16.1f}{r['stall']:>12.1f}")
    return "\n".join(lines)


def selftest():
    # G1: DRAM + energy + compute_work
    w1 = Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m1 = Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    hw1 = HW(bank_size=1024, banks=32, dram_bw=64)
    assert dram_bits(m1, w1)["total"] == 196608
    assert compute_work(w1) == 2048
    assert energy_breakdown(m1, w1, hw1)["total"] == 40804352.0
    # G2: psum spill
    m2 = Mapping(perm=("K", "M", "N"), m_in=2, k_in=1, n_in=2)
    assert dram_bits(m2, w1)["Cw"] == 262144 and dram_bits(m2, w1)["Cr"] == 131072
    # G3: sequence stall
    w3 = Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 2]], act_bits=8)
    m3 = Mapping(perm=("N", "M", "K"), m_in=2, k_in=2, n_in=1)
    hw3 = HW(bank_size=1024, banks=32, dram_bw=32)
    assert stall_fill(m3, w3, hw3) == (256.0, 768.0)
    assert actual_cycle(m3, w3, hw3) == 1536.0
    print("selftest: OK")


def _load_wbits(path, MT, KT):
    import json
    with open(path) as f:
        wb = json.load(f)
    if len(wb) != MT or any(len(r) != KT for r in wb):
        raise ValueError(f"wbits file must be {MT}x{KT}")
    return wb


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="MXP_scheduler — GEMM mapping cost-model + optimizer")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--crosscheck", action="store_true")
    p.add_argument("--M", type=int); p.add_argument("--K", type=int); p.add_argument("--N", type=int)
    p.add_argument("--bank-size", type=int, default=1024)
    p.add_argument("--banks", type=int, default=32)
    p.add_argument("--dram-bw", type=float, default=64.0)
    p.add_argument("--act", type=int, default=8)
    p.add_argument("--bits-file", help="JSON MT x KT avg-weight-bits; default all=act")
    p.add_argument("--max-cycle", type=float, default=None)
    p.add_argument("--coeffs", help="JSON override of energy coeffs")
    a = p.parse_args(argv)
    if a.selftest:
        selftest(); return 0
    if a.crosscheck:
        crosscheck(); return 0
    if not (a.M and a.K and a.N):
        p.error("provide --M --K --N (or --selftest/--crosscheck)")
    MT, KT = -(-a.M // TILE), -(-a.K // TILE)
    wbits = _load_wbits(a.bits_file, MT, KT) if a.bits_file else [[a.act] * KT for _ in range(MT)]
    coeffs = dict(DEFAULT_COEFFS)
    if a.coeffs:
        import json
        coeffs.update(json.load(open(a.coeffs)))
    w = Work(M=a.M, K=a.K, N=a.N, wbits=wbits, act_bits=a.act)
    hw = HW(bank_size=a.bank_size, banks=a.banks, dram_bw=a.dram_bw, coeffs=coeffs)
    ranked = optimize(w, hw, max_cycle=a.max_cycle)
    print(report(ranked, w, hw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: `crosscheck()` is defined in Task 10; until then `--crosscheck` will NameError (not exercised by Task 9 tests).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: PASS (20 passed).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/mxp_scheduler.py MXP_scheduler/test_mxp_scheduler.py
git commit -m "feat(scheduler): report + CLI + embedded --selftest"
```

---

## Task 10: Annotated twin (`mxp_scheduler_annotated.py`) + --explain + --crosscheck

**Files:**
- Create: `MXP_scheduler/mxp_scheduler_annotated.py`
- Modify: `MXP_scheduler/mxp_scheduler.py` (add `crosscheck()`)
- Test: `MXP_scheduler/test_mxp_scheduler.py`

**Approach:** `mxp_scheduler_annotated.py` re-implements the SAME logic as `mxp_scheduler.py` (identical function/class names and formulas) but with line-by-line 한국어 comments explaining each quantity (reload factors, sequence stall, etc.) and an extra `--explain` mode that prints the step-by-step intermediate values for a single mapping. It must NOT import the standard file (so it reads top-to-bottom standalone). `crosscheck()` (in the standard file) imports both modules and asserts identical `evaluate()` output on a sweep — this is the equivalence guarantee, so the annotated file does not need its own golden numbers reproduced in this plan.

- [ ] **Step 1: Write the failing test**

```python
def test_crosscheck_passes():
    s.crosscheck()   # raises AssertionError if the two files disagree on any swept mapping


def test_annotated_imports_and_matches_one_case():
    import importlib, pathlib, sys
    sys.path.insert(0, str(pathlib.Path(s.__file__).parent))
    ann = importlib.import_module("mxp_scheduler_annotated")
    w_args = dict(M=64, K=64, N=64, wbits=[[2, 8], [8, 2]], act_bits=8)
    hw_args = dict(bank_size=1024, banks=32, dram_bw=32)
    a_w, b_w = s.Work(**w_args), ann.Work(**w_args)
    a_hw, b_hw = s.HW(**hw_args), ann.HW(**hw_args)
    m_a = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=1)
    m_b = ann.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=1)
    ea, eb = s.evaluate(m_a, a_w, a_hw), ann.evaluate(m_b, b_w, b_hw)
    assert ea["energy"] == eb["energy"]
    assert ea["actual_cycle"] == eb["actual_cycle"]
    assert ea["dram"] == eb["dram"]


def test_annotated_explain_cli_exit0():
    import subprocess, sys, pathlib
    here = pathlib.Path(s.__file__).parent
    r = subprocess.run([sys.executable, "mxp_scheduler_annotated.py", "--explain",
                        "--M", "64", "--K", "64", "--N", "64", "--dram-bw", "32"],
                       cwd=here, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "compute_work" in r.stdout and "stall" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'crosscheck'` and `ModuleNotFoundError: mxp_scheduler_annotated`.

- [ ] **Step 3a: Add `crosscheck()` to `mxp_scheduler.py`**

Append (above `def main`):

```python
def crosscheck(verbose=False):
    """Assert the annotated twin produces identical evaluate() output on a sweep."""
    import importlib, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ann = importlib.import_module("mxp_scheduler_annotated")
    cases = [
        (64, 64, 64, [[2, 8], [8, 2]], 8, 1024, 32, 32.0),
        (128, 64, 96, [[4, 2, 8], [8, 8, 2], [2, 4, 4], [8, 2, 8]], 4, 512, 16, 48.0),
    ]
    for (M, K, N, wb, act, bs, bk, bw) in cases:
        w_s, w_a = Work(M, K, N, wb, act), ann.Work(M, K, N, wb, act)
        hw_s, hw_a = HW(bs, bk, bw), ann.HW(bs, bk, bw)
        for m_s in gen_mappings(w_s):
            m_a = ann.Mapping(perm=m_s.perm, m_in=m_s.m_in, k_in=m_s.k_in, n_in=m_s.n_in)
            r_s, r_a = evaluate(m_s, w_s, hw_s), ann.evaluate(m_a, w_a, hw_a)
            for key in ("feasible", "energy", "actual_cycle", "stall", "fill"):
                assert r_s[key] == r_a[key], f"mismatch {key}: {r_s[key]} != {r_a[key]} @ {m_s}"
            assert r_s["dram"] == r_a["dram"]
    print("crosscheck: OK")
```

- [ ] **Step 3b: Create `mxp_scheduler_annotated.py`**

Mirror `mxp_scheduler.py` exactly (same names/formulas), adding 한국어 line comments and an `--explain` mode. The numeric logic must be character-for-character equivalent in behavior (crosscheck enforces this). Skeleton to fill — copy each function body from `mxp_scheduler.py`, prepend explanatory comments, and add `explain()`:

```python
"""MXP_scheduler (주석 상세본) — 표준본과 동일 결과. 위에서 아래로 읽으며 cost-model 을 이해하는 용도.
표준본(mxp_scheduler.py)과 동일 로직. --crosscheck 로 동치성 보장.
"""
from dataclasses import dataclass, field
import itertools

TILE = 32                                  # 물리 systolic array = spatial mapping (HW 고정, 탐색 안 함)
DEFAULT_COEFFS = {"dram": 200.0, "onchip": 6.0, "mac": 1.0, "rmw": 5.0}  # 정규화 상대 energy 계수
_DISP = {8: 1, 4: 2, 2: 4}                 # activation mode 별 col-fire 당 RMW dispatch 수

# --- [config], [mapping], [evaluate], [timing], [energy], [optimize] 를 표준본에서 그대로 옮기되
# --- 각 줄에 "왜 이렇게 세는가" 주석을 단다. (예: reload(W)=N_out 은 W 가 N 에 무관해 N-tile 마다 재적재)
# ... (mirror divisors / HW / Work / Mapping / _out_in / gen_mappings / footprint_bits / feasible /
#      dram_bits / mac_ops / rmw_ops / compute_work / onchip_bits / energy_breakdown /
#      stall_fill / actual_cycle / _blocks / _stall_of_order / lpt_headroom / evaluate / optimize /
#      report  — 본문은 표준본과 동일, 주석만 풍부)


def explain(m, w, hw):
    """한 mapping 에 대해 단계별 중간값을 한 줄씩 출력 — 따라가며 이해."""
    out, inn = _out_in(m, w)
    print(f"[입력] GEMM {w.M}x{w.K}x{w.N}  타일 {w.MT}x{w.KT}x{w.NT}  act={w.act_bits}b  bw={hw.dram_bw}")
    print(f"[mapping] perm={''.join(m.perm)} (perm[0]=최외곽)  inner={inn}  outer={out}")
    print(f"[capacity] footprint={footprint_bits(m, w)}b  cap={hw.cap_bits}b  feasible={feasible(m, w, hw)}")
    d = dram_bits(m, w)
    print(f"[DRAM] reload A=M_out={out['M']} W=N_out={out['N']} C=K_out={out['K']}")
    print(f"[DRAM] A={d['A']} W={d['W']} Cw={d['Cw']} Cr={d['Cr']} total={d['total']}b")
    print(f"[compute] work(=ideal 하한, order 무관)=32*NT*Σwbits={compute_work(w)} cyc")
    stall, fill = stall_fill(m, w, hw)
    print(f"[stall] 시퀀스별 stall_i=max(0, fetch/bw - 직전 block compute) 누적")
    print(f"[stall] fill={fill:.1f}  total_stall={stall:.1f}  actual_cycle={actual_cycle(m, w, hw):.1f}")
    eb = energy_breakdown(m, w, hw)
    print(f"[energy] dram={eb['dram']:.0f} onchip={eb['onchip']:.0f} mac={eb['mac']:.0f} "
          f"rmw={eb['rmw']:.0f} total={eb['total']:.0f}")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="MXP_scheduler 주석 상세본")
    p.add_argument("--explain", action="store_true")
    p.add_argument("--M", type=int); p.add_argument("--K", type=int); p.add_argument("--N", type=int)
    p.add_argument("--bank-size", type=int, default=1024)
    p.add_argument("--banks", type=int, default=32)
    p.add_argument("--dram-bw", type=float, default=64.0)
    p.add_argument("--act", type=int, default=8)
    p.add_argument("--bits-file")
    a = p.parse_args(argv)
    if not (a.M and a.K and a.N):
        p.error("provide --M --K --N")
    MT, KT = -(-a.M // TILE), -(-a.K // TILE)
    if a.bits_file:
        import json
        wbits = json.load(open(a.bits_file))
    else:
        wbits = [[a.act] * KT for _ in range(MT)]
    w = Work(M=a.M, K=a.K, N=a.N, wbits=wbits, act_bits=a.act)
    hw = HW(bank_size=a.bank_size, banks=a.banks, dram_bw=a.dram_bw)
    ranked = optimize(w, hw)
    best = ranked[0]["mapping"] if ranked else gen_mappings(w)[0]
    if a.explain:
        explain(best, w, hw)
    else:
        print(report(ranked, w, hw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Implementer instruction:** fill the `...` by copying every function listed from `mxp_scheduler.py` verbatim (logic identical), adding 한국어 comments only. Do not change any formula. Then run `--crosscheck` to prove equivalence.

- [ ] **Step 4: Run tests + crosscheck to verify they pass**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q && python mxp_scheduler.py --crosscheck`
Expected: PASS (23 passed) and `crosscheck: OK`.

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/mxp_scheduler.py MXP_scheduler/mxp_scheduler_annotated.py MXP_scheduler/test_mxp_scheduler.py
git commit -m "feat(scheduler): annotated twin + --explain + --crosscheck equivalence guard"
```

---

## Task 11: Pareto + constraint ON/OFF tradeoff report (spec step 4)

**Files:**
- Modify: `MXP_scheduler/mxp_scheduler.py`
- Test: `MXP_scheduler/test_mxp_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
def test_pareto_front_is_nondominated():
    w = s.Work(M=128, K=128, N=128, wbits=[[2, 8, 2, 8]] * 4, act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    ranked = s.optimize(w, hw)
    front = s.pareto_front(ranked)
    # every point on the front is non-dominated (no other point lower in BOTH energy and cycle)
    for p in front:
        for q in ranked:
            assert not (q["energy"] < p["energy"] and q["actual_cycle"] < p["actual_cycle"])
    # sorted by energy ascending on the front
    assert front == sorted(front, key=lambda r: r["energy"])


def test_tradeoff_on_off():
    w = s.Work(M=128, K=128, N=128, wbits=[[2, 8, 2, 8]] * 4, act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    t = s.tradeoff(w, hw)
    assert "off" in t and "on" in t            # off = global min energy; on = min energy at min cycle
    assert t["off"]["energy"] <= t["on"]["energy"] + 1e-9   # constraint can only raise energy
    assert t["on"]["actual_cycle"] <= t["off"]["actual_cycle"] + 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'pareto_front'`.

- [ ] **Step 3: Write minimal implementation**

Append to `mxp_scheduler.py` (before `crosscheck`):

```python
def pareto_front(ranked):
    """Non-dominated set on (energy, actual_cycle), sorted by energy asc."""
    pts = sorted(ranked, key=lambda r: (r["energy"], r["actual_cycle"]))
    front = []
    best_cycle = float("inf")
    for r in pts:
        if r["actual_cycle"] < best_cycle - 1e-9:
            front.append(r)
            best_cycle = r["actual_cycle"]
    return front


def tradeoff(w, hw):
    """OFF = global min-energy mapping. ON = min-energy among min-actual-cycle mappings."""
    ranked = optimize(w, hw)
    off = ranked[0]
    min_cycle = min(r["actual_cycle"] for r in ranked)
    on = min((r for r in ranked if r["actual_cycle"] <= min_cycle + 1e-9),
             key=lambda r: r["energy"])
    return {"off": off, "on": on, "pareto": pareto_front(ranked)}
```

Also add to `selftest()` (before `print("selftest: OK")`):

```python
    # Pareto/tradeoff sanity
    wt = Work(M=128, K=128, N=128, wbits=[[2, 8, 2, 8]] * 4, act_bits=8)
    hwt = HW(bank_size=1024, banks=32, dram_bw=32)
    t = tradeoff(wt, hwt)
    assert t["off"]["energy"] <= t["on"]["energy"] + 1e-9
    assert t["on"]["actual_cycle"] <= t["off"]["actual_cycle"] + 1e-9
```

And mirror `pareto_front` + `tradeoff` into `mxp_scheduler_annotated.py` (with comments), then re-run `--crosscheck`.

- [ ] **Step 4: Run tests + crosscheck**

Run: `cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q && python mxp_scheduler.py --crosscheck && python mxp_scheduler.py --selftest`
Expected: PASS (25 passed), `crosscheck: OK`, `selftest: OK`.

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/
git commit -m "feat(scheduler): Pareto front + constraint ON/OFF tradeoff"
```

---

## Task 12: README + run on the project workload (128³)

**Files:**
- Create: `MXP_scheduler/README.md`
- Test: manual run (documented below)

- [ ] **Step 1: Write README**

```markdown
# MXP_scheduler
GEMM mapping cost-model + optimizer for the MXP bit-serial mixed-precision SA.
Spec: ../docs/superpowers/specs/2026-06-04-mxp-scheduler-design.md

## Two files, identical results
- `mxp_scheduler.py`            — standard (clean) build
- `mxp_scheduler_annotated.py`  — 주석 상세본 + `--explain` (단계별 따라가기)

## Run
    python mxp_scheduler.py --M 128 --K 128 --N 128 --bank-size 1024 --banks 32 --dram-bw 64 --act 8
    python mxp_scheduler.py --bits-file wbits.json --M 128 --K 128 --N 128 ...   # per-tile avg bits
    python mxp_scheduler.py --selftest        # embedded golden checks
    python mxp_scheduler.py --crosscheck      # standard ↔ annotated equivalence
    python mxp_scheduler_annotated.py --explain --M 64 --K 64 --N 64 --dram-bw 32

## wbits.json format
MT x KT array (MT=ceil(M/32), KT=ceil(K/32)) of average weight bits per 32x32 tile, each in [2,8].

## Dev tests
    python -m pytest test_mxp_scheduler.py -q
```

- [ ] **Step 2: Run the project workload (uniform 8-bit, current HW)**

Run:
```bash
cd MXP_scheduler && python mxp_scheduler.py --M 128 --K 128 --N 128 \
    --bank-size 1024 --banks 32 --dram-bw 64 --act 8
```
Expected: a ranked table printed (top mapping has out factors all 1 = all-resident when capacity allows), no error.

- [ ] **Step 3: Run --explain on a small mixed case**

Run:
```bash
cd MXP_scheduler && printf '[[2,8],[8,2]]' > /tmp/wb.json && \
    python mxp_scheduler_annotated.py --explain --M 64 --K 64 --N 64 \
    --dram-bw 32 --bits-file /tmp/wb.json
```
Expected: step-by-step `[입력]/[mapping]/[capacity]/[DRAM]/[compute]/[stall]/[energy]` lines.

- [ ] **Step 4: Final verification — all gates green**

Run:
```bash
cd MXP_scheduler && python -m pytest test_mxp_scheduler.py -q \
    && python mxp_scheduler.py --selftest && python mxp_scheduler.py --crosscheck
```
Expected: all pytest pass, `selftest: OK`, `crosscheck: OK`.

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/README.md
git commit -m "docs(scheduler): README + project-workload run instructions"
```

---

## Self-Review (run by plan author)

**1. Spec coverage:**
- §3 I/O contract → Tasks 1 (Work/HW), 9 (CLI flags incl. dram-bw, bits-file, coeffs, max-cycle). ✓
- §5 nested-blocked mapping (perm × blocking, divisors) → Task 2. ✓
- §6.1 capacity/footprint → Task 2. ✓
- §6.2 DRAM reload factors + psum spill → Task 3. ✓
- §6.3 on-chip (variable refill) / MAC / RMW → Task 4. ✓
- §6.4 compute_work (ideal, order-independent) → Task 4. ✓
- §7 energy = counts·coeffs (separate) → Task 5. ✓
- §8 sequence-aware stall (inputs only, not roofline) + actual cycle → Task 6. ✓
- §8 constraint ON/OFF + Pareto → Task 11. ✓
- §9 exhaustive optimize, scoped → Task 7; §9.2 LPT headroom → Task 8. ✓
- §10 validation: `--selftest` (Task 9) + golden pytest throughout + crosscheck (Task 10). TB calibration (§10/C6) is explicitly out of v1 code (requires running RTL); left as a follow-up, not a code task. ✓ (gap noted, intentional)
- §12 two files + --explain + --crosscheck → Task 10. ✓

**2. Placeholder scan:** Task 10 Step 3b contains `...` for the mirrored bodies — this is intentional (copy verbatim from the standard file; the bodies are fully defined in Tasks 1–9 and 11) with an explicit implementer instruction and the `--crosscheck` gate enforcing equivalence. No other placeholders.

**3. Type consistency:** Function/class names locked in the "Conventions" block and used identically across Tasks 1–12. `Mapping` is `frozen=True` (hashable; fine — never mutated). `evaluate()` dict keys (`mapping, feasible, energy, energy_breakdown, dram, compute_work, stall, fill, actual_cycle`) defined once in Task 7 and consumed by Tasks 8/11/report consistently. `stall_fill` returns `(stall, fill)` everywhere.

**Note on assumptions still open (do not block implementation — runtime params/defaults):** A3 energy coeff values (default 200/6/1/5 used; `--coeffs` overrides), A4 DRAM BW (CLI `--dram-bw`), A7 fill/drain simplification (fill = first-block input fetch). These match the spec's flagged-open assumptions.
