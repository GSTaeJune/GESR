# MXP_scheduler/mxp_scheduler.py
"""MXP_scheduler — GEMM mapping cost-model + optimizer for the MXP bit-serial SA.
Spec: docs/superpowers/specs/2026-06-04-mxp-scheduler-design.md
Standard (clean) build. stdlib only.
"""
from dataclasses import dataclass, field
import itertools

TILE = 32                # physical systolic-array dim = spatial mapping (HW-fixed)
FP32_BITS = 32           # FP32 psum word width (DRAM C traffic + on-chip C storage)
DEFAULT_COEFFS = {"dram": 200.0, "onchip": 6.0, "mac": 1.0, "rmw": 5.0}


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


@dataclass
class HW:
    bank_size: int          # words per bank
    banks: int
    dram_bw: float          # bits per DRAM cycle (bus throughput)
    word_bits: int = 32     # FP32 psum word
    freq_ratio: float = 1.0  # on-chip cycles per DRAM cycle (= f_chip / f_dram)
    coeffs: dict = field(default_factory=lambda: dict(DEFAULT_COEFFS))

    def __post_init__(self):
        for name, v in (("bank_size", self.bank_size), ("banks", self.banks),
                        ("word_bits", self.word_bits)):
            if v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")
        # dram_bw divides every fetch in stall_fill/_stall_of_order; 0 would crash and a
        # negative value would yield impossibly low (negative) cycles. Require positive.
        if self.dram_bw <= 0:
            raise ValueError(f"dram_bw must be positive (bits/cycle), got {self.dram_bw}")
        # freq_ratio converts DRAM transfer time into on-chip cycles (the cycle unit of
        # compute_work). 0/negative would invert or zero the conversion. Require positive.
        if self.freq_ratio <= 0:
            raise ValueError(f"freq_ratio must be positive (f_chip/f_dram), got {self.freq_ratio}")
        # coeffs feed the optimizer's objective: a negative coeff would let a HIGH-traffic mapping
        # win (negative energy), and a missing key would crash deep inside energy_breakdown. Require
        # all four keys present and each a non-negative number.
        for k in ("dram", "onchip", "mac", "rmw"):
            if k not in self.coeffs:
                raise ValueError(f"coeffs missing required key {k!r}")
            if not isinstance(self.coeffs[k], (int, float)) or self.coeffs[k] < 0:
                raise ValueError(f"coeffs[{k!r}] must be a non-negative number, got {self.coeffs[k]!r}")

    @property
    def eff_bw(self):
        # effective DRAM bandwidth in ON-CHIP cycle terms (bits per on-chip cycle):
        # bits/DRAM-cycle * (DRAM-cycles per on-chip cycle) = dram_bw / freq_ratio.
        # freq_ratio=1 (same clock) -> eff_bw == dram_bw (back-compatible).
        return self.dram_bw / self.freq_ratio

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
        # M/K/N must be positive multiples of TILE: wbits is a per-32x32-tile map, so a
        # partial last tile is not representable, and this keeps every count on ONE basis
        # (raw M*K*N == tiled MT*TILE*...), removing the raw-vs-tiled mismatch.
        for name, v in (("M", self.M), ("K", self.K), ("N", self.N)):
            if v <= 0 or v % TILE != 0:
                raise ValueError(f"{name} must be a positive multiple of TILE={TILE}, got {v}")
        if self.act_bits not in (2, 4, 8):
            raise ValueError(f"act_bits must be 2/4/8, got {self.act_bits}")
        if len(self.wbits) != self.MT or any(len(r) != self.KT for r in self.wbits):
            raise ValueError(f"wbits must be {self.MT}x{self.KT}")
        # per-tile AVERAGE weight bits — may be fractional, must lie in [2, 8]
        if any(not (2 <= b <= 8) for row in self.wbits for b in row):
            raise ValueError("each wbits entry (average weight bits) must be in [2, 8]")

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


@dataclass(frozen=True)
class Mapping:
    perm: tuple             # permutation of ("M","K","N"), perm[0] = outermost
    m_in: int               # resident (inner) tile count per dim
    k_in: int
    n_in: int

    def __post_init__(self):
        # _blocks/_stall_of_order index by idx["M"/"K"/"N"]; a non-permutation perm would
        # raise a late KeyError or silently mis-walk. Validate at construction.
        if tuple(sorted(self.perm)) != ("K", "M", "N"):
            raise ValueError(f"perm must be a permutation of ('M','K','N'), got {self.perm}")


def _out_in(m, w):
    """Return (out, inn) dicts of outer/inner factors per dimension.

    Blocking factors must divide their tile counts exactly. gen_mappings only emits
    such factors; this guard rejects a hand-built Mapping that would otherwise
    floor-divide to a wrong or zero outer count and silently corrupt the cost
    (e.g. out["K"]==0 -> negative Cr -> total<0 winning the optimizer)."""
    for dim, T, f in (("M", w.MT, m.m_in), ("K", w.KT, m.k_in), ("N", w.NT, m.n_in)):
        if f < 1 or T % f != 0:
            raise ValueError(f"{dim} blocking factor {f} must be a divisor of tile count {T}")
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
    # Resident on-chip storage that must hold simultaneously (spec §6.1). foot_a and foot_c are
    # the (mapping-constant) resident A and C windows; foot_w is the RESIDENT-WINDOW W bits,
    # maxed over the outer blocks (NOT a global-max over all tiles, which would over-count and
    # over-prune low-bit-resident mixed-precision mappings). Walking _blocks also validates the
    # mapping's blocking factors (via _out_in) so feasible() rejects invalid mappings consistently.
    blocks = list(_blocks(m, w))
    _, _, a_blk, _w_blk0, c_blk = blocks[0]
    foot_w = max(b[3] for b in blocks)   # b[3] = w_blk = resident M_in x K_in window bits
    return a_blk + foot_w + c_blk


def feasible(m, w, hw):
    return footprint_bits(m, w) <= hw.cap_bits


def dram_bits(m, w):
    """Order-DEPENDENT DRAM traffic (bits). Counts the loads/stores that actually happen
    in this loop order, on the SAME outer-block walk as the stall model — so the A/W DRAM
    volume equals stall_fill's total input fetch by construction (no two-model mismatch).

    Supersedes the spec §6.2 order-independent reload factors (reload A=M_out, W=N_out,
    C=K_out), which only hold for the worst-case order (reused dim outermost). Decision
    2026-06-05: energy must reflect the actual loop order so the optimizer can choose it.

      A (=activation[K,N], reused over M): (re)load when the (K,N) outer index changes
        (first block always loads). M inner to both K,N -> A stays resident -> fewer loads.
      W (=weight[M,K], reused over N): (re)load when the (M,K) outer index changes.
      C (=output[M,N], reduced over K): output-stationary when K is the INNERMOST loop
        -> the psum accumulates fully resident, one final write, no reload. Otherwise the
        C tile is evicted and re-touched once per outer-K -> spill (K_out writes, K_out-1
        reads; first touch zero-init)."""
    out, _ = _out_in(m, w)
    a = w_dram = 0.0
    prev = None
    for idx, _compute, a_blk, w_blk, _c_blk in _blocks(m, w):
        if prev is None or (idx["K"], idx["N"]) != (prev["K"], prev["N"]):
            a += a_blk                                  # A reload (A indexed K,N)
        if prev is None or (idx["M"], idx["K"]) != (prev["M"], prev["K"]):
            w_dram += w_blk                             # W reload (W indexed M,K)
        prev = idx
    full_c = w.M * w.N * FP32_BITS
    # A C tile accumulates over all outer-K visits. It stays resident for the whole
    # accumulation (one final write, no spill) UNLESS it is evicted and re-touched, which
    # happens only when K is tiled (K_out>1) AND some loop INNER to K is non-trivial (out>1)
    # so other (M,N) tiles interleave the K visits. K innermost (no inner loops) and K_out==1
    # are both output-stationary. (Binary "K innermost" alone would over-count the case where
    # K is outer but every loop nested inside it is trivial.)
    k_pos = m.perm.index("K")
    interleaved = any(out[d] > 1 for d in m.perm[k_pos + 1:])
    if out["K"] > 1 and interleaved:
        cw = full_c * out["K"]                          # one partial write per outer-K visit
        cr = full_c * (out["K"] - 1)                    # reload prior partial; first touch zero-init
    else:
        cw, cr = full_c, 0                              # output-stationary: no psum spill
    return {"A": a, "W": w_dram, "Cw": cw, "Cr": cr, "total": a + w_dram + cw + cr}


_DISP = {8: 1, 4: 2, 2: 4}   # RMW dispatches per col fire by activation mode


def mac_ops(w):
    return w.M * w.K * w.N


def rmw_ops(w):
    return w.MT * w.KT * w.NT * TILE * TILE * _DISP[w.act_bits]


def compute_work(w):
    # Σ_cube 32·b = TILE · NT · Σ_{mt,kt} wbits[mt][kt]  (order-independent ideal lower bound).
    # Exact (no int()): with fractional avg wbits this must equal Σ per-block compute in _blocks,
    # otherwise actual_cycle (= compute_work + fill + stall) would mix a truncated and an exact base.
    return TILE * w.NT * sum(sum(row) for row in w.wbits)


def onchip_bits(m, w):
    # All three terms are on-chip buffer accesses and share the single "onchip" energy
    # coefficient (spec §7): SA-facing reads (a_rd, w_rd) PLUS the on-chip write of data
    # refilled from DRAM. The DRAM-side transfer energy of that refill is counted
    # separately in dram_bits (weighted by the "dram" coeff) — this is the on-chip side.
    a_rd = (w.MT * w.KT * w.NT) * TILE * TILE * w.act_bits   # A read once per cube
    w_rd = w.NT * w.total_w_bits                              # W tile read once per n_t
    d = dram_bits(m, w)
    refill = d["A"] + d["W"] + d["Cr"]                        # DRAM -> on-chip loads (mapping-variable)
    return a_rd + w_rd + refill                               # exact (fractional when avg wbits is)


def energy_breakdown(m, w, hw):
    c = hw.coeffs
    e_dram = dram_bits(m, w)["total"] * c["dram"]
    e_onchip = onchip_bits(m, w) * c["onchip"]
    e_mac = mac_ops(w) * c["mac"]
    e_rmw = rmw_ops(w) * c["rmw"]
    return {"dram": e_dram, "onchip": e_onchip, "mac": e_mac, "rmw": e_rmw,
            "total": e_dram + e_onchip + e_mac + e_rmw}


def _blocks(m, w):
    """Yield (idx, compute_blk, a_blk, w_blk, c_blk) per outer block in perm order (outermost first).
    idx = outer index dict per dim; compute_blk = 32*N_in*Σwbits(resident M_in x K_in window);
    a_blk = A fetch bits (mapping-constant: same inner A tile every block);
    w_blk = W fetch bits for this block (varies — depends on the block's resident wbits);
    c_blk = resident C psum window bits (mapping-constant): spilled when its (M,N) tile is
            evicted, reloaded when re-entered."""
    out, inn = _out_in(m, w)
    a_blk = inn["K"] * inn["N"] * TILE * TILE * w.act_bits   # mapping-constant, same every block
    c_blk = inn["M"] * inn["N"] * TILE * TILE * FP32_BITS    # mapping-constant resident C window
    ranges = [range(out[d]) for d in m.perm]
    for combo in itertools.product(*ranges):
        idx = dict(zip(m.perm, combo))
        mo, ko = idx["M"], idx["K"]
        w_sum = sum(w.wbits[mt][kt]
                    for mt in range(mo * inn["M"], (mo + 1) * inn["M"])
                    for kt in range(ko * inn["K"], (ko + 1) * inn["K"]))
        yield idx, TILE * inn["N"] * w_sum, a_blk, w_sum * TILE * TILE, c_blk


def _stall_of_order(blocks, bw):
    """Total stall for a given block ORDER. One DRAM channel of bandwidth bw is shared by
    ALL traffic: input fetch (A,W reloads), C psum reload-reads, and C psum spill-writes.
    Each block's compute window hides the transfers around it -- the next block's inputs and
    C reload (must arrive before it computes) PLUS the previous block's C spill-write (drains
    after it computes). Leftover transfer time is stall. The last block's C write has no
    following compute to hide it -> trailing drain (always added).

    A reloads on (K,N) change, W on (M,K) change, C reloads when an (M,N) tile is re-entered
    after a spill, C spills when an (M,N) tile is left (every episode-end, incl. the final
    write). (Supersedes the inputs-only spec §8 stall: C psum bandwidth is now modeled.)"""
    n = len(blocks)
    if n == 0:
        return 0.0
    seen = set()
    prev = None
    prev_compute = 0.0
    total = 0.0
    for idx, compute_blk, a_blk, w_blk, c_blk in blocks:
        mn = (idx["M"], idx["N"])
        if prev is not None:
            prev_mn = (prev["M"], prev["N"])
            fetch = 0.0
            if (idx["K"], idx["N"]) != (prev["K"], prev["N"]):
                fetch += a_blk                              # A reloads (A indexed K,N)
            if (idx["M"], idx["K"]) != (prev["M"], prev["K"]):
                fetch += w_blk                              # W reloads (W indexed M,K)
            if mn != prev_mn:
                fetch += c_blk                              # prev block left its C tile -> spill drains
                if mn in seen:
                    fetch += c_blk                          # re-entering a spilled C tile -> reload read
            total += max(0.0, fetch / bw - prev_compute)
        seen.add(mn)
        prev = idx
        prev_compute = compute_blk
    total += blocks[-1][4] / bw                             # trailing C drain of the last tile
    return total


def stall_fill(m, w, hw):
    """Sequence-aware stall over outer-iteration blocks. DRAM bandwidth is shared by A/W
    input fetch AND C psum spill/reload (see _stall_of_order). fill = first block's input
    fetch time (unhidable startup); stall = the rest (per-block un-hidden transfer + the
    trailing C drain)."""
    blocks = list(_blocks(m, w))
    bw = hw.eff_bw                                          # DRAM bw in on-chip cycle terms
    _, _, a0, w0, _c0 = blocks[0]
    fill = (a0 + w0) / bw                                   # first-block C read is 0 (first touch)
    return _stall_of_order(blocks, bw), fill


def actual_cycle(m, w, hw):
    stall, fill = stall_fill(m, w, hw)
    return float(compute_work(w)) + fill + stall


def evaluate(m, w, hw):
    feas = feasible(m, w, hw)
    eb = energy_breakdown(m, w, hw)
    stall, fill = stall_fill(m, w, hw)
    cw = compute_work(w)
    return {
        "mapping": m,
        "feasible": feas,
        "energy": eb["total"],
        "energy_breakdown": eb,
        "dram": dram_bits(m, w),
        "compute_work": cw,
        "stall": stall,
        "fill": fill,
        "actual_cycle": float(cw) + fill + stall,
    }


def optimize(w, hw, max_cycle=None):
    """Exhaustive over (perm x blocking). Return feasible results sorted by energy asc,
    ties broken by ascending actual_cycle. If max_cycle given, further filters to
    actual_cycle <= max_cycle BEFORE the sort.

    Energy is order-dependent (DRAM traffic depends on the loop order), so perms generally
    differ; where two mappings still tie on energy, ascending actual_cycle breaks the tie
    deterministically (instead of gen_mappings insertion order). At equal energy, fewer cycles
    is strictly better, so it is the right key."""
    results = [evaluate(m, w, hw) for m in gen_mappings(w)]
    results = [r for r in results if r["feasible"]]
    if max_cycle is not None:
        results = [r for r in results if r["actual_cycle"] <= max_cycle + 1e-9]
    results.sort(key=lambda r: (r["energy"], r["actual_cycle"]))
    return results


def lpt_headroom(m, w, hw):
    """Headroom indicator (spec §9.2): natural perm-order stall vs the stall if outer blocks
    were reordered Longest-Processing-Time-first (descending block compute). Both stalls use
    the full bandwidth model (A/W fetch + C psum spill/reload). LPT is NOT a drop-in optimum
    here (tiles are reuse-coupled, and reordering perturbs C residency), so headroom may be <= 0."""
    blocks = list(_blocks(m, w))
    natural = _stall_of_order(blocks, hw.eff_bw)
    lpt = _stall_of_order(sorted(blocks, key=lambda b: -b[1]), hw.eff_bw)  # b[1] = compute_blk, desc
    return {"natural_stall": natural, "lpt_stall": lpt, "headroom": natural - lpt}


def report(ranked, w, hw, top=10):
    lines = [f"# GEMM ({w.M}x{w.K}x{w.N})  cap_bits={hw.cap_bits}  dram_bw={hw.dram_bw}",
             f"{'perm':<10}{'m_in':>5}{'k_in':>5}{'n_in':>5}{'energy':>16}{'actual_cycle':>16}{'stall':>12}"]
    for r in ranked[:top]:
        m = r["mapping"]
        lines.append(f"{''.join(m.perm):<10}{m.m_in:>5}{m.k_in:>5}{m.n_in:>5}"
                     f"{r['energy']:>16.0f}{r['actual_cycle']:>16.1f}{r['stall']:>12.1f}")
    return "\n".join(lines)


def selftest():
    # G1: all-resident (single block) -- DRAM + compute_work + energy (unchanged by order/C model)
    w1 = Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m1 = Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    hw1 = HW(bank_size=1024, banks=32, dram_bw=64)
    assert dram_bits(m1, w1)["total"] == 196608
    assert compute_work(w1) == 2048
    assert energy_breakdown(m1, w1, hw1)["total"] == 40804352.0
    # G2: genuine psum spill needs K_out>1 AND a non-trivial loop inner to K (here N, n_in=1)
    m2 = Mapping(perm=("K", "M", "N"), m_in=2, k_in=1, n_in=1)
    assert dram_bits(m2, w1)["Cw"] == 262144 and dram_bits(m2, w1)["Cr"] == 131072
    # G3: 2-bit weights -> tiny compute, FP32 C psum dominates the shared-bandwidth stall
    w3 = Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 2]], act_bits=8)
    m3 = Mapping(perm=("N", "M", "K"), m_in=2, k_in=2, n_in=1)
    hw3 = HW(bank_size=1024, banks=32, dram_bw=32)
    assert stall_fill(m3, w3, hw3) == (4352.0, 768.0)
    assert actual_cycle(m3, w3, hw3) == 5632.0
    # Pareto / tradeoff sanity
    wt = Work(M=128, K=128, N=128, wbits=[[2, 8, 2, 8]] * 4, act_bits=8)
    hwt = HW(bank_size=1024, banks=32, dram_bw=32)
    t = tradeoff(wt, hwt)
    assert t["off"]["energy"] <= t["on"]["energy"] + 1e-9
    assert t["on"]["actual_cycle"] <= t["off"]["actual_cycle"] + 1e-9
    print("selftest: OK")


def _load_wbits(path, MT, KT):
    import json
    with open(path) as f:
        wb = json.load(f)
    if len(wb) != MT or any(len(r) != KT for r in wb):
        raise ValueError(f"wbits file must be {MT}x{KT}")
    return wb


def pareto_front(ranked):
    """Non-dominated set on (energy, actual_cycle), sorted by energy asc. After sorting by
    (energy, cycle), a point is on the front iff its cycle strictly improves on every
    lower-energy point kept so far."""
    pts = sorted(ranked, key=lambda r: (r["energy"], r["actual_cycle"]))
    front = []
    best_cycle = float("inf")
    for r in pts:
        if r["actual_cycle"] < best_cycle - 1e-9:
            front.append(r)
            best_cycle = r["actual_cycle"]
    return front


def tradeoff(w, hw):
    """OFF = global min-energy mapping (cycle ignored). ON = min-energy among the
    minimum-actual-cycle mappings (the fastest schedule, cheapest energy at that speed)."""
    ranked = optimize(w, hw)
    if not ranked:
        raise ValueError("no feasible mapping for this HW (check capacity)")
    off = ranked[0]
    min_cycle = min(r["actual_cycle"] for r in ranked)
    on = min((r for r in ranked if r["actual_cycle"] <= min_cycle + 1e-9),
             key=lambda r: r["energy"])
    return {"off": off, "on": on, "pareto": pareto_front(ranked)}


def crosscheck(verbose=False):
    """Assert the annotated twin produces identical evaluate() output on a sweep (drift guard)."""
    import importlib, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ann = importlib.import_module("mxp_scheduler_annotated")
    # (M, K, N, wbits, act, bank_size, banks, dram_bw, freq_ratio, coeffs|None). The cases
    # deliberately cover all the drift-prone surfaces: every perm/blocking (via gen_mappings),
    # spill + no-spill, freq_ratio != 1, FRACTIONAL wbits (int()-truncation footgun) and a
    # NON-default coeffs override (energy-coeff plumbing). A one-sided edit to any of these
    # paths must trip the equivalence check.
    cases = [
        (64, 64, 64, [[2, 8], [8, 2]], 8, 1024, 32, 32.0, 1.0, None),
        # M=128 -> MT=4, K=96 -> KT=3 matches the 4x3 wbits map (asymmetric, mixed-prec, freq_ratio=2)
        (128, 96, 64, [[4, 2, 8], [8, 8, 2], [2, 4, 4], [8, 2, 8]], 4, 512, 16, 48.0, 2.0, None),
        # fractional avg wbits — guards the no-int()-truncation invariant on both files
        (64, 64, 64, [[2.5, 7.5], [6.0, 3.5]], 8, 1024, 32, 32.0, 1.0, None),
        # non-default coeffs + freq_ratio<1 — guards energy-coeff plumbing and eff_bw scaling
        (64, 64, 64, [[2, 8], [8, 2]], 8, 1024, 32, 32.0, 0.5,
         {"dram": 137.0, "onchip": 11.0, "mac": 2.0, "rmw": 3.0}),
    ]
    for (M, K, N, wb, act, bs, bk, bw, fr, co) in cases:
        w_s, w_a = Work(M, K, N, wb, act), ann.Work(M, K, N, wb, act)
        hw_s = HW(bs, bk, bw, freq_ratio=fr, coeffs=dict(co)) if co else HW(bs, bk, bw, freq_ratio=fr)
        hw_a = ann.HW(bs, bk, bw, freq_ratio=fr, coeffs=dict(co)) if co else ann.HW(bs, bk, bw, freq_ratio=fr)
        for m_s in gen_mappings(w_s):
            m_a = ann.Mapping(perm=m_s.perm, m_in=m_s.m_in, k_in=m_s.k_in, n_in=m_s.n_in)
            r_s, r_a = evaluate(m_s, w_s, hw_s), ann.evaluate(m_a, w_a, hw_a)
            for key in ("feasible", "energy", "actual_cycle", "stall", "fill"):
                assert r_s[key] == r_a[key], f"mismatch {key}: {r_s[key]} != {r_a[key]} @ {m_s}"
            assert r_s["dram"] == r_a["dram"], f"mismatch dram @ {m_s}"
            assert r_s["energy_breakdown"] == r_a["energy_breakdown"], f"mismatch energy_breakdown @ {m_s}"
            assert lpt_headroom(m_s, w_s, hw_s) == ann.lpt_headroom(m_a, w_a, hw_a), f"mismatch lpt @ {m_s}"
        # pareto/tradeoff parity across the twins (Mapping classes differ -> compare signatures)
        def _sig(r):
            mm = r["mapping"]
            return (mm.perm, mm.m_in, mm.k_in, mm.n_in, r["energy"], r["actual_cycle"])
        ranked_s, ranked_a = optimize(w_s, hw_s), ann.optimize(w_a, hw_a)
        assert [_sig(r) for r in pareto_front(ranked_s)] == [_sig(r) for r in ann.pareto_front(ranked_a)], \
            f"pareto mismatch @ case M={M}"
        ts, ta = tradeoff(w_s, hw_s), ann.tradeoff(w_a, hw_a)
        assert _sig(ts["off"]) == _sig(ta["off"]) and _sig(ts["on"]) == _sig(ta["on"]), \
            f"tradeoff mismatch @ case M={M}"
    print("crosscheck: OK")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="MXP_scheduler — GEMM mapping cost-model + optimizer")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--crosscheck", action="store_true")
    p.add_argument("--M", type=int); p.add_argument("--K", type=int); p.add_argument("--N", type=int)
    p.add_argument("--bank-size", type=int, default=1024)
    p.add_argument("--banks", type=int, default=32)
    p.add_argument("--dram-bw", type=float, default=64.0, help="bits per DRAM cycle")
    p.add_argument("--freq-ratio", type=float, default=1.0,
                   help="on-chip cycles per DRAM cycle (f_chip/f_dram); 1.0 = same clock")
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
    hw = HW(bank_size=a.bank_size, banks=a.banks, dram_bw=a.dram_bw,
            freq_ratio=a.freq_ratio, coeffs=coeffs)
    ranked = optimize(w, hw, max_cycle=a.max_cycle)
    print(report(ranked, w, hw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
