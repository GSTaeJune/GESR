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
    dram_bw: float          # bits per cycle
    word_bits: int = 32     # FP32 psum word
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
    # Conservative GLOBAL upper bound for resident W bit-width: max over ALL tiles, not
    # just the resident m_in x k_in window. Safe direction (never marks an infeasible
    # mapping feasible); for mixed-precision partial blocking it can over-estimate W and
    # over-prune. A resident-window-aware refinement is a spec-level modeling decision.
    max_wbits = max(max(row) for row in w.wbits)
    foot_a = m.k_in * m.n_in * TILE * TILE * w.act_bits
    foot_w = m.m_in * m.k_in * TILE * TILE * max_wbits
    foot_c = m.m_in * m.n_in * TILE * TILE * FP32_BITS
    # exact (may be fractional when avg wbits is fractional); do NOT int()-truncate, which
    # would round a footprint DOWN and could mark an over-capacity mapping feasible.
    return foot_a + foot_w + foot_c


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
    for idx, _compute, a_blk, w_blk in _blocks(m, w):
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
    """Yield (idx, compute_blk, a_blk, w_blk) per outer block in perm order (perm[0] outermost).
    idx = outer index dict per dim; compute_blk = 32*N_in*Σwbits(resident M_in x K_in window);
    a_blk = A fetch bits (mapping-constant: same inner A tile every block);
    w_blk = W fetch bits for this block (varies — depends on the block's resident wbits)."""
    out, inn = _out_in(m, w)
    a_blk = inn["K"] * inn["N"] * TILE * TILE * w.act_bits   # mapping-constant, same every block
    ranges = [range(out[d]) for d in m.perm]
    for combo in itertools.product(*ranges):
        idx = dict(zip(m.perm, combo))
        mo, ko = idx["M"], idx["K"]
        w_sum = sum(w.wbits[mt][kt]
                    for mt in range(mo * inn["M"], (mo + 1) * inn["M"])
                    for kt in range(ko * inn["K"], (ko + 1) * inn["K"]))
        yield idx, TILE * inn["N"] * w_sum, a_blk, w_sum * TILE * TILE


def _stall_of_order(blocks, bw):
    """Sum of double-buffer stalls for a given block ORDER. Inputs (A,W) only (spec §8):
    each block's input fetch is hidden behind the PREVIOUS block's compute.
    A reloads when (K_idx, N_idx) changes; W reloads when (M_idx, K_idx) changes."""
    prev = None
    prev_compute = 0.0
    total = 0.0
    for idx, compute_blk, a_blk, w_blk in blocks:
        if prev is not None:
            fetch = 0.0
            if (idx["K"], idx["N"]) != (prev["K"], prev["N"]):
                fetch += a_blk                              # A reloads (A indexed K,N)
            if (idx["M"], idx["K"]) != (prev["M"], prev["K"]):
                fetch += w_blk                              # W reloads (W indexed M,K)
            total += max(0.0, fetch / bw - prev_compute)
        prev = idx
        prev_compute = compute_blk
    return total


def stall_fill(m, w, hw):
    """Sequence-aware stall over outer-iteration blocks. Inputs (A,W) only (spec §8).
    fill = first block's input fetch time (unhidable); stall = natural perm-order stall."""
    blocks = list(_blocks(m, w))
    bw = hw.dram_bw
    _, _, a0, w0 = blocks[0]
    fill = (a0 + w0) / bw
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

    The (energy, actual_cycle) key matters because energy depends only on the blocking
    (out factors), so all 6 perms of a given blocking tie on energy; without the cycle
    tiebreak the winner would fall to gen_mappings() insertion order — an arbitrary hidden
    preference. At equal energy, fewer cycles is strictly better, so it is the right key."""
    results = [evaluate(m, w, hw) for m in gen_mappings(w)]
    results = [r for r in results if r["feasible"]]
    if max_cycle is not None:
        results = [r for r in results if r["actual_cycle"] <= max_cycle + 1e-9]
    results.sort(key=lambda r: (r["energy"], r["actual_cycle"]))
    return results


def lpt_headroom(m, w, hw):
    """Headroom indicator (spec §9.2): natural perm-order stall vs the stall if outer blocks
    were reordered Longest-Processing-Time-first (descending block compute). LPT is NOT a
    drop-in optimum here (tiles are reuse-coupled), so headroom may be <= 0."""
    blocks = list(_blocks(m, w))
    natural = _stall_of_order(blocks, hw.dram_bw)
    lpt = _stall_of_order(sorted(blocks, key=lambda b: -b[1]), hw.dram_bw)  # b[1] = compute_blk, desc
    return {"natural_stall": natural, "lpt_stall": lpt, "headroom": natural - lpt}
