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
    return int(foot_a + foot_w + foot_c)


def feasible(m, w, hw):
    return footprint_bits(m, w) <= hw.cap_bits


def dram_bits(m, w):
    out, _ = _out_in(m, w)
    a = (w.K * w.N * w.act_bits) * out["M"]            # reload(A) = M_out
    wt = w.total_w_bits * out["N"]                      # reload(W) = N_out
    cw = (w.M * w.N * FP32_BITS) * out["K"]             # each outer-K writes partial (FP32 psum)
    cr = (w.M * w.N * FP32_BITS) * (out["K"] - 1)       # reload for accumulate; first touch zero-init
    return {"A": int(a), "W": int(wt), "Cw": int(cw), "Cr": int(cr),
            "total": int(a + wt + cw + cr)}


_DISP = {8: 1, 4: 2, 2: 4}   # RMW dispatches per col fire by activation mode


def mac_ops(w):
    return w.M * w.K * w.N


def rmw_ops(w):
    return w.MT * w.KT * w.NT * TILE * TILE * _DISP[w.act_bits]


def compute_work(w):
    # Σ_cube 32·b = TILE · NT · Σ_{mt,kt} wbits[mt][kt]  (order-independent ideal lower bound)
    return int(TILE * w.NT * sum(sum(row) for row in w.wbits))


def onchip_bits(m, w):
    # All three terms are on-chip buffer accesses and share the single "onchip" energy
    # coefficient (spec §7): SA-facing reads (a_rd, w_rd) PLUS the on-chip write of data
    # refilled from DRAM. The DRAM-side transfer energy of that refill is counted
    # separately in dram_bits (weighted by the "dram" coeff) — this is the on-chip side.
    a_rd = (w.MT * w.KT * w.NT) * TILE * TILE * w.act_bits   # A read once per cube
    w_rd = w.NT * w.total_w_bits                              # W tile read once per n_t
    d = dram_bits(m, w)
    refill = d["A"] + d["W"] + d["Cr"]                        # DRAM -> on-chip loads (mapping-variable)
    return int(a_rd + w_rd + refill)


def energy_breakdown(m, w, hw):
    c = hw.coeffs
    e_dram = dram_bits(m, w)["total"] * c["dram"]
    e_onchip = onchip_bits(m, w) * c["onchip"]
    e_mac = mac_ops(w) * c["mac"]
    e_rmw = rmw_ops(w) * c["rmw"]
    return {"dram": e_dram, "onchip": e_onchip, "mac": e_mac, "rmw": e_rmw,
            "total": e_dram + e_onchip + e_mac + e_rmw}


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
    """Exhaustive over (perm x blocking). Return feasible results sorted by energy asc.
    If max_cycle given, further filters to actual_cycle <= max_cycle BEFORE the sort."""
    results = [evaluate(m, w, hw) for m in gen_mappings(w)]
    results = [r for r in results if r["feasible"]]
    if max_cycle is not None:
        results = [r for r in results if r["actual_cycle"] <= max_cycle + 1e-9]
    results.sort(key=lambda r: r["energy"])
    return results
