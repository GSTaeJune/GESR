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


def dram_bits(m, w):
    out, _ = _out_in(m, w)
    a = (w.K * w.N * w.act_bits) * out["M"]            # reload(A) = M_out
    wt = w.total_w_bits * out["N"]                      # reload(W) = N_out
    cw = (w.M * w.N * 32) * out["K"]                    # each outer-K writes partial
    cr = (w.M * w.N * 32) * (out["K"] - 1)              # reload for accumulate; first touch zero-init
    return {"A": int(a), "W": int(wt), "Cw": int(cw), "Cr": int(cr),
            "total": int(a + wt + cw + cr)}


_DISP = {8: 1, 4: 2, 2: 4}   # RMW dispatches per col fire by activation mode


def mac_ops(w):
    return w.M * w.K * w.N


def rmw_ops(w):
    return w.MT * w.KT * w.NT * TILE * TILE * _DISP[w.act_bits]


def compute_work(w):
    # Σ_cube 32·b = 32 · NT · Σ_{m,k} wbits  (order-independent ideal lower bound)
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
