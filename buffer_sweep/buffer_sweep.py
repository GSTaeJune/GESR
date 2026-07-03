# buffer_sweep/buffer_sweep.py
"""SRAM buffer-partition sweep for the MXP GEMM accelerator.

Standalone single-file program (stdlib + matplotlib for plots). Imports nothing
from MXP_scheduler/. Spec: docs/superpowers/specs/2026-07-03-buffer-sweep-design.md

SRAM organization (fixed): three dedicated spaces, each ping-pong duplicated (x2):
    W buffer: m x k weights   @ 8b   -> 8*m*k bits per copy
    A buffer: k x n acts      @ 8b   -> 8*k*n bits per copy
    O buffer: m x n psums     @ 32b  -> 32*m*n bits per copy
Budget: 2*(8*m*k + 8*k*n + 32*m*n) <= cap_bits. m, k, n are multiples of 32
(SA is 32x32: row=K, col=N, cycle=M).

Dataflow (fixed): O-stationary. For each output tile (i,j) (i over M, j over N),
sweep kk innermost accumulating into the O buffer; write the finished FP32 tile
to DRAM once (no psum spill/reload). Loop order: for i: for j: for kk.
Ping-pong: while computing step t, prefetch step t+1's W/A tiles into the shadow
copies; a finished O tile drains from the shadow O spread evenly across the next
tile's Kt steps (see o_share below). A tile is fetched iff its tile id differs
from the previous step's id (so Kt==1 keeps W(i) resident across the whole j
sweep, etc.).

Cost model:
    stall(t)   = max(0, (load(t+1) + o_share(t)) / eff_bw - compute(t))
    o_share(t) = prev tile's O write bits / Kt, spread evenly over the current
                 tile's Kt steps (the shadow O only has to be empty by the time
                 the CURRENT tile finishes, so its drain competes for DRAM BW
                 across the whole tile, not a single step) -- decision 2026-07-03
    fill       = load(0) / eff_bw          (first fetch, nothing to hide it)
    drain      = owrite(T-1) / eff_bw      (last O write, nothing to hide it)
    cycles     = compute_total + fill + steady_stall + drain
    energy_pj  = dram_bits*C_DRAM + onchip_bits*C_ONCHIP + mac*C_MAC + rmw*C_RMW

Known modeling conservatisms (deliberate; decisions 2026-07-03):
  * Ping-pong shadow copies are PREFETCH/DRAIN STAGING ONLY -- they never extend
    residency. E.g. at Kt==2 the two W copies could physically hold both k-chunks
    and skip all W refetch across the j sweep; the model still charges the
    refetch (up to ~2x W traffic overstatement for exactly-Kt==2 partitions).
    Keeps the controller model a simple toggle.
  * The O drain is spread EVENLY over the next tile's steps (trickle writeback);
    an optimal scheduler could exploit per-step slack slightly better.
  * EFF_BW is peak DRAM bandwidth (no effective-BW derate); stalls are optimistic.

Two evaluators (single cost semantics, two implementations):
    walk_gemm()  -- naive per-step reference walk (slow, exact)
    eval_gemm()  -- grouped analytic (fast; tile classes x multiplicities)
`--selftest` asserts they agree bit-for-bit-ish (isclose) over a battery of
shapes covering Mt/Nt/Kt in {1,2,3,4,5}, residual tiles, and reuse corners.

Usage:
    python buffer_sweep.py                       # all models, caps 64/128/256 KB
    python buffer_sweep.py --model deit_tiny --caps 64
    python buffer_sweep.py --selftest
Outputs: work/buffer_sweep/<model>.csv, <model>_cap<KB>_{energy,cycles}.png,
summary_{energy,cycles}.png, console top-10 tables (ASCII only).
"""
import argparse
import csv
import math
import os

# ----------------------------------------------------------------------------
# CONFIG -- edit these directly.
# ----------------------------------------------------------------------------
TILE = 32                 # SA dimension; m/k/n granularity
WBITS = 8                 # weight precision (bits)
ABITS = 8                 # activation precision (bits)
OBITS = 32                # output psum precision (FP32)

CAPS_KB = [64, 128, 256]  # physical SRAM totals to sweep (include ping-pong x2)
K_LIST = [32, 64, 128, 256, 512, 1024]   # k (buffer K-depth) candidates

# Energy coefficients, pJ per bit / per op. DRAM anchored at LPDDR5 full-system
# 9.0 pJ/b (docs/dram-energy/README.md). The other three keep M0's relative
# structure (dram:onchip:mac:rmw = 200:6:1:5) rescaled to the 9.0 anchor --
# placeholders: refresh onchip per cap from CACTI when needed.
C_DRAM = 9.0              # pJ per DRAM bit (read or write)
C_ONCHIP = 0.27           # pJ per on-chip SRAM bit access
C_MAC = 0.045             # pJ per 1b-MAC-equivalent op
C_RMW = 0.225             # pJ per RMW lane op

# Bandwidth: LPDDR5-6400 x16 -> 32 bits per DRAM bus cycle @ 3200 MHz.
DRAM_BW_BITS_PER_CYCLE = 32.0
DRAM_FREQ_MHZ = 3200.0
CHIP_FREQ_MHZ = 1000.0
EFF_BW = DRAM_BW_BITS_PER_CYCLE * DRAM_FREQ_MHZ / CHIP_FREQ_MHZ  # bits per chip cycle

CYCLES_PER_BIT = 1.0      # on-chip cycles per weight bit-plane (bit-serial SA)

# anchored to the repo root (next to this file's parent), not the cwd
OUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "work", "buffer_sweep"))

# ----------------------------------------------------------------------------
# Workload library. Each model: list of (name, M, K, N, count_per_layer) x layers.
# Attention BMMs included (count = #query heads). Dims are padded up to
# multiples of TILE at evaluation time (only S=197 -> 224 is actually ragged).
# ----------------------------------------------------------------------------
def _transformer_gemms(S, D, heads, kv_heads, head_dim, mlp_gemms):
    """Per-layer GEMM list shared by all four models.
    mlp_gemms: [(name, K_in, N_out, count), ...] -- plain MLP (fc1/fc2) or
    SwiGLU (gate/up/down)."""
    q_out = heads * head_dim
    kv_out = kv_heads * head_dim
    g = [
        ("q_proj",   S, D, q_out, 1),
        ("k_proj",   S, D, kv_out, 1),
        ("v_proj",   S, D, kv_out, 1),
        ("out_proj", S, q_out, D, 1),
        ("qk_bmm",   S, head_dim, S, heads),   # Q.K^T per head
        ("av_bmm",   S, S, head_dim, heads),   # attn.V per head
    ]
    g += [(name, S, kin, nout, cnt) for (name, kin, nout, cnt) in mlp_gemms]
    return g


MODELS = {
    # DeiT: S = 196 patches + CLS = 197 (padded to 224). Plain MLP (fc1/fc2).
    "deit_tiny": {
        "layers": 12,
        "gemms": _transformer_gemms(
            S=197, D=192, heads=3, kv_heads=3, head_dim=64,
            mlp_gemms=[("fc1", 192, 768, 1), ("fc2", 768, 192, 1)]),
    },
    "deit_small": {
        "layers": 12,
        "gemms": _transformer_gemms(
            S=197, D=384, heads=6, kv_heads=6, head_dim=64,
            mlp_gemms=[("fc1", 384, 1536, 1), ("fc2", 1536, 384, 1)]),
    },
    # Qwen2.5 prefill S=128. GQA (kv_heads < heads). SwiGLU (gate+up+down).
    "qwen2.5-0.5b": {
        "layers": 24,
        "gemms": _transformer_gemms(
            S=128, D=896, heads=14, kv_heads=2, head_dim=64,
            mlp_gemms=[("gate", 896, 4864, 1), ("up", 896, 4864, 1),
                       ("down", 4864, 896, 1)]),
    },
    "qwen2.5-1.5b": {
        "layers": 28,
        "gemms": _transformer_gemms(
            S=128, D=1536, heads=12, kv_heads=2, head_dim=128,
            mlp_gemms=[("gate", 1536, 8960, 1), ("up", 1536, 8960, 1),
                       ("down", 8960, 1536, 1)]),
    },
}


# ----------------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------------
def pad32(x):
    return ((x + TILE - 1) // TILE) * TILE


def budget_bits(m, k, n):
    """Physical SRAM bits used by the (m,k,n) partition, ping-pong x2 on all three."""
    return 2 * (WBITS * m * k + ABITS * k * n + OBITS * m * n)


def _axis(dim_padded, t):
    """Tile the padded dim by t. Returns (nt, full, last):
    nt tiles; interior tiles are `full` (= t), the last tile is `last`
    (== full when t divides the dim). All values multiples of TILE."""
    nt = (dim_padded + t - 1) // t
    last = dim_padded - (nt - 1) * t
    full = t if nt > 1 else last
    return nt, full, last


def _comp(ms, ks, ns):
    """Compute cycles of one (ms x ks x ns) step: bit-serial SA, per 32^3 cube
    = CYCLES_PER_BIT * 32 * WBITS cycles. Order-invariant."""
    return CYCLES_PER_BIT * WBITS * ms * ks * ns / (TILE * TILE)


# ----------------------------------------------------------------------------
# Reference evaluator: naive per-step walk (used by --selftest; exact but slow)
# ----------------------------------------------------------------------------
def walk_gemm(M, K, N, m, k, n, bw=None):
    """Walk every step (i,j,kk) of the fixed dataflow. Returns dict:
    reads, writes (DRAM bits), fill, steady, drain (cycles), compute (cycles)."""
    bw = EFF_BW if bw is None else bw
    Mp, Kp, Np = pad32(M), pad32(K), pad32(N)
    Mt, mf, ml = _axis(Mp, m)
    Nt, nf, nl = _axis(Np, n)
    Kt, kf, kl = _axis(Kp, k)
    msz = lambda i: ml if i == Mt - 1 else mf
    nsz = lambda j: nl if j == Nt - 1 else nf
    ksz = lambda q: kl if q == Kt - 1 else kf

    steps = [(i, j, q) for i in range(Mt) for j in range(Nt) for q in range(Kt)]
    T = len(steps)
    load = [0.0] * T
    owr = [0.0] * T
    comp = [0.0] * T
    prev_w = prev_a = None
    for t, (i, j, q) in enumerate(steps):
        wid, aid = (i, q), (q, j)
        if wid != prev_w:
            load[t] += msz(i) * ksz(q) * WBITS
        if aid != prev_a:
            load[t] += ksz(q) * nsz(j) * ABITS
        if q == Kt - 1:                       # tile (i,j) finishes at this step
            owr[t] = msz(i) * nsz(j) * OBITS
        comp[t] = _comp(msz(i), ksz(q), nsz(j))
        prev_w, prev_a = wid, aid

    fill = load[0] / bw
    steady = 0.0
    for t, (i, j, q) in enumerate(steps):
        nxt = load[t + 1] if t + 1 < T else 0.0
        # previous tile's O write drains evenly across this tile's Kt steps
        if (i, j) != (0, 0):
            pi, pj = (i, j - 1) if j > 0 else (i - 1, Nt - 1)
            share = msz(pi) * nsz(pj) * OBITS / Kt
        else:
            share = 0.0
        steady += max(0.0, (nxt + share) / bw - comp[t])
    drain = owr[T - 1] / bw
    return {"reads": sum(load), "writes": sum(owr), "fill": fill,
            "steady": steady, "drain": drain, "compute": sum(comp)}


# ----------------------------------------------------------------------------
# Fast evaluator: grouped analytic (tile classes x multiplicities)
# ----------------------------------------------------------------------------
def _reps(T):
    """Representative indices of an axis of T tiles with multiplicities.
    Distinguishing features are: pos==0, pos==T-2 (successor is last),
    pos==T-1 (last), everything else identical (generic interior)."""
    out = []
    seen = set()
    for pos in ([0, 1, T - 2, T - 1] if T >= 2 else [0]):
        if pos < 0 or pos in seen:
            continue
        seen.add(pos)
        mult = (T - 3) if (T >= 4 and pos == 1) else 1
        out.append((pos, mult))
    return out


def eval_gemm(M, K, N, m, k, n, bw=None):
    """Grouped-analytic equivalent of walk_gemm (same return dict). Enumerates
    <=16 (i,j) tile classes instead of Mt*Nt*Kt steps."""
    bw = EFF_BW if bw is None else bw
    Mp, Kp, Np = pad32(M), pad32(K), pad32(N)
    Mt, mf, ml = _axis(Mp, m)
    Nt, nf, nl = _axis(Np, n)
    Kt, kf, kl = _axis(Kp, k)
    msz = lambda i: ml if i == Mt - 1 else mf
    nsz = lambda j: nl if j == Nt - 1 else nf
    k0 = kf if Kt > 1 else kl               # k-size of a tile's first step
    k1 = kl if Kt == 2 else kf              # k-size of a tile's second step

    reads = writes = steady = 0.0
    fill = drain = 0.0

    for (i, mi) in _reps(Mt):
        for (j, mj) in _reps(Nt):
            mult = mi * mj
            ms, ns = msz(i), nsz(j)
            first = (i == 0 and j == 0)
            last = (i == Mt - 1 and j == Nt - 1)

            # O write of the PREVIOUS tile (drained evenly across this tile's
            # Kt steps via `share` below)
            if first:
                prev_o = 0.0
            elif j > 0:
                prev_o = ms * nsz(j - 1) * OBITS       # nsz(j-1) is never last here
            else:
                prev_o = mf * nl * OBITS               # row change: prev=(i-1, Nt-1)

            # load bits of the SUCCESSOR tile's first step (W part skips if id kept)
            if not last:
                if j < Nt - 1:                          # succ = (i, j+1)
                    w_part = ms * k0 * WBITS if Kt > 1 else 0.0
                    a_part = k0 * nsz(j + 1) * ABITS
                else:                                   # succ = (i+1, 0)
                    w_part = msz(i + 1) * k0 * WBITS
                    a_part = 0.0 if (Kt == 1 and Nt == 1) else k0 * nsz(0) * ABITS
                succ_l = w_part + a_part
            else:
                succ_l = 0.0

            # ---- traffic ----
            own_first = ms * k0 * WBITS + k0 * ns * ABITS  # this tile's 1st-step load
            if first:
                fill = own_first / bw
                reads += own_first * mult                  # not part of any succ_l
            # 1st-step loads of non-first tiles are charged via their
            # predecessor's succ_l (counted below), avoiding double count.
            within = (ms * (Kp - k0) * WBITS + (Kp - k0) * ns * ABITS) if Kt > 1 else 0.0
            reads += (within + succ_l) * mult
            writes += ms * ns * OBITS * mult

            # ---- stall ----
            comp_f = _comp(ms, kf, ns)
            comp_l = _comp(ms, kl, ns)
            share = prev_o / Kt          # prev-tile O drain, spread over Kt steps
            if Kt == 1:
                # single-step tile: window = succ load + the whole O share
                steady += mult * max(0.0, (succ_l + share) / bw - comp_l)
            else:
                # first step hides: 2nd step's W+A + its O drain share
                w0 = (ms * k1 * WBITS + k1 * ns * ABITS + share)
                steady += mult * max(0.0, w0 / bw - comp_f)
                # interior steps q=1..Kt-2 hide load of step q+1 (full k except last)
                n_into_full = max(0, Kt - 3)
                if n_into_full:
                    wf = (ms * kf * WBITS + kf * ns * ABITS + share)
                    steady += mult * n_into_full * max(0.0, wf / bw - comp_f)
                if Kt >= 3:
                    # transition s[Kt-2] -> s[Kt-1] (into the residual-k step);
                    # for Kt==2 that transition IS the first step's window above.
                    wl = (ms * kl * WBITS + kl * ns * ABITS + share)
                    steady += mult * max(0.0, wl / bw - comp_f)
                # final step hides the successor tile's first load + O share
                steady += mult * max(0.0, (succ_l + share) / bw - comp_l)
            if last:
                drain = ms * ns * OBITS / bw

    compute = CYCLES_PER_BIT * WBITS * Mp * Kp * Np / (TILE * TILE)
    return {"reads": reads, "writes": writes, "fill": fill,
            "steady": steady, "drain": drain, "compute": compute}


# ----------------------------------------------------------------------------
# Energy / cycles rollup
# ----------------------------------------------------------------------------
def gemm_cost(M, K, N, m, k, n):
    """Full cost of one GEMM on the (m,k,n) partition. Uses padded dims."""
    r = eval_gemm(M, K, N, m, k, n)
    Mp, Kp, Np = pad32(M), pad32(K), pad32(N)
    cubes = (Mp // TILE) * (Kp // TILE) * (Np // TILE)
    dram = r["reads"] + r["writes"]
    # on-chip accesses: DRAM refill writes + SA-facing reads (A and W each read
    # once per cube). O-buffer accumulate accesses (32b read+write per output
    # element per k-chunk) are folded into the rmw term at C_RMW per lane op --
    # a known undercount in absolute pJ, but config-invariant (does not move
    # the argmin over (m,k,n)).
    onchip = r["reads"] + cubes * TILE * TILE * (ABITS + WBITS)
    mac = Mp * Kp * Np
    rmw = cubes * TILE * TILE           # one RMW lane op per output element pass
    energy = dram * C_DRAM + onchip * C_ONCHIP + mac * C_MAC + rmw * C_RMW
    cycles = r["compute"] + r["fill"] + r["steady"] + r["drain"]
    return {"dram_read": r["reads"], "dram_write": r["writes"],
            "energy": energy, "cycles": cycles, "steady": r["steady"]}


def model_cost(model, m, k, n):
    """Sum gemm_cost over the model's per-layer GEMMs x layer count."""
    spec = MODELS[model]
    L = spec["layers"]
    tot = {"dram_read": 0.0, "dram_write": 0.0, "energy": 0.0,
           "cycles": 0.0, "steady": 0.0}
    for (_name, M, K, N, cnt) in spec["gemms"]:
        c = gemm_cost(M, K, N, m, k, n)
        for key in tot:
            tot[key] += c[key] * cnt * L
    return tot


# ----------------------------------------------------------------------------
# Sweep driver
# ----------------------------------------------------------------------------
def sweep_model(model, caps_kb, klist):
    """Enumerate all feasible (cap, k, m, n) and cost them. Returns row dicts.
    m/n larger than every GEMM's padded dim are strictly dominated (identical
    cost, more SRAM), so enumeration is capped there -- noted on the console,
    no silent truncation."""
    spec = MODELS[model]
    max_m = max(pad32(M) for (_g, M, _K, _N, _c) in spec["gemms"])
    max_n = max(pad32(N) for (_g, _M, _K, N, _c) in spec["gemms"])
    print("%s: m capped at %d, n at %d (model's max padded dims)"
          % (model, max_m, max_n))
    rows = []
    for cap_kb in caps_kb:
        cap_bits = cap_kb * 1024 * 8
        for k in klist:
            if budget_bits(TILE, k, TILE) > cap_bits:
                continue                                  # k alone too deep
            m = TILE
            while m <= max_m and budget_bits(m, k, TILE) <= cap_bits:
                n = TILE
                while n <= max_n and budget_bits(m, k, n) <= cap_bits:
                    c = model_cost(model, m, k, n)
                    rows.append({
                        "model": model, "cap_kb": cap_kb, "k": k, "m": m, "n": n,
                        "sram_bits": budget_bits(m, k, n),
                        "dram_read_bits": c["dram_read"],
                        "dram_write_bits": c["dram_write"],
                        "energy_pj": c["energy"], "cycles": c["cycles"],
                        "steady_stall": c["steady"],
                        "edp": c["energy"] * c["cycles"],
                    })
                    n += TILE
                m += TILE
    return rows


def write_csv(rows, path):
    cols = ["model", "cap_kb", "k", "m", "n", "sram_bits", "dram_read_bits",
            "dram_write_bits", "energy_pj", "cycles", "steady_stall", "edp"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)


def print_top(rows, model, cap_kb, top=10):
    sub = [r for r in rows if r["cap_kb"] == cap_kb]
    print("%s  cap=%dKB  (%d feasible configs)" % (model, cap_kb, len(sub)))
    if not sub:
        print("  (no feasible configuration)")
        return
    # sram_bits last: among cost ties prefer the smallest physical SRAM
    by_e = sorted(sub, key=lambda r: (r["energy_pj"], r["cycles"], r["sram_bits"]))
    by_c = min(sub, key=lambda r: (r["cycles"], r["energy_pj"], r["sram_bits"]))
    print("  %4s %5s %5s %14s %14s %12s" % ("k", "m", "n", "energy(uJ)", "cycles(M)", "stall(cyc)"))
    for r in by_e[:top]:
        print("  %4d %5d %5d %14.3f %14.3f %12.0f"
              % (r["k"], r["m"], r["n"], r["energy_pj"] / 1e6,
                 r["cycles"] / 1e6, r["steady_stall"]))
    print("  best-by-cycles: k=%d m=%d n=%d  cycles=%.3fM  energy=%.3fuJ"
          % (by_c["k"], by_c["m"], by_c["n"], by_c["cycles"] / 1e6,
             by_c["energy_pj"] / 1e6))


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def _plt():
    """Lazy matplotlib import; None if unavailable (plots skipped, sweep continues)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_model(rows, model, cap_kb, metric, out_path):
    """One figure per (model, cap, metric): heatmap over (m,n) per k subplot."""
    plt = _plt()
    sub = [r for r in rows if r["cap_kb"] == cap_kb]
    if plt is None or not sub:
        return False
    ks = sorted({r["k"] for r in sub})
    best = min(sub, key=lambda r: r[metric])
    vmin = min(r[metric] for r in sub)          # shared color scale across k
    vmax = max(r[metric] for r in sub)
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 4), squeeze=False)
    for ax, k in zip(axes[0], ks):
        kk = [r for r in sub if r["k"] == k]
        ms = sorted({r["m"] for r in kk})
        ns = sorted({r["n"] for r in kk})
        grid = [[math.nan] * len(ns) for _ in ms]
        for r in kk:
            grid[ms.index(r["m"])][ns.index(r["n"])] = r[metric]
        half = TILE // 2
        im = ax.imshow(grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
                       extent=[ns[0] - half, ns[-1] + half, ms[0] - half, ms[-1] + half])
        ax.set_title("k=%d" % k)
        ax.set_xlabel("n")
        ax.set_ylabel("m")
        if best["k"] == k:
            ax.plot(best["n"], best["m"], "r*", markersize=14)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("%s cap=%dKB %s (star = best: k=%d m=%d n=%d)"
                 % (model, cap_kb, metric, best["k"], best["m"], best["n"]))
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def plot_summary(all_rows, caps_kb, metric, out_path):
    """Best <metric> vs cap, one line per model."""
    plt = _plt()
    if plt is None or not all_rows:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    for model in sorted({r["model"] for r in all_rows}):
        xs, ys = [], []
        for cap in caps_kb:
            sub = [r for r in all_rows if r["model"] == model and r["cap_kb"] == cap]
            if sub:
                xs.append(cap)
                ys.append(min(r[metric] for r in sub))
        ax.plot(xs, ys, marker="o", label=model)
    ax.set_xlabel("SRAM cap (KB, incl. ping-pong)")
    ax.set_ylabel("best " + metric)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------
# Selftest: grouped evaluator == naive walk, plus hand-checked invariants
# ----------------------------------------------------------------------------
def selftest():
    shapes = []
    # battery: Mt/Nt/Kt in {1,2,3,4,5} incl. residual tiles and reuse corners
    # (K=128 with k=32 gives Kt=4: first case with BOTH an interior-full and a
    #  residual transition in one tile)
    for M in (32, 64, 96, 160, 224):
        for K in (32, 96, 128, 160):
            for N in (32, 64, 224):
                for m in (32, 64, 96):
                    for k in (32, 64, 160):
                        for n in (32, 96, 224):
                            shapes.append((M, K, N, m, k, n))
    shapes.append((224, 1024, 224, 32, 32, 32))   # deep-K spot check (Kt=32)
    shapes.append((197, 192, 160, 32, 64, 96))    # ragged M (DeiT S=197 -> 224)
    n_checked = 0
    for (M, K, N, m, k, n) in shapes:
        for bw in (EFF_BW, 1.0, 1e9):     # default, stall-heavy, stall-free
            a = walk_gemm(M, K, N, m, k, n, bw)
            b = eval_gemm(M, K, N, m, k, n, bw)
            for key in ("reads", "writes", "fill", "steady", "drain", "compute"):
                assert math.isclose(a[key], b[key], rel_tol=1e-9, abs_tol=1e-6), \
                    "mismatch %s: %r vs %r at %s bw=%g" % (
                        key, a[key], b[key], (M, K, N, m, k, n), bw)
            n_checked += 1

    # hand-checked invariants (generic case, no reuse corners): closed forms
    M, K, N, m, k, n = 224, 160, 224, 32, 32, 32
    Mp, Kp, Np = pad32(M), pad32(K), pad32(N)
    Mt, Nt = Mp // m, Np // n
    r = eval_gemm(M, K, N, m, k, n)
    assert r["writes"] == Mp * Np * OBITS
    assert r["reads"] == Nt * Mp * Kp * WBITS + Mt * Kp * Np * ABITS

    # Kt==1 keeps W(i) resident across the j sweep: W loaded once per i only
    M, K, N, m, k, n = 96, 64, 128, 32, 64, 32
    r = eval_gemm(M, K, N, m, k, n)
    Mp, Kp, Np = pad32(M), pad32(K), pad32(N)
    Mt = Mp // m
    assert r["reads"] == Mp * Kp * WBITS + Mt * Kp * Np * ABITS

    # ragged dims pad to the same problem: 197 == 224 after pad32
    for (m, k, n) in ((32, 32, 32), (64, 96, 32)):
        assert gemm_cost(197, 192, 192, m, k, n) == gemm_cost(224, 192, 192, m, k, n)

    # energy/cycles rollup, hand-checked on a single-cube GEMM (Mt=Nt=Kt=1)
    c = gemm_cost(32, 32, 32, 32, 32, 32)
    reads = 2 * 32 * 32 * 8                       # one W + one A tile, first touch
    writes = 32 * 32 * OBITS                      # one O tile
    onchip = reads + 32 * 32 * (ABITS + WBITS)    # refill + SA reads of 1 cube
    assert c["dram_read"] == reads and c["dram_write"] == writes
    assert math.isclose(c["energy"], (reads + writes) * C_DRAM + onchip * C_ONCHIP
                        + 32 ** 3 * C_MAC + 32 * 32 * C_RMW)
    assert math.isclose(c["cycles"],
                        CYCLES_PER_BIT * WBITS * 32 + reads / EFF_BW + writes / EFF_BW)

    # budget arithmetic
    assert budget_bits(32, 1024, 32) == 2 * (8 * 32 * 1024 + 8 * 1024 * 32 + 32 * 32 * 32)

    # huge bw -> no steady stall; tiny bw -> stall
    assert eval_gemm(224, 224, 224, 64, 64, 64, bw=1e12)["steady"] == 0.0
    assert eval_gemm(224, 224, 224, 64, 64, 64, bw=1.0)["steady"] > 0.0

    print("selftest OK (%d walk-vs-grouped cases + invariants)" % n_checked)
    return 0


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="SRAM (m,k,n) buffer-partition sweep")
    p.add_argument("--model", default="all", choices=["all"] + sorted(MODELS))
    p.add_argument("--caps", default=",".join(str(c) for c in CAPS_KB),
                   help="comma-separated caps in KB (default %(default)s)")
    p.add_argument("--klist", default=",".join(str(k) for k in K_LIST),
                   help="comma-separated k candidates (default %(default)s)")
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()

    try:
        caps = [int(x) for x in a.caps.split(",")]
        klist = [int(x) for x in a.klist.split(",")]
    except ValueError as e:
        p.error("--caps/--klist must be comma-separated integers (%s)" % e)
    caps = list(dict.fromkeys(caps))          # dedupe, keep order
    klist = list(dict.fromkeys(klist))
    for c in caps:
        if c <= 0:
            p.error("cap %d must be a positive KB count" % c)
    for k in klist:
        if k <= 0 or k % TILE:
            p.error("k=%d must be a positive multiple of %d" % (k, TILE))
    models = sorted(MODELS) if a.model == "all" else [a.model]
    os.makedirs(a.out, exist_ok=True)
    if _plt() is None:
        print("WARNING: matplotlib not available -- plots skipped, CSVs still written")

    METRICS = {"energy_pj": "energy", "cycles": "cycles"}  # metric key -> file tag
    all_rows = []
    for model in models:
        rows = sweep_model(model, caps, klist)
        all_rows += rows
        write_csv(rows, os.path.join(a.out, "%s.csv" % model))
        for cap in caps:
            print_top(rows, model, cap, a.top)
            for metric, tag in METRICS.items():
                name = "%s_cap%d_%s.png" % (model, cap, tag)
                plot_model(rows, model, cap, metric, os.path.join(a.out, name))
            print()

    for metric, tag in METRICS.items():
        plot_summary(all_rows, caps, metric,
                     os.path.join(a.out, "summary_%s.png" % tag))
    if not all_rows:
        print("no feasible configuration for any model/cap -- nothing to report")
        return 1
    print("wrote CSVs and plots to %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
