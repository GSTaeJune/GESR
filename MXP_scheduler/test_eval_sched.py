# MXP_scheduler/test_eval_sched.py
import pytest
import mxp_scheduler as s
import eval_sched as es


def test_all_cubes_canonical_order():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [8, 2]], act_bits=4)  # MT=2,KT=2,NT=1
    cubes = es.all_cubes(w)
    assert len(cubes) == 2 * 2 * 1
    # canonical order = product(range(MT), range(KT), range(NT))
    assert cubes == [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]


def test_tile_identities():
    c = (1, 0, 2)  # (mt, kt, nt)
    assert es.a_tile(c) == ("A", 0, 2)   # A indexed (kt, nt)
    assert es.w_tile(c) == ("W", 1, 0)   # W indexed (mt, kt)
    assert es.c_tile(c) == ("C", 1, 2)   # C indexed (mt, nt)


def test_tile_sizes():
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 8], [4, 6]], act_bits=4)
    assert es.tile_size(("A", 0, 0), w) == s.TILE * s.TILE * 4          # act_bits
    assert es.tile_size(("W", 0, 1), w) == 8 * s.TILE * s.TILE          # wbits[0][1]
    assert es.tile_size(("W", 1, 0), w) == 4 * s.TILE * s.TILE          # wbits[1][0]
    assert es.tile_size(("C", 0, 0), w) == s.TILE * s.TILE * s.FP32_BITS


def test_cube_compute_sums_to_compute_work():
    # per-cube compute summed over all cubes must equal compute_work(w, cpb)
    w = s.Work(M=64, K=96, N=64, wbits=[[2, 8, 4], [6, 2, 8]], act_bits=8)  # MT=2,KT=3,NT=2
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32, cycles_per_bit=1.5)
    total = sum(es.cube_compute(c, w, hw) for c in es.all_cubes(w))
    assert total == s.compute_work(w, hw.cycles_per_bit)


def test_cube_compute_value():
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 8], [4, 6]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32, cycles_per_bit=2.0)
    # cube (mt=0,kt=1,nt=0): cycles_per_bit * TILE * wbits[0][1] = 2.0 * 32 * 8 = 512
    assert es.cube_compute((0, 1, 0), w, hw) == 2.0 * 32 * 8


def _state0():
    return es.SchedState(done=frozenset(), resident=frozenset(), last_compute=-1.0)


def test_apply_cube_first_cube_loads_a_w_and_zero_inits_c():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)  # MT=2,KT=2,NT=1
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    c = (0, 0, 0)
    ns, read, spill, unhidden, cap_ok = es.apply_cube(_state0(), c, frozenset(), w, hw)
    a_sz = s.TILE * s.TILE * 2     # 2048
    w_sz = 2 * s.TILE * s.TILE     # 2048
    assert read == a_sz + w_sz     # C is zero-init (counter 0) -> free
    assert spill == 0.0
    assert cap_ok is True
    assert ns.last_compute == es.cube_compute(c, w, hw)
    assert (es.a_tile(c) in ns.resident and es.w_tile(c) in ns.resident
            and es.c_tile(c) in ns.resident)        # C occupies space even though load was free
    # first cube: unhidden == fill == (read+spill)/eff_bw
    assert unhidden == (read + spill) / hw.eff_bw


def test_apply_cube_evicting_partial_c_charges_spill():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    # state: C(0,0) resident with counter 1 (cube (0,0,0) already done), about to run (1,0,0)
    st = es.SchedState(done=frozenset({(0, 0, 0)}),
                       resident=frozenset({("A", 0, 0), ("W", 0, 0), ("C", 0, 0)}),
                       last_compute=es.cube_compute((0, 0, 0), w, hw))
    c = (1, 0, 0)  # needs A(0,0)[resident], W(1,0)[new], C(1,0)[new zero-init]
    ns, read, spill, unhidden, cap_ok = es.apply_cube(st, c, frozenset({("C", 0, 0)}), w, hw)
    c_sz = s.TILE * s.TILE * s.FP32_BITS   # 32768
    assert spill == c_sz                   # C(0,0) counter 1, 0<1<KT=2 -> partial spill
    assert read == 2 * s.TILE * s.TILE     # only W(1,0); A resident, C(1,0) zero-init free
    assert ("C", 0, 0) not in ns.resident


def test_apply_cube_reload_of_spilled_c_is_charged():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    # C(0,0) counter 1 but NOT resident (was spilled). Re-running a cube into it reloads.
    st = es.SchedState(done=frozenset({(0, 0, 0)}),
                       resident=frozenset({("A", 1, 0), ("W", 0, 1)}),
                       last_compute=10.0 ** 9)   # huge -> nothing stalls
    c = (0, 1, 0)  # needs A(1,0)[resident], W(0,1)[resident], C(0,0)[counter 1 -> reload]
    ns, read, spill, unhidden, cap_ok = es.apply_cube(st, c, frozenset(), w, hw)
    c_sz = s.TILE * s.TILE * s.FP32_BITS
    assert read == c_sz            # only the C reload
    assert spill == 0.0
    assert unhidden == 0.0         # transfer fully hidden by the huge prev compute


def test_apply_cube_capacity_violation_flagged():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    # cap smaller than a single cube's working set (A+W+C = 2048+2048+32768 = 36864)
    hw = s.HW(bank_size=1, banks=1, dram_bw=32, word_bits=32)   # cap_bits = 32
    ns, read, spill, unhidden, cap_ok = es.apply_cube(_state0(), (0, 0, 0), frozenset(), w, hw)
    assert cap_ok is False


def test_eviction_choices_includes_empty_when_fits():
    # cube fits with no eviction -> empty set is among the choices (and, since deficit<=0,
    # so is every subset of evictable -> voluntary eviction is offered).
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)   # big cap
    st = es.SchedState(done=frozenset({(0, 0, 0)}),
                       resident=frozenset({("A", 0, 0), ("W", 0, 0), ("C", 0, 0)}),
                       last_compute=1.0)
    choices = set(es.eviction_choices(st, (1, 0, 0), w, hw))
    assert frozenset() in choices
    # evictable = resident - needed(cube (1,0,0)=A(0,0),W(1,0),C(1,0)) = {W(0,0), C(0,0)}
    # all 4 subsets are capacity-feasible (big cap)
    assert frozenset({("W", 0, 0)}) in choices
    assert frozenset({("C", 0, 0)}) in choices
    assert len(choices) == 4


def test_eviction_choices_excludes_insufficient_subsets_when_tight():
    # cap holds exactly one cube working set (36864). Running a 2nd cube that needs a fresh
    # C tile forces evicting enough; subsets that DON'T free enough must NOT be offered.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=36, banks=32, dram_bw=32, word_bits=32)   # cap_bits = 36*32*32 = 36864
    st = es.SchedState(done=frozenset({(0, 0, 0)}),
                       resident=frozenset({("A", 0, 0), ("W", 0, 0), ("C", 0, 0)}),
                       last_compute=1.0)
    # cube (1,0,0) needs A(0,0)[resident], W(1,0)[+2048], C(1,0)[+32768]. After load w/o evict:
    # 2048+2048+32768 + 2048 + 32768 = 71680 > 36864 -> deficit = 34816.
    # evictable = {W(0,0)=2048, C(0,0)=32768}. Only subsets freeing >= 34816 qualify:
    #   {W(0,0),C(0,0)} frees 34816 (==deficit) -> OK; {C(0,0)} frees 32768 < deficit -> NO;
    #   {W(0,0)} frees 2048 -> NO; {} -> NO.
    choices = set(es.eviction_choices(st, (1, 0, 0), w, hw))
    assert choices == {frozenset({("W", 0, 0), ("C", 0, 0)})}


def test_eval_sched_rejects_non_permutation():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    cubes = es.all_cubes(w)
    with pytest.raises(ValueError):
        es.eval_sched(w, hw, cubes[:-1], [frozenset()] * (len(cubes) - 1))  # missing a cube
    with pytest.raises(ValueError):
        es.eval_sched(w, hw, cubes, [frozenset()] * (len(cubes) - 1))       # evictions length mismatch


def test_eval_sched_all_resident_no_eviction():
    # All tiles fit -> zero reloads/spills; read = first-touch A+W; final = all C written once.
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)   # MT=KT=NT=2, T=8
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)                    # cap_bits = 1048576
    cubes = es.all_cubes(w)
    r = es.eval_sched(w, hw, cubes, [frozenset()] * len(cubes))
    assert r["capacity_feasible"] is True
    assert r["dram_spill_bits"] == 0.0
    # 4 A tiles x 8192 + 4 W tiles x 8192 = 65536 (C zero-init free)
    assert r["dram_read_bits"] == 65536
    assert r["dram_final_bits"] == 4 * (s.TILE * s.TILE * s.FP32_BITS)  # 131072
    # energy = (read + spill) * coef. No reloads/spills, but the mandatory first-touch A/W loads
    # are REAL DRAM energy (final C writes are the excluded invariant constant) -> NOT zero.
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    assert r["energy"] == 65536 * coef


def test_eval_sched_total_dram_matches_closed_form_all_resident():
    # Parity (spec §8): for a fully-resident schedule, read+spill+final == closed-form total.
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    cubes = es.all_cubes(w)
    r = es.eval_sched(w, hw, cubes, [frozenset()] * len(cubes))
    total_dram = r["dram_read_bits"] + r["dram_spill_bits"] + r["dram_final_bits"]
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)   # single all-resident block
    assert total_dram == s.dram_bits(m, w)["total"]               # both 196608


def test_eval_sched_hand_built_spill_accounting():
    # Force one partial-C eviction + its reload via explicit evictions; check exact bits.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)  # MT=2,KT=2,NT=1, T=4
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)                   # cap large: everything fits
    order = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    evictions = [frozenset(), frozenset({("C", 0, 0)}), frozenset(), frozenset()]
    r = es.eval_sched(w, hw, order, evictions)
    c_sz = s.TILE * s.TILE * s.FP32_BITS    # 32768
    a_sz = w_sz = s.TILE * s.TILE * 2       # 2048
    # reads: every A loaded once (2), every W once (4), one C reload (the spilled C(0,0))
    assert r["dram_read_bits"] == 2 * a_sz + 4 * w_sz + c_sz        # 4096 + 8192 + 32768 = 45056
    assert r["dram_spill_bits"] == c_sz                            # exactly one partial spill
    assert r["dram_final_bits"] == 2 * c_sz                        # 2 C tiles, one final write each
    # variable energy uses the unified (dram + onchip) per-bit coefficient
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    assert r["energy"] == (45056 + c_sz) * coef                    # (read + spill) * coef


def test_eval_sched_stall0_flag_and_actual_cycle():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)   # infinite BW -> all transfers hidden
    cubes = es.all_cubes(w)
    r = es.eval_sched(w, hw, cubes, [frozenset()] * len(cubes))
    assert r["steady_stall"] < 1.0
    assert r["stall0_feasible"] is True
    # actual_cycle = compute_work(cpb) + sa_fill + fill + steady + drain + sa_drain
    assert r["actual_cycle"] == pytest.approx(
        float(s.compute_work(w, hw.cycles_per_bit)) + hw.sa_fill_cycles
        + r["fill"] + r["steady_stall"] + r["drain"] + hw.sa_drain_cycles)


def test_cycles_per_bit_scales_stall_hide_budget():
    # Invariant #5: cube_compute (the per-step hide budget) scales with cycles_per_bit, so a
    # fetch that stalls at cpb=1 becomes hidden at cpb=8. (M0's note: the hide budget was
    # unscaled; M1 scales BOTH sides consistently.) Big cap -> no reload, fetch is first-touch.
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    cubes = es.all_cubes(w)
    ev = [frozenset()] * len(cubes)
    slow = s.HW(bank_size=4096, banks=32, dram_bw=32, cycles_per_bit=1.0)   # cap holds everything
    fast = s.HW(bank_size=4096, banks=32, dram_bw=32, cycles_per_bit=8.0)
    r1 = es.eval_sched(w, slow, cubes, ev)
    r8 = es.eval_sched(w, fast, cubes, ev)
    assert r1["steady_stall"] > 0          # at cpb=1 the per-cube A+W fetch is NOT fully hidden
    assert r8["steady_stall"] == 0         # 8x compute hides every fetch (consistent scaling)
