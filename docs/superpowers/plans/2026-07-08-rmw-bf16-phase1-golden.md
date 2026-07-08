# RMW BF16 — Phase 1 (Golden) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bit-exact BF16 accumulation path to `mxint_gemm_golden` (software model of the HW RMW accumulator), keeping the FP32 path byte-identical, so the golden is ready to validate the future BF16 RTL.

**Architecture:** Parameterize `mxint_gemm_golden(accum_dtype ∈ {"fp32"(default),"bf16"})`. The bf16 path mirrors the HW datapath: per K-block, `int→bf16` (RNE) → single integer exponent shift (`np.ldexp`, no premature underflow) → native bf16 add. FP32 truth (`C_fp32`) is untouched; `compare.py`'s gate stays bit-exact. This is Phase 1 of the spec `docs/superpowers/specs/2026-07-08-rmw-bf16-design.md`; the golden is **PROVISIONAL** until Phase 2a cross-checks HardFloat vs ml_dtypes.

**Tech Stack:** Python, numpy, `ml_dtypes.bfloat16` (new optional dependency), pytest.

**Scope:** Phase 1 only (MXP_Tools, no RTL). Phases 2a/2b get their own plans after this lands and is reviewed.

---

## File Structure

- `MXP_Tools/mxp_tools/gemm.py` — **modify**: add `accum_dtype` param + bf16 accumulation branch.
- `MXP_Tools/mxp_tools/cli.py` — **modify**: `ref --accum {fp32,bf16}`, suffixed output filename, informational SNR.
- `MXP_Tools/mxp_tools/compare.py` — **modify**: add `bf16_accuracy_stats` helper (reuses `_pair_stats`).
- `MXP_Tools/pyproject.toml` — **modify**: `ml_dtypes` optional extra.
- `MXP_Tools/tests/_bf16_ref.py` — **create**: independent pure-python bf16-RNE reference (test anchor).
- `MXP_Tools/tests/test_bf16_ref.py` — **create**: validate `ml_dtypes` rounding against the independent reference.
- `MXP_Tools/tests/test_gemm_bf16.py` — **create**: bf16 golden gate tests (determinism, subset, uniform==mixed, zero, invalid-dtype, rounding-topology, subnormal).
- `MXP_Tools/tests/test_gemm_fp32_frozen.py` — **create**: fp32 byte-identical frozen regression.
- `MXP_Tools/tests/fixtures/golden_fp32_frozen.npz` — **create**: captured pre-refactor fp32 output.
- `MXP_Tools/tests/test_cli_ref_bf16.py` — **create**: `ref --accum bf16` writes suffixed npz with `accum` meta.

All commands below run from `MXP_Tools/` unless noted (`cd MXP_Tools`).

---

## Task 1: Independent bf16-RNE reference (anti-tautology anchor)

Validates that `ml_dtypes.bfloat16` rounds fp32→bf16 as IEEE round-half-to-even, using a reference that does **not** call `ml_dtypes`. This is what makes the later bf16 tests a real gate rather than "ml_dtypes vs itself".

**Files:**
- Create: `MXP_Tools/tests/_bf16_ref.py`
- Create: `MXP_Tools/tests/test_bf16_ref.py`

- [ ] **Step 1: Write the independent reference**

Create `MXP_Tools/tests/_bf16_ref.py`:

```python
"""Independent fp32->bf16 rounding reference (pure bit manipulation, no ml_dtypes).

Used only by tests, to anchor ml_dtypes' bf16 rounding to the IEEE round-half-to-
even spec. bf16 shares fp32's 8-bit exponent+bias, so a bf16 value's bit pattern
is exactly the top 16 bits of the corresponding fp32 pattern; rounding is RNE on
the discarded low 16 bits. This also makes bf16 subnormals fp32-representable, so
the same top-16 + RNE rule handles the subnormal range.
"""
import numpy as np


def f32_to_bf16_bits_rne(x_f32):
    """Round one np.float32 scalar to bf16; return the 16-bit pattern as int."""
    b = int(np.float32(x_f32).view(np.uint32))
    exp = (b >> 23) & 0xFF
    mant = b & 0x7FFFFF
    top = b >> 16
    if exp == 0xFF:                       # inf / NaN
        if mant != 0 and (top & 0x7F) == 0:
            top |= 1                      # keep NaN a NaN (don't truncate to inf)
        return top & 0xFFFF
    lsb = (b >> 16) & 1
    round_bit = (b >> 15) & 1
    sticky = (b & 0x7FFF) != 0
    if round_bit and (sticky or lsb):     # round-half-to-even; carry may reach exp/inf
        top += 1
    return top & 0xFFFF


def ml_bf16_bits(x_f32, bfloat16):
    """The bit pattern ml_dtypes produces for the same fp32 scalar (for comparison)."""
    return int(np.float32(x_f32).astype(bfloat16).view(np.uint16))
```

- [ ] **Step 2: Write the failing test**

Create `MXP_Tools/tests/test_bf16_ref.py`:

```python
"""ml_dtypes bf16 rounding must equal an independent IEEE-RNE reference."""
import numpy as np
import pytest

from tests._bf16_ref import f32_to_bf16_bits_rne, ml_bf16_bits

bfloat16 = pytest.importorskip("ml_dtypes").bfloat16


def _check(vals):
    for v in vals:
        x = np.float32(v)
        assert f32_to_bf16_bits_rne(x) == ml_bf16_bits(x, bfloat16), f"mismatch at {x!r}"


def test_random_normals():
    rng = np.random.default_rng(0)
    _check((rng.standard_normal(200000).astype(np.float32) * rng.uniform(1e-3, 1e3, 200000)).tolist())


def test_directed_ties_and_edges():
    # exact bf16 values, their +/- half-ULP ties, zeros, and powers of two
    edges = [0.0, -0.0, 1.0, -1.0, 2.0, 1.5, 256.0, 257.0, 3.4e38, -3.4e38]
    # half-ULP ties around 1.0: bf16 ULP at 1.0 is 2^-7; tie is at 1 + 2^-8
    edges += [1.0 + 2.0**-8, 1.0 + 3.0 * 2.0**-8, 1.0 - 2.0**-9]
    _check(edges)


def test_subnormal_range():
    # bf16 subnormals live in [2^-133, 2^-126); include values that must flush or round up
    rng = np.random.default_rng(1)
    vals = (rng.uniform(2.0**-140, 2.0**-125, 100000)).tolist()
    vals += [2.0**-133, 2.0**-134, 1.5 * 2.0**-133, 2.0**-149]
    _check(vals)
```

- [ ] **Step 3: Run to verify it fails (import path not yet wired)**

Run: `cd MXP_Tools && python -m pytest tests/test_bf16_ref.py -q`
Expected: FAIL if `tests` is not importable as a package. If it fails with `ModuleNotFoundError: tests`, add an empty `tests/__init__.py` (Step 4). If ml_dtypes is absent, it SKIPS — install it first: `pip install ml_dtypes`.

- [ ] **Step 4: Make `tests` importable**

Create empty file `MXP_Tools/tests/__init__.py` (so `from tests._bf16_ref import ...` resolves). If the suite already collects fine without it, skip.

- [ ] **Step 5: Run to verify it passes**

Run: `cd MXP_Tools && python -m pytest tests/test_bf16_ref.py -q`
Expected: PASS (3 tests). If `test_subnormal_range` fails, do NOT weaken it — investigate whether the reference or ml_dtypes is wrong (this is the anchor; a real discrepancy must be understood, per spec §3.3).

- [ ] **Step 6: Commit**

```bash
cd MXP_Tools && git add tests/_bf16_ref.py tests/test_bf16_ref.py tests/__init__.py
git commit -m "test(rmw-bf16): independent bf16-RNE reference anchors ml_dtypes"
```

---

## Task 2: `accum_dtype` param + FP32 byte-identical (frozen regression)

Add the parameter with the fp32 path unchanged, pinned by a frozen-output test captured **before** the refactor.

**Files:**
- Modify: `MXP_Tools/mxp_tools/gemm.py`
- Create: `MXP_Tools/tests/fixtures/golden_fp32_frozen.npz`
- Create: `MXP_Tools/tests/test_gemm_fp32_frozen.py`

- [ ] **Step 1: Capture the current fp32 output (BEFORE editing gemm.py)**

Run this one-off from `MXP_Tools/` to snapshot the pre-refactor behavior:

```bash
cd MXP_Tools && python -c "
import os, numpy as np
from mxp_tools.const import INT8
from mxp_tools.gemm import mxint_gemm_golden
from mxp_tools.quant import quantize_matrix_mx
rng = np.random.default_rng(12345)
A = (rng.standard_normal((32,64)).astype(np.float32)*0.5)
B = (rng.standard_normal((64,32)).astype(np.float32)*0.5)
iA,sA = quantize_matrix_mx(A, INT8, block_axis=1)
iB,sB = quantize_matrix_mx(B, INT8, block_axis=0)
C = mxint_gemm_golden(iA,sA,INT8,iB,sB,INT8)
os.makedirs('tests/fixtures', exist_ok=True)
np.savez('tests/fixtures/golden_fp32_frozen.npz', iA=iA,sA=sA,iB=iB,sB=sB,C=C)
print('captured', C.shape, C.dtype)
"
```
Expected: `captured (32, 32) float32`.

- [ ] **Step 2: Write the failing test**

Create `MXP_Tools/tests/test_gemm_fp32_frozen.py`:

```python
"""FP32 golden must stay byte-identical across the accum_dtype refactor."""
import os
import numpy as np

from mxp_tools.const import INT8
from mxp_tools.gemm import mxint_gemm_golden

_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "golden_fp32_frozen.npz")


def test_fp32_path_bit_identical_to_frozen():
    d = np.load(_FIX)
    C = mxint_gemm_golden(d["iA"], d["sA"], INT8, d["iB"], d["sB"], INT8, accum_dtype="fp32")
    assert np.array_equal(C, d["C"]), "fp32 accum path drifted from frozen reference"


def test_fp32_is_default():
    d = np.load(_FIX)
    C_default = mxint_gemm_golden(d["iA"], d["sA"], INT8, d["iB"], d["sB"], INT8)
    assert np.array_equal(C_default, d["C"]), "default accum_dtype must be fp32"
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd MXP_Tools && python -m pytest tests/test_gemm_fp32_frozen.py -q`
Expected: FAIL — `mxint_gemm_golden() got an unexpected keyword argument 'accum_dtype'`.

- [ ] **Step 4: Add the param (fp32 path verbatim)**

In `MXP_Tools/mxp_tools/gemm.py`, change the signature:

```python
def mxint_gemm_golden(int_A, scale_A, prec_A, int_B, scale_B, prec_B, accum_dtype="fp32"):
```

Immediately after the two `dtype` checks at the top of the body (the `if int_A.dtype... / if scale_A.dtype...` block), add:

```python
    if accum_dtype not in ("fp32", "bf16"):
        raise ValueError(f"accum_dtype must be 'fp32' or 'bf16', got {accum_dtype!r}")
```

Leave the entire rest of the function (the fp32 loop and `return C`) unchanged. Update the module docstring's "Pure numpy" line to note ml_dtypes is used only for `accum_dtype='bf16'`.

- [ ] **Step 5: Run to verify it passes**

Run: `cd MXP_Tools && python -m pytest tests/test_gemm_fp32_frozen.py tests/test_gemm.py -q`
Expected: PASS (frozen tests + all existing `test_gemm.py`).

- [ ] **Step 6: Commit**

```bash
cd MXP_Tools && git add mxp_tools/gemm.py tests/test_gemm_fp32_frozen.py tests/fixtures/golden_fp32_frozen.npz
git commit -m "feat(rmw-bf16): add accum_dtype param; pin fp32 path byte-identical"
```

---

## Task 3: BF16 accumulation path

Add the bf16 branch mirroring the HW datapath, with lazy import and dtype guard. Then the core bf16 behavior tests.

**Files:**
- Modify: `MXP_Tools/mxp_tools/gemm.py`
- Create: `MXP_Tools/tests/test_gemm_bf16.py`

- [ ] **Step 1: Write the failing tests**

Create `MXP_Tools/tests/test_gemm_bf16.py`:

```python
"""BF16 accumulation path in mxint_gemm_golden."""
import numpy as np
import pytest

from mxp_tools.const import BLOCK_SIZE, INT2, INT4, INT8
from mxp_tools.gemm import mxint_gemm_golden
from mxp_tools.quant import quantize_matrix_mx

bfloat16 = pytest.importorskip("ml_dtypes").bfloat16


def _rand_case(seed, M=32, K=64, N=32, prec_a=INT8, prec_b=INT8, scale=0.5):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K)).astype(np.float32) * scale
    B = rng.standard_normal((K, N)).astype(np.float32) * scale
    int_A, scale_A = quantize_matrix_mx(A, prec_a, block_axis=1)
    int_B, scale_B = quantize_matrix_mx(B, prec_b, block_axis=0)
    return int_A, scale_A, prec_a, int_B, scale_B, prec_b


def test_bf16_deterministic():
    args = _rand_case(0)
    c1 = mxint_gemm_golden(*args, accum_dtype="bf16")
    c2 = mxint_gemm_golden(*args, accum_dtype="bf16")
    assert np.array_equal(c1, c2)


def test_bf16_output_is_bf16_representable():
    C = mxint_gemm_golden(*_rand_case(1), accum_dtype="bf16")
    assert np.array_equal(C, C.astype(bfloat16).astype(np.float32))


def test_bf16_uniform_prec_array_matches_scalar():
    rng = np.random.default_rng(2)
    M, K, N = 32, 64, 32
    A = rng.standard_normal((M, K)).astype(np.float32) * 0.5
    B = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    for prec in (INT2, INT4, INT8):
        int_A, scale_A = quantize_matrix_mx(A, prec, block_axis=1)
        int_B, scale_B = quantize_matrix_mx(B, INT8, block_axis=0)
        c_scalar = mxint_gemm_golden(int_A, scale_A, prec, int_B, scale_B, INT8, accum_dtype="bf16")
        prec_arr = np.full((M, K // BLOCK_SIZE), prec, dtype=np.uint8)
        c_array = mxint_gemm_golden(int_A, scale_A, prec_arr, int_B, scale_B, INT8, accum_dtype="bf16")
        assert np.array_equal(c_scalar, c_array), f"prec={prec}"


def test_bf16_zero_inputs_zero_output():
    int_A = np.zeros((32, 64), dtype=np.int8)
    scale_A = np.zeros((32, 2), dtype=np.uint8)
    int_B = np.zeros((64, 32), dtype=np.int8)
    scale_B = np.zeros((2, 32), dtype=np.uint8)
    C = mxint_gemm_golden(int_A, scale_A, INT8, int_B, scale_B, INT8, accum_dtype="bf16")
    assert np.all(C == 0.0)


def test_invalid_accum_dtype_raises():
    with pytest.raises(ValueError, match="accum_dtype"):
        mxint_gemm_golden(*_rand_case(3), accum_dtype="fp16")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd MXP_Tools && python -m pytest tests/test_gemm_bf16.py -q`
Expected: FAIL — the bf16 path is not implemented (results won't be bf16-representable / determinism aside, `test_bf16_output_is_bf16_representable` fails because fp32 accumulation isn't rounded to bf16). `test_invalid_accum_dtype_raises` already passes from Task 2.

- [ ] **Step 3: Implement the bf16 branch**

In `MXP_Tools/mxp_tools/gemm.py`, locate the point right **after** the prec_A scalar/array handling (where `impl_a_scalar` or `impl_a_mat` + `prec_a_is_array` are defined) and **before** `a_scale_fp = _e8m0_to_fp32(scale_A)`. Insert:

```python
    if accum_dtype == "bf16":
        try:
            from ml_dtypes import bfloat16
        except ImportError as e:
            raise ImportError(
                "accum_dtype='bf16' requires ml_dtypes. Install with: "
                "pip install ml_dtypes  (or `pip install -e .[bf16]`)."
            ) from e
        # Mirror the HW RMW datapath, per K-block in order 0..n-1 (see spec D5):
        #   int32 block sum (exact) -> int->bf16 (RNE, = INToRecFN)
        #   -> single integer exponent shift 2^E (exact, = recoded exp-add)
        #   -> bf16 add (RNE, = AddRecFN).
        # E = (e_a - 127) + (e_b - 127) - impl_a - impl_b, per (m, n).
        ea = scale_A.astype(np.int64)          # (M, n_blocks) E8M0 biased exponent
        eb = scale_B.astype(np.int64)          # (n_blocks, N)
        C = np.zeros((M, N), dtype=bfloat16)
        for blk in range(n_blocks):
            sl = slice(blk * BLOCK_SIZE, (blk + 1) * BLOCK_SIZE)
            block_int = int_A[:, sl].astype(np.int32) @ int_B[sl, :].astype(np.int32)
            if prec_a_is_array:
                impl_a_here = impl_a_mat[:, blk][:, np.newaxis]        # (M, 1)
            else:
                impl_a_here = impl_a_scalar                            # scalar
            E = ea[:, blk:blk + 1] + eb[blk:blk + 1, :] - 254 - impl_a_here - impl_b
            # block_int magnitude < 2^20, so float32 is exact and the cast to bf16 is RNE.
            r = block_int.astype(np.float32).astype(bfloat16)
            # ldexp scales by 2^E in float64 (no premature underflow), then one bf16 round.
            fp_a = np.ldexp(r.astype(np.float64), E).astype(bfloat16)
            C = C + fp_a                                              # native bf16 add
            assert C.dtype == bfloat16, "bf16 accumulator was promoted to a wider dtype"
        return C.astype(np.float32)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd MXP_Tools && python -m pytest tests/test_gemm_bf16.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the full suite (no regression)**

Run: `cd MXP_Tools && python -m pytest -q`
Expected: PASS (all prior + new; bf16 tests skip only if ml_dtypes missing).

- [ ] **Step 6: Commit**

```bash
cd MXP_Tools && git add mxp_tools/gemm.py tests/test_gemm_bf16.py
git commit -m "feat(rmw-bf16): bf16 accumulation path (ldexp dequant, native bf16 add, dtype guard)"
```

---

## Task 4: Correctness anchors — rounding topology + subnormal

Prove the golden rounds **per K-tile add** (not fp32-accumulate-then-cast-once), and that the subnormal-dequant path is exercised and self-consistent. These are the non-tautological correctness tests (spec §5.2 #2, #3).

**Files:**
- Modify: `MXP_Tools/tests/test_gemm_bf16.py`

- [ ] **Step 1: Add an independent per-add reference + the two tests**

Append to `MXP_Tools/tests/test_gemm_bf16.py`:

```python
from mxp_tools.const import IMPLICIT_SCALE_EXP


def _per_add_bf16_ref(int_A, scale_A, prec_A, int_B, scale_B, prec_B):
    """Independent explicit-loop bf16 accumulation (scalar prec_A only).

    Uses the explicit fp64-intermediate add form `(f64(C)+f64(fp_a))->bf16`;
    the golden uses native `C + fp_a`. These are provably identical (spec 3.3),
    so equality here also cross-checks native-vs-explicit add.
    """
    M, K = int_A.shape
    _, N = int_B.shape
    nb = K // BLOCK_SIZE
    impl = IMPLICIT_SCALE_EXP[prec_A] + IMPLICIT_SCALE_EXP[prec_B]
    C = np.zeros((M, N), dtype=bfloat16)
    for blk in range(nb):
        sl = slice(blk * BLOCK_SIZE, (blk + 1) * BLOCK_SIZE)
        bi = int_A[:, sl].astype(np.int32) @ int_B[sl, :].astype(np.int32)
        E = (scale_A[:, blk:blk + 1].astype(np.int64)
             + scale_B[blk:blk + 1, :].astype(np.int64) - 254 - impl)
        r = bi.astype(np.float32).astype(bfloat16)
        fp_a = np.ldexp(r.astype(np.float64), E).astype(bfloat16)
        C = (C.astype(np.float64) + fp_a.astype(np.float64)).astype(bfloat16)
    return C.astype(np.float32)


def _cast_once_ref(int_A, scale_A, prec_A, int_B, scale_B, prec_B):
    """Wrong-on-purpose: round each block's dequant to bf16 (like HW), but sum
    the partials in float64 and cast to bf16 only ONCE at the end."""
    M, K = int_A.shape
    _, N = int_B.shape
    nb = K // BLOCK_SIZE
    impl = IMPLICIT_SCALE_EXP[prec_A] + IMPLICIT_SCALE_EXP[prec_B]
    acc = np.zeros((M, N), dtype=np.float64)
    for blk in range(nb):
        sl = slice(blk * BLOCK_SIZE, (blk + 1) * BLOCK_SIZE)
        bi = int_A[:, sl].astype(np.int32) @ int_B[sl, :].astype(np.int32)
        E = (scale_A[:, blk:blk + 1].astype(np.int64)
             + scale_B[blk:blk + 1, :].astype(np.int64) - 254 - impl)
        r = bi.astype(np.float32).astype(bfloat16)
        acc += np.ldexp(r.astype(np.float64), E)
    return acc.astype(bfloat16).astype(np.float32)


def test_bf16_rounds_per_k_tile_not_once():
    # Deep-K (16 blocks) so swamping distinguishes per-add rounding from cast-once.
    args = _rand_case(7, M=32, K=32 * 16, N=32)
    golden = mxint_gemm_golden(*args, accum_dtype="bf16")
    assert np.array_equal(golden, _per_add_bf16_ref(*args)), \
        "golden must accumulate per-K-tile in bf16"
    cast_once = _cast_once_ref(*args)
    assert not np.array_equal(golden, cast_once), (
        "test is vacuous: this case does not distinguish per-add from cast-once; "
        "pick a seed/shape where they differ"
    )


def test_bf16_subnormal_dequant_self_consistent():
    # Tiny magnitudes push E8M0 exponents far below 127 so some block partials
    # land in the bf16-subnormal range (|value| < 2^-126), exercising ldexp's
    # subnormal rounding. Golden must match the independent per-add reference there.
    rng = np.random.default_rng(11)
    M, K, N = 32, 64, 32
    A = rng.standard_normal((M, K)).astype(np.float32) * (2.0 ** -65)
    B = rng.standard_normal((K, N)).astype(np.float32) * (2.0 ** -65)
    int_A, scale_A = quantize_matrix_mx(A, INT8, block_axis=1)
    int_B, scale_B = quantize_matrix_mx(B, INT8, block_axis=0)
    golden = mxint_gemm_golden(int_A, scale_A, INT8, int_B, scale_B, INT8, accum_dtype="bf16")
    ref = _per_add_bf16_ref(int_A, scale_A, INT8, int_B, scale_B, INT8)
    assert np.array_equal(golden, ref)
```

- [ ] **Step 2: Run to verify they pass**

Run: `cd MXP_Tools && python -m pytest tests/test_gemm_bf16.py -q`
Expected: PASS. If `test_bf16_rounds_per_k_tile_not_once` fails on the `not array_equal` line, the chosen case did not diverge — change the seed in `_rand_case(7, ...)` (try 8, 9, …) until per-add and cast-once differ, then confirm the first assertion still holds.

- [ ] **Step 3: Commit**

```bash
cd MXP_Tools && git add tests/test_gemm_bf16.py
git commit -m "test(rmw-bf16): per-K-tile rounding topology + subnormal-dequant anchors"
```

---

## Task 5: CLI `ref --accum` + informational SNR

Let `ref` emit a bf16 golden to a suffixed file (coexists with the fp32 ref) and print the bf16-vs-truth SNR with a loud warning on catastrophic accuracy. `compare.py` gate is untouched.

**Files:**
- Modify: `MXP_Tools/mxp_tools/compare.py`
- Modify: `MXP_Tools/mxp_tools/cli.py`
- Create: `MXP_Tools/tests/test_cli_ref_bf16.py`

- [ ] **Step 1: Add the accuracy helper to compare.py**

Append to `MXP_Tools/mxp_tools/compare.py`:

```python
def bf16_accuracy_stats(C_sw, C_fp32):
    """Informational: bf16 golden vs FP32 truth. Reuses _pair_stats.

    Returns the _pair_stats dict (snr_db, rmse, ...). NOT a gate — the compare
    pass/fail gate stays hw_sw-only. `catastrophic` flags negative-dB SNR so the
    fp32-fallback decision (spec D4) is actionable rather than buried.
    """
    stats = _pair_stats(C_sw.astype(np.float32), C_fp32.astype(np.float32))
    stats["catastrophic"] = bool(np.isfinite(stats["snr_db"]) and stats["snr_db"] < 0.0)
    return stats
```

- [ ] **Step 2: Write the failing CLI test**

Create `MXP_Tools/tests/test_cli_ref_bf16.py`:

```python
"""`ref --accum bf16` writes a suffixed npz carrying the accum tag."""
import os
import numpy as np
import pytest

pytest.importorskip("ml_dtypes")
from mxp_tools.cli import main


def _run(argv):
    with pytest.raises(SystemExit) as ei:
        main(argv)
    assert ei.value.code in (0, None), f"{argv} exited {ei.value.code}"


def test_ref_bf16_writes_suffixed_npz(tmp_path):
    out = str(tmp_path)
    _run(["gen", "--out", out, "-M", "32", "-K", "64", "-N", "32", "--seed", "0"])
    _run(["emit", "--out", out, "--prec", "8"])
    _run(["ref", "--out", out, "--prec-a", "8", "--prec-b", "8", "--accum", "bf16"])

    bf16_path = os.path.join(out, "sw_ref", "C_sw_mxint8_mxint8_bf16.npz")
    fp32_path = os.path.join(out, "sw_ref", "C_sw_mxint8_mxint8.npz")
    assert os.path.exists(bf16_path), "bf16 ref must be suffixed"
    d = np.load(bf16_path)
    assert str(d["accum"]) == "bf16"
    assert d["C_sw"].shape == (32, 32)
    # default fp32 ref must NOT be created by a bf16-only ref call
    assert not os.path.exists(fp32_path)


def test_ref_fp32_default_unsuffixed(tmp_path):
    out = str(tmp_path)
    _run(["gen", "--out", out, "-M", "32", "-K", "64", "-N", "32", "--seed", "0"])
    _run(["emit", "--out", out, "--prec", "8"])
    _run(["ref", "--out", out, "--prec-a", "8", "--prec-b", "8"])   # default accum
    assert os.path.exists(os.path.join(out, "sw_ref", "C_sw_mxint8_mxint8.npz"))
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd MXP_Tools && python -m pytest tests/test_cli_ref_bf16.py -q`
Expected: FAIL — `ref` has no `--accum` flag (argparse error / `SystemExit(2)`).

- [ ] **Step 4: Wire `--accum` into cli.py**

In `MXP_Tools/mxp_tools/cli.py`, import the helper near the top (it already does `from . import compare as cmp_mod`). In `cmd_ref`, replace the `C_sw_pad = gemm.mxint_gemm_golden(...)` call and the save block with:

```python
            C_sw_pad = gemm.mxint_gemm_golden(
                qa["a_int"], qa["a_scale"], pa,
                qb["b_int"], qb["b_scale"], pb,
                accum_dtype=args.accum,
            )
            C_sw = C_sw_pad[:M, :N]
            suffix = "" if args.accum == "fp32" else f"_{args.accum}"
            path = os.path.join(out_ref, f"C_sw_{PREC_NAMES[pa]}_{PREC_NAMES[pb]}{suffix}.npz")
            np.savez(path, C_sw=C_sw, C_fp32=C_fp32, prec_a=pa, prec_b=pb, accum=args.accum)
            print(f"ref: {PREC_NAMES[pa]} x {PREC_NAMES[pb]} [{args.accum}]  ->  {path}")
            if args.accum == "bf16":
                s = cmp_mod.bf16_accuracy_stats(C_sw, C_fp32)
                print(f"  bf16 vs fp32-truth: SNR={s['snr_db']:.2f} dB  rmse={s['rmse']:.3e}")
                if s["catastrophic"]:
                    print("  WARNING: bf16 SNR < 0 dB - accumulation may be unusable; "
                          "consider the fp32 fallback (spec D4).")
```

In the `ref` argparse block, add the flag:

```python
    g.add_argument("--accum", choices=("fp32", "bf16"), default="fp32",
                   help="golden accumulator precision (default fp32)")
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd MXP_Tools && python -m pytest tests/test_cli_ref_bf16.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
cd MXP_Tools && git add mxp_tools/compare.py mxp_tools/cli.py tests/test_cli_ref_bf16.py
git commit -m "feat(rmw-bf16): ref --accum bf16 (suffixed npz) + informational SNR/warning"
```

---

## Task 6: Optional dependency + docs

Declare `ml_dtypes` as an optional extra so the numpy-only default path and users are unaffected; the bf16 branch already lazy-imports with a clear error (Task 3).

**Files:**
- Modify: `MXP_Tools/pyproject.toml`

- [ ] **Step 1: Add the optional extra**

In `MXP_Tools/pyproject.toml`, under `[project.optional-dependencies]`, add a `bf16` extra and include it in `dev`:

```toml
[project.optional-dependencies]
viz = ["matplotlib>=3.5"]
bf16 = ["ml_dtypes>=0.4"]
dev = ["pytest>=7", "matplotlib>=3.5", "ml_dtypes>=0.4"]
```

- [ ] **Step 2: Verify the extra installs and the suite is green**

Run: `cd MXP_Tools && pip install -e .[bf16] && python -m pytest -q`
Expected: install succeeds; PASS (all tests, no skips for bf16).

- [ ] **Step 3: Verify graceful behavior without ml_dtypes (optional check)**

If feasible in a throwaway env without ml_dtypes: `python -c "from mxp_tools.gemm import mxint_gemm_golden"` imports fine (numpy-only), and calling with `accum_dtype='bf16'` raises the helpful `ImportError` from Task 3. Do not remove ml_dtypes from the working env to test this; inspection of the Task-3 lazy-import block is sufficient.

- [ ] **Step 4: Commit**

```bash
cd MXP_Tools && git add pyproject.toml
git commit -m "build(rmw-bf16): ml_dtypes as optional [bf16] extra"
```

---

## Phase 1 completion checklist

- [ ] `cd MXP_Tools && python -m pytest -q` — all green (bf16 tests run, not skipped).
- [ ] `python buffer_sweep/buffer_sweep.py --selftest` and the fp32 9-mode integration sweep (`bash sim/run_integration_sweep.sh`) still pass — **fp32 path unregressed** (human-run; the sweep is ~10 min).
- [ ] Per CLAUDE.md rule 2: run `/superpowers:requesting-code-review` and `/review`; iterate to convergence.
- [ ] Note in the spec/kickoff that the golden is **PROVISIONAL** until Phase 2a.

---

## Self-review notes (author)

- **Spec coverage:** D1 (parameterize) → Task 2/3; D2 (ml_dtypes optional+lazy) → Task 3/6; single-exponent-shift dequant (§3.2) → Task 3; dtype guard → Task 3; §5.2 gate: independent ref #1 → Task 1, topology #2 → Task 4, subnormal #3 → Task 4, uniform==mixed #4 → Task 3, bf16-subset #5 → Task 3, fp32 frozen #6 → Task 2, reference-semantics #7 → covered by Task 1 (directed ties are hand-derived spec values), informational SNR #8 → Task 5. `ref --accum` → Task 5. D5 loop-order: the golden loops 0..n-1 by construction; the *runtime guard* against a different RTL feed order is a Phase-2 concern (documented in the Task-3 comment).
- **Deferred to later plans (not Phase 1):** HardFloat re-vendor, int_to_bf16/bf16_adder + cross-check TBs (Phase 2a); RTL width narrowing, hwio bf16 reader, integration sweep (Phase 2b).
