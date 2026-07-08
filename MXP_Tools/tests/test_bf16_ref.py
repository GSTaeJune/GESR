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
