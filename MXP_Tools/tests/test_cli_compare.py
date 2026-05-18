"""Tests for mxp_tools.cli — specifically the cmd_compare fail-gate.

Regression guard: an earlier version of cmd_compare printed stats but never
returned non-zero on mismatch. Shell wrappers (`sim/run_mixed_one.sh`,
`sim/run_integration_sweep.sh`) then printed PASS even when bit-exact compare
failed. These tests pin the gate behavior.
"""
import os
import tempfile

import numpy as np
import pytest

from mxp_tools import cli


def _write_bank_files(tmpdir, C, n_banks=1):
    """Dump (M, N) float32 matrix as $writememh-style bank files using the
    single-bank row-major layout (one file, sequential words)."""
    bank_path = os.path.join(tmpdir, "bank0.mem")
    with open(bank_path, "w", newline="\n") as f:
        flat = C.reshape(-1).view(np.uint32)
        for w in flat:
            f.write(f"{int(w):08x}\n")
    return [bank_path]


def _write_ref_npz(tmpdir, C_sw, C_fp32):
    path = os.path.join(tmpdir, "ref.npz")
    np.savez(path, C_sw=C_sw, C_fp32=C_fp32)
    return path


def test_compare_exits_0_on_match():
    with tempfile.TemporaryDirectory() as d:
        C = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        banks = _write_bank_files(d, C)
        ref = _write_ref_npz(d, C, C)
        with pytest.raises(SystemExit) as exc:
            cli.main([
                "compare", "--ref", ref, "--hw-banks", *banks,
                "--layout", "single",
            ])
        assert exc.value.code == 0, "bit-exact match must exit 0"


def test_compare_exits_1_on_mismatch():
    """The fail-gate. Without this, sweep scripts report false-positive PASS."""
    with tempfile.TemporaryDirectory() as d:
        C_hw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        C_sw = C_hw.copy()
        C_sw[0, 0] = 999.0  # introduce a single-element mismatch
        banks = _write_bank_files(d, C_hw)
        ref = _write_ref_npz(d, C_sw, C_sw)
        with pytest.raises(SystemExit) as exc:
            cli.main([
                "compare", "--ref", ref, "--hw-banks", *banks,
                "--layout", "single",
            ])
        assert exc.value.code == 1, (
            "bit-exact mismatch must exit 1 — otherwise sweep PASS is fake"
        )


def test_compare_exits_1_on_nan_in_hw():
    """NaN in HW dump (uninit SRAM word) must trip the gate, not pass silently."""
    with tempfile.TemporaryDirectory() as d:
        C_hw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        C_hw[0, 0] = np.nan
        C_sw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        banks = _write_bank_files(d, C_hw)
        ref = _write_ref_npz(d, C_sw, C_sw)
        with pytest.raises(SystemExit) as exc:
            cli.main([
                "compare", "--ref", ref, "--hw-banks", *banks,
                "--layout", "single",
            ])
        assert exc.value.code == 1, "NaN in HW must fail the gate"


def test_other_commands_exit_0():
    """gen/emit/ref/viz/rmw-gen don't gate; their return None must map to exit 0."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit) as exc:
            cli.main(["gen", "--out", d, "-M", "32", "-K", "32", "-N", "32"])
        assert exc.value.code == 0
