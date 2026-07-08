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
