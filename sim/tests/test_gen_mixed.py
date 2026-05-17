"""sim/gen_mixed.py 단위 검증 — W_PREC map 분포 + 재현성."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from gen_mixed import build_w_prec_map


def test_w_prec_map_shape_and_values():
    pm = build_w_prec_map(seed=0, M=128, K_T=4)
    assert pm.shape == (128, 4)
    assert pm.dtype == np.uint8
    assert set(np.unique(pm).tolist()) <= {2, 4, 8}


def test_w_prec_map_reproducible():
    pm1 = build_w_prec_map(seed=0)
    pm2 = build_w_prec_map(seed=0)
    np.testing.assert_array_equal(pm1, pm2)


def test_w_prec_map_uniform_distribution_within_5pct():
    pm = build_w_prec_map(seed=0)
    total = pm.size  # 512
    counts = {v: int((pm == v).sum()) for v in (2, 4, 8)}
    # 균등 1/3 ≈ 171. ±5% (8 개) 마진.
    for v, c in counts.items():
        assert abs(c - total / 3) < total * 0.05, f"W={v}: count={c}"


def test_visualize_writes_png_and_txt(tmp_path):
    from gen_mixed import build_w_prec_map, visualize_prec_map
    pm = build_w_prec_map(seed=0)
    visualize_prec_map(pm, A_PREC=8, out_dir=tmp_path)
    assert (tmp_path / "precision_map.png").exists()
    txt = (tmp_path / "precision_map.txt").read_text()
    # ASCII map 의 한 row 가 K_T(=4) 글자.
    lines = [ln for ln in txt.splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 128
    for ln in lines:
        assert len(ln) == 4
        assert set(ln) <= {"2", "4", "8"}
