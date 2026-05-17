"""sim/gen_mixed.py 단위 검증 — W_PREC map 분포 + 재현성."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MXP_Tools"))

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


def test_quant_per_block_uniform_w_matches_mxp_tools():
    """모든 (M, K-block) 의 W_PREC 를 8 로 통일했을 때, 그 결과가
    mxp_tools.quant.quantize_matrix_mx(prec=8, block_axis=1) 와 일치."""
    from mxp_tools.quant import quantize_matrix_mx
    from gen_mixed import build_raw_data, quantize_weight_mixed

    W_fp, A_fp = build_raw_data(seed=0, M=128, K=128, N=128)
    prec_map_uniform = np.full((128, 4), 8, dtype=np.uint8)
    W_int, W_scale = quantize_weight_mixed(W_fp, prec_map_uniform)

    W_int_ref, W_scale_ref = quantize_matrix_mx(W_fp, prec=8, block_axis=1)
    np.testing.assert_array_equal(W_int, W_int_ref)
    np.testing.assert_array_equal(W_scale, W_scale_ref)


def test_quant_per_block_mixed_uses_per_block_prec():
    """row 0 의 K-block 0 만 W=2, 나머지 W=8 인 경우, 양자화된 INT 값이
    각 block 의 prec 따라 다른 범위를 갖는지 확인."""
    from gen_mixed import build_raw_data, quantize_weight_mixed
    from mxp_tools.const import MAX_INT
    W_fp, _ = build_raw_data(seed=0, M=128, K=128, N=128)
    pm = np.full((128, 4), 8, dtype=np.uint8)
    pm[0, 0] = 2
    W_int, _ = quantize_weight_mixed(W_fp, pm)
    # block (0, 0) 는 W=2 라 INT 값 범위 ≤ MAX_INT[2].
    block_00 = W_int[0, 0:32]
    assert np.max(np.abs(block_00.astype(np.int32))) <= MAX_INT[2]
