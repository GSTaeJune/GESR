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


def test_golden_matches_mxp_tools_when_uniform():
    """모든 W_PREC=8, A_PREC=8 일 때 mxp_tools.gemm.mxint_gemm_golden 과 일치."""
    from mxp_tools.gemm import mxint_gemm_golden
    from gen_mixed import build_raw_data, quantize_weight_mixed, \
        quantize_activation_uniform, compute_golden_mixed

    W_fp, A_fp = build_raw_data(seed=0)
    pm_uniform = np.full((128, 4), 8, dtype=np.uint8)
    W_int, W_scale = quantize_weight_mixed(W_fp, pm_uniform)
    A_int, A_scale = quantize_activation_uniform(A_fp, prec=8)

    C_ours = compute_golden_mixed(W_int, W_scale, A_int, A_scale, pm_uniform)
    # mxint_gemm_golden(int_A, scale_A, prec_A, int_B, scale_B, prec_B)
    # prec_A = WEIGHT prec (우리의 W_int), prec_B = ACTIVATION prec (우리의 A_int).
    C_ref = mxint_gemm_golden(W_int, W_scale, 8, A_int, A_scale, 8)
    # FP32 bit-exact 일치 요구.
    assert C_ours.dtype == np.float32
    np.testing.assert_array_equal(
        C_ours.view(np.uint32), C_ref.view(np.uint32),
        err_msg="mixed-w/uniform-prec 의 골든이 mxp_tools 와 비트단위 일치하지 않음"
    )


def test_golden_shape_and_dtype():
    from gen_mixed import build_raw_data, quantize_weight_mixed, \
        quantize_activation_uniform, build_w_prec_map, compute_golden_mixed
    W_fp, A_fp = build_raw_data(seed=0)
    pm = build_w_prec_map(seed=0)
    W_int, W_scale = quantize_weight_mixed(W_fp, pm)
    A_int, A_scale = quantize_activation_uniform(A_fp, prec=4)
    C = compute_golden_mixed(W_int, W_scale, A_int, A_scale, pm)
    assert C.shape == (128, 128)
    assert C.dtype == np.float32


def test_emit_hex_files_exist_with_expected_line_counts(tmp_path):
    from gen_mixed import build_raw_data, build_w_prec_map, \
        quantize_weight_mixed, quantize_activation_uniform, emit_hex
    W_fp, A_fp = build_raw_data(seed=0)
    pm = build_w_prec_map(seed=0)
    W_int, W_scale = quantize_weight_mixed(W_fp, pm)
    A_int, A_scale = quantize_activation_uniform(A_fp, prec=8)
    out = tmp_path / "hw_input"
    emit_hex(W_int, W_scale, A_int, A_scale, pm, A_PREC=8, out_dir=out)

    expected = {
        "a_input_BS_mixed.hex": 4096,     # K_T(4) * M_T(4) * TILE(32) * W=8 padding
        "b_input_mixed_A8.hex": 512,      # N_T(4) * K_T(4) * TILE(32)
        "a_scale_mixed.hex": 512,         # K_T*M_T*TILE
        "b_scale_mixed.hex": 512,         # N_T*K_T*TILE
        "w_prec_per_block.hex": 128,      # M rows
    }
    for name, n_lines in expected.items():
        p = out / name
        assert p.exists(), f"missing {name}"
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        assert len(lines) == n_lines, f"{name}: expected {n_lines} lines, got {len(lines)}"


def test_w_prec_per_block_encoding():
    """w_prec_per_block.hex 의 한 line = m row 의 4 K-block W_CTRL_CODE packed.
    bit[2k+1:2k] = W_CTRL_CODE for (m, k_t=k).  W=8→11, W=4→10, W=2→01."""
    from gen_mixed import emit_hex_w_prec_only
    pm = np.array([[8, 4, 2, 8]], dtype=np.uint8)  # 1 row × 4 K-block
    line = emit_hex_w_prec_only(pm)  # returns str
    # bit[1:0] = W=8 → 11, bit[3:2] = W=4 → 10, bit[5:4] = W=2 → 01, bit[7:6] = W=8 → 11
    # → 0b11_01_10_11 = 0xDB
    assert line.strip() == "db"
