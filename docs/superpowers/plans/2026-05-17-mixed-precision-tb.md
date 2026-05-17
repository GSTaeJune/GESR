# Mixed-Precision (per-block W_PREC) 검증 TB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** weight 의 (M-row, K-block) 단위 random W_PREC ∈ {2,4,8} mix 가 RTL 변경 없이 bit-exact 동작함을 3 시나리오 (A_PREC=2/4/8) 로 검증한다.

**Architecture:** Python `sim/gen_mixed.py` 가 random W_PREC map 생성 + per-block MX 양자화 + FP32 골든 + hex emit + precision map visualize. 신규 `tb/gemm_sram_top_mixed_tb.v` 가 기존 9-mode TB 의 INIT/LOAD/CONFIG/DRIVE/DRAIN/DUMP 구조 답습하되 inner-loop 만 mixed-aware. `sim/run_mixed_sweep.sh` 가 3 시나리오 직렬 + bit-exact gate.

**Tech Stack:** Python (numpy + matplotlib + `mxp_tools.quant` import), Verilog-2001 TB, XSim batch, MXP_Tools `compare` subcommand 재사용.

**Spec:** `docs/superpowers/specs/2026-05-17-mixed-precision-tb-design.md`

---

## File Structure

| 파일 | 책임 |
|---|---|
| `sim/gen_mixed.py` (신규) | random W_PREC map + per-block MX quant + FP32 golden + hex emit + visualize. CLI entry. |
| `sim/tests/test_gen_mixed.py` (신규) | gen_mixed.py 의 단위 검증 (pytest). |
| `tb/gemm_sram_top_mixed_tb.v` (신규) | mixed-aware integration TB. 기존 `tb/gemm_sram_top_tb.v` 의 구조 답습. 한글 헤더 + 검증 목적/시나리오/동작 의도 명시. |
| `sim/run_mixed_one.sh` (신규) | A_PREC 1 시나리오 end-to-end (gen_mixed → xsim → compare). |
| `sim/run_mixed_sweep.sh` (신규) | A=2/4/8 직렬 + `ALL 3 MIXED SCENARIOS PASSED` gate. |
| `.gitignore` (수정) | `work/mixed_A*/` 추가. |

---

## Task 1: `sim/gen_mixed.py` — W_PREC map 생성기

**Files:**
- Create: `sim/gen_mixed.py`
- Create: `sim/tests/__init__.py` (빈 파일)
- Create: `sim/tests/test_gen_mixed.py`

- [ ] **Step 1: 실패 테스트 작성** (`sim/tests/test_gen_mixed.py`)

```python
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
```

- [ ] **Step 2: 테스트 실행, 실패 확인**

Run: `python -m pytest sim/tests/test_gen_mixed.py -v`
Expected: ImportError (`gen_mixed` 없음).

- [ ] **Step 3: `sim/gen_mixed.py` 의 `build_w_prec_map` 구현**

```python
"""sim/gen_mixed.py — mixed-precision TB 용 데이터/골든/hex emit 생성기.

검증 목적:
  weight 의 (M-row, K-block) 단위 random W_PREC ∈ {2,4,8} mix 가 RTL 변경
  없이 bit-exact 동작함을 검증하기 위해 TB 입력 hex + FP32 골든 + precision
  map visualize 까지 한 파일에서 산출.

사용:
  python sim/gen_mixed.py --A 8 --seed 0 --out work/mixed_A8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_w_prec_map(seed: int = 0, M: int = 128, K_T: int = 4) -> np.ndarray:
    """(M, K_T) shape 의 W_PREC array ∈ {2,4,8} 균등 random.

    seed 고정 시 재현 가능. dtype=uint8.
    """
    rng = np.random.default_rng(seed)
    choices = np.array([2, 4, 8], dtype=np.uint8)
    return choices[rng.integers(0, 3, size=(M, K_T), dtype=np.int64)]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest sim/tests/test_gen_mixed.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sim/gen_mixed.py sim/tests/__init__.py sim/tests/test_gen_mixed.py
git commit -m "feat(gen_mixed): random W_PREC map generator with seed reproducibility"
```

---

## Task 2: `sim/gen_mixed.py` — precision map visualize (PNG + TXT)

**Files:**
- Modify: `sim/gen_mixed.py`
- Modify: `sim/tests/test_gen_mixed.py`

- [ ] **Step 1: 실패 테스트 추가** (`sim/tests/test_gen_mixed.py` 끝에 append)

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest sim/tests/test_gen_mixed.py::test_visualize_writes_png_and_txt -v`
Expected: ImportError on `visualize_prec_map`.

- [ ] **Step 3: `visualize_prec_map` 구현** (`sim/gen_mixed.py` 끝에 append)

```python
def visualize_prec_map(prec_map: np.ndarray, A_PREC: int, out_dir: Path) -> None:
    """precision_map.png + .txt 산출.

    PNG 는 matplotlib 사용 (없으면 PNG 생략, TXT 만). TXT 는 ASCII heatmap.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TXT 먼저 — matplotlib 의존성 없음.
    txt_path = out_dir / "precision_map.txt"
    with txt_path.open("w") as f:
        f.write(f"# precision_map  A_PREC={A_PREC}  seed=0\n")
        f.write(f"# rows = M=0..127, cols = K-block 0..3\n")
        f.write(f"# values: 2 / 4 / 8 (W_PREC per block)\n")
        for m in range(prec_map.shape[0]):
            f.write("".join(str(int(v)) for v in prec_map[m]) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib 없으면 PNG 생략.

    fig, ax = plt.subplots(figsize=(3, 12))
    # {2,4,8} → {0,1,2} 로 mapping 후 색 3개.
    lookup = {2: 0, 4: 1, 8: 2}
    img = np.vectorize(lookup.get)(prec_map)
    cmap = matplotlib.colors.ListedColormap(["#d62728", "#ff7f0e", "#1f77b4"])
    ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_xlabel("K-block (0..3)")
    ax.set_ylabel("M-row (0..127)")
    ax.set_title(f"W_PREC map (A_PREC={A_PREC}, seed=0)")
    ax.set_xticks(range(4))
    cbar = fig.colorbar(ax.images[0], ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["W=2", "W=4", "W=8"])
    fig.tight_layout()
    fig.savefig(out_dir / "precision_map.png", dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest sim/tests/test_gen_mixed.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sim/gen_mixed.py sim/tests/test_gen_mixed.py
git commit -m "feat(gen_mixed): precision map visualize (PNG + TXT)"
```

---

## Task 3: `sim/gen_mixed.py` — raw FP data + per-block MX 양자화

**Files:**
- Modify: `sim/gen_mixed.py`
- Modify: `sim/tests/test_gen_mixed.py`

- [ ] **Step 1: 실패 테스트 추가**

```python
def test_quant_per_block_uniform_w_matches_mxp_tools():
    """모든 (M, K-block) 의 W_PREC 를 8 로 통일했을 때, 그 결과가
    mxp_tools.quant.quantize_matrix_mx(prec=8, block_axis=1) 와 일치."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MXP_Tools"))
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
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest sim/tests/test_gen_mixed.py::test_quant_per_block_uniform_w_matches_mxp_tools -v`
Expected: ImportError on `build_raw_data`.

- [ ] **Step 3: 구현** (`sim/gen_mixed.py` 에 append)

```python
import sys

# MXP_Tools 를 import path 에 추가 (sim/gen_mixed.py 기준 ../MXP_Tools/)
_MXP_TOOLS = Path(__file__).resolve().parent.parent / "MXP_Tools"
if str(_MXP_TOOLS) not in sys.path:
    sys.path.insert(0, str(_MXP_TOOLS))

from mxp_tools.quant import quantize_block_mx, quantize_matrix_mx  # noqa: E402
from mxp_tools.const import BLOCK_SIZE  # noqa: E402


def build_raw_data(seed: int, M: int = 128, K: int = 128, N: int = 128):
    """FP32 raw weight [M, K] + activation [K, N]. seed 분리해서 W/A 독립.

    값 범위는 ~N(0, 1) — MX 양자화가 정상 동작하도록.
    """
    rng_w = np.random.default_rng(seed * 2 + 1)
    rng_a = np.random.default_rng(seed * 2 + 2)
    W = rng_w.standard_normal((M, K)).astype(np.float32)
    A = rng_a.standard_normal((K, N)).astype(np.float32)
    return W, A


def quantize_weight_mixed(W_fp32: np.ndarray, prec_map: np.ndarray):
    """weight [M, K] 를 (M, K-block) 단위 W_PREC 로 양자화.

    Returns:
      W_int   : int8, shape (M, K)
      W_scale : uint8, shape (M, K_T)
    """
    M, K = W_fp32.shape
    K_T = K // BLOCK_SIZE
    assert prec_map.shape == (M, K_T), f"prec_map shape mismatch: {prec_map.shape}"
    W_int = np.zeros((M, K), dtype=np.int8)
    W_scale = np.zeros((M, K_T), dtype=np.uint8)
    for m in range(M):
        for kt in range(K_T):
            sl = slice(kt * BLOCK_SIZE, (kt + 1) * BLOCK_SIZE)
            prec = int(prec_map[m, kt])
            q, s = quantize_block_mx(W_fp32[m, sl], prec)
            W_int[m, sl] = q
            W_scale[m, kt] = s
    return W_int, W_scale


def quantize_activation_uniform(A_fp32: np.ndarray, prec: int):
    """activation [K, N] 를 K-axis (block_axis=0) 단위로 prec 로 양자화."""
    return quantize_matrix_mx(A_fp32, prec=prec, block_axis=0)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest sim/tests/test_gen_mixed.py -v`
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sim/gen_mixed.py sim/tests/test_gen_mixed.py
git commit -m "feat(gen_mixed): per-block MX quant for weight, uniform for activation"
```

---

## Task 4: `sim/gen_mixed.py` — FP32 GEMM 골든 (K-block 단위 누적)

**Files:**
- Modify: `sim/gen_mixed.py`
- Modify: `sim/tests/test_gen_mixed.py`

- [ ] **Step 1: 실패 테스트 추가**

```python
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
    C_ref = mxint_gemm_golden(W_int, W_scale, A_int, A_scale, prec_A=8, prec_B=8)
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
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest sim/tests/test_gen_mixed.py::test_golden_matches_mxp_tools_when_uniform -v`
Expected: ImportError on `compute_golden_mixed`.

- [ ] **Step 3: 먼저 mxp_tools.gemm.mxint_gemm_golden 의 시그니처 확인**

Run: `python -c "from mxp_tools.gemm import mxint_gemm_golden; help(mxint_gemm_golden)"`

산출 시그니처에 맞춰 다음 step 의 `compute_golden_mixed` 의 호출 인자/loop 구조를 보정. 본 plan 의 step 3 는 시그니처가 `mxint_gemm_golden(A_int, A_scale, B_int, B_scale, prec_A, prec_B)` 형태로 W/A 양자화 결과를 받는다고 가정 (mxp_tools 의 이름 컨벤션은 prec_A=WEIGHT). 시그니처가 다르면 step 4 의 구현을 그에 맞게 조정.

- [ ] **Step 4: `compute_golden_mixed` 구현**

```python
def compute_golden_mixed(W_int, W_scale, A_int, A_scale, prec_map) -> np.ndarray:
    """FP32 GEMM golden — K-block 단위 누적, RMW 4 회 누적과 동일.

    각 (m, n) 의 출력 = Σ over k_t ∈ [0, K_T): block_contrib(m, n, k_t).
    block_contrib = FP32 ( (Σ k W_int[m, 32 k_t + k] × A_int[32 k_t + k, n])
                            × 2^(W_scale[m, k_t] + A_scale[k_t, n] - 254) ).

    RMW 의 INT32 dot product + (act_scale + weight_scale - 127) → FP32 dequant
    + FP32 누적 과 정확히 동일한 순서로 계산. 실제 HW 의 FP32 add 순서
    (k_t 0→1→2→3) 를 numpy 의 단순 sum 으로 재현.
    """
    M, K = W_int.shape
    _, N = A_int.shape
    K_T = K // BLOCK_SIZE
    assert prec_map.shape == (M, K_T)

    C = np.zeros((M, N), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            acc = np.float32(0.0)
            for kt in range(K_T):
                sl = slice(kt * BLOCK_SIZE, (kt + 1) * BLOCK_SIZE)
                # int32 dot product (no overflow at K=32 with INT8 max).
                dot = int(np.dot(
                    W_int[m, sl].astype(np.int32),
                    A_int[sl, n].astype(np.int32)
                ))
                # combined scale = w_scale + a_scale - 127 (9-bit signed).
                combined_exp = int(W_scale[m, kt]) + int(A_scale[kt, n]) - 127
                # FP32 dequant: dot × 2^(combined_exp - 127).
                # (mxp_tools 컨벤션과 동일 — IMPLICIT_SCALE_EXP 가 INT→FP 의
                # 분모로 들어가는데, mxint_gemm_golden 가 그 처리를 함.
                # 여기선 acc 의 단순 FP32 add 만 재현; INT→FP 변환식의
                # 정확한 형태는 mxp_tools 참조해 맞춤.)
                dequant = np.float32(dot) * np.float32(2.0) ** np.float32(combined_exp - 127)
                acc = np.float32(acc + dequant)
            C[m, n] = acc
    return C
```

**주의:** Step 1 의 첫 테스트가 비트단위 일치를 요구하므로, 위 구현이 mxp_tools 의 정확한 INT→FP 변환식과 누적 순서를 따르지 않으면 fail. fail 시 `MXP_Tools/mxp_tools/gemm.py::mxint_gemm_golden` 의 본문을 직접 읽어 누적 순서 + 정밀도 변환식을 맞춤. ULP 오차 허용 불가.

- [ ] **Step 5: 테스트 통과 확인**

Run: `python -m pytest sim/tests/test_gen_mixed.py -v`
Expected: 8 tests PASS. 만약 `test_golden_matches_mxp_tools_when_uniform` 가 fail 이면 `MXP_Tools/mxp_tools/gemm.py` 읽어서 dequant 식/누적 순서 맞춤.

- [ ] **Step 6: Commit**

```bash
git add sim/gen_mixed.py sim/tests/test_gen_mixed.py
git commit -m "feat(gen_mixed): FP32 golden GEMM with per-block W_PREC scale handling"
```

---

## Task 5: `sim/gen_mixed.py` — hex emit (5 파일)

**Files:**
- Modify: `sim/gen_mixed.py`
- Modify: `sim/tests/test_gen_mixed.py`

- [ ] **Step 1: 실패 테스트 추가**

```python
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
    from gen_mixed import build_w_prec_map, emit_hex_w_prec_only
    pm = np.array([[8, 4, 2, 8]], dtype=np.uint8)  # 1 row × 4 K-block
    line = emit_hex_w_prec_only(pm)  # returns str
    # bit[1:0] = W=8 → 11, bit[3:2] = W=4 → 10, bit[5:4] = W=2 → 01, bit[7:6] = W=8 → 11
    # → 0b11_01_10_11 = 0xDB
    assert line.strip() == "db"
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest sim/tests/test_gen_mixed.py::test_emit_hex_files_exist_with_expected_line_counts -v`
Expected: ImportError on `emit_hex`.

- [ ] **Step 3: 구현**

```python
# W → W_CTRL_CODE lookup (precision_modes_protocol.md §1 과 일치)
_W_CTRL_LUT = {8: 0b11, 4: 0b10, 2: 0b01}


def emit_hex_w_prec_only(prec_map: np.ndarray) -> str:
    """prec_map (M, K_T) → 'hh\n' 행 누적 문자열. 단위 검증용."""
    out = []
    for m in range(prec_map.shape[0]):
        byte = 0
        for kt in range(prec_map.shape[1]):
            byte |= _W_CTRL_LUT[int(prec_map[m, kt])] << (2 * kt)
        out.append(f"{byte:02x}\n")
    return "".join(out)


def _emit_a_input_bs(W_int: np.ndarray, out_path: Path, K_T=4, M_T=4, TILE=32, W_PAD=8):
    """weight bit-serial hex. W=8 padding 컨벤션 — 모든 entry 8 cycle/row.

    한 워드 (32-bit) = TB cycle 한 시점의 32 K-row 의 1 bit.
    파일 인덱싱: a_bs[k_t * M_T * TILE * W_PAD + m_t * TILE * W_PAD + o]
    o ∈ [0, TILE * W_PAD), o = m_in * W_PAD + bit_pos (bit_pos = MSB..LSB).
    """
    lines = []
    for k_t in range(K_T):
        for m_t in range(M_T):
            for m_in in range(TILE):
                m = m_t * TILE + m_in
                # 한 (m, k_t) 의 32 K-element weight (INT8 signed).
                w_block = W_int[m, k_t * TILE:(k_t + 1) * TILE].astype(np.int32)
                # 2's complement → unsigned 8-bit 으로 (bit-serial 추출용).
                w_unsigned = (w_block & 0xFF).astype(np.uint8)
                for bit_pos in range(W_PAD - 1, -1, -1):
                    word = 0
                    for k in range(TILE):
                        if (w_unsigned[k] >> bit_pos) & 1:
                            word |= (1 << k)
                    lines.append(f"{word:08x}\n")
    out_path.write_text("".join(lines))


def _emit_b_input(A_int: np.ndarray, out_path: Path, A_PREC: int,
                  N_T=4, K_T=4, TILE=32):
    """activation 256-bit 워드 hex.

    한 워드 = 한 (n_t, k_t, col_target) 의 32 K-row × 8-bit activation.
    파일 인덱싱: b_bp[n_t * K_T * TILE + k_t * TILE + col_target]
    A_PREC=4: 8-bit 안에 INT4 두 개 packed (TB 의 build_in_b 가 풀어냄).
    A_PREC=2: 8-bit 안에 INT2 네 개 packed.

    구현 단순화: emit 단계는 A_PREC 무관 INT8 raw 8-bit 그대로. TB 의 V3 pack
    함수가 mode 별로 4-bit/2-bit 추출. (기존 9-mode sweep 의 b_input_mxint8.hex
    가 A_PREC=2/4 모드에서도 동일 layout 으로 쓰이고 TB 가 분해하는 패턴 답습.)
    """
    lines = []
    K, N = A_int.shape
    for n_t in range(N_T):
        for k_t in range(K_T):
            for col in range(TILE):
                n = n_t * TILE + col
                # 32 K-row × 8-bit = 256-bit.
                k_col = A_int[k_t * TILE:(k_t + 1) * TILE, n].astype(np.int32)
                k_col_u = (k_col & 0xFF).astype(np.uint8)
                word = 0
                for k in range(TILE):
                    word |= int(k_col_u[k]) << (8 * k)
                lines.append(f"{word:064x}\n")
    out_path.write_text("".join(lines))


def _emit_a_scale(W_scale: np.ndarray, out_path: Path,
                  K_T=4, M_T=4, TILE=32):
    """weight scale (E8M0, 1 byte) — (k_t, m_t, m_in) 순서로 emit.
    파일 인덱싱: a_scale[k_t*M_T*TILE + m_t*TILE + m_in]"""
    lines = []
    for k_t in range(K_T):
        for m_t in range(M_T):
            for m_in in range(TILE):
                m = m_t * TILE + m_in
                lines.append(f"{int(W_scale[m, k_t]):02x}\n")
    out_path.write_text("".join(lines))


def _emit_b_scale(A_scale: np.ndarray, out_path: Path,
                  N_T=4, K_T=4, TILE=32):
    """activation scale — (n_t, k_t, col) 순서."""
    lines = []
    for n_t in range(N_T):
        for k_t in range(K_T):
            for col in range(TILE):
                n = n_t * TILE + col
                lines.append(f"{int(A_scale[k_t, n]):02x}\n")
    out_path.write_text("".join(lines))


def emit_hex(W_int, W_scale, A_int, A_scale, prec_map, A_PREC: int, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _emit_a_input_bs(W_int, out_dir / "a_input_BS_mixed.hex")
    _emit_b_input(A_int, out_dir / f"b_input_mixed_A{A_PREC}.hex", A_PREC)
    _emit_a_scale(W_scale, out_dir / "a_scale_mixed.hex")
    _emit_b_scale(A_scale, out_dir / "b_scale_mixed.hex")
    (out_dir / "w_prec_per_block.hex").write_text(emit_hex_w_prec_only(prec_map))
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest sim/tests/test_gen_mixed.py -v`
Expected: 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sim/gen_mixed.py sim/tests/test_gen_mixed.py
git commit -m "feat(gen_mixed): hex emit (5 files) — weight bit-serial + W_PREC map"
```

---

## Task 6: `sim/gen_mixed.py` — CLI 진입점

**Files:**
- Modify: `sim/gen_mixed.py`

- [ ] **Step 1: CLI 구현** (`sim/gen_mixed.py` 끝에 append)

```python
def main():
    p = argparse.ArgumentParser(description="mixed-precision TB data + golden generator")
    p.add_argument("--A", type=int, required=True, choices=[2, 4, 8],
                   help="layer-wide A_PREC")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True,
                   help="출력 디렉토리 (예: work/mixed_A8)")
    p.add_argument("-M", type=int, default=128)
    p.add_argument("-K", type=int, default=128)
    p.add_argument("-N", type=int, default=128)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[gen_mixed] A_PREC={args.A}  seed={args.seed}  shape=({args.M},{args.K},{args.N})")

    prec_map = build_w_prec_map(seed=args.seed, M=args.M, K_T=args.K // BLOCK_SIZE)
    W_fp, A_fp = build_raw_data(seed=args.seed, M=args.M, K=args.K, N=args.N)
    W_int, W_scale = quantize_weight_mixed(W_fp, prec_map)
    A_int, A_scale = quantize_activation_uniform(A_fp, prec=args.A)
    C_golden = compute_golden_mixed(W_int, W_scale, A_int, A_scale, prec_map)

    emit_hex(W_int, W_scale, A_int, A_scale, prec_map, A_PREC=args.A, out_dir=out / "hw_input")
    visualize_prec_map(prec_map, A_PREC=args.A, out_dir=out)

    sw_ref = out / "sw_ref"
    sw_ref.mkdir(exist_ok=True)
    np.savez(sw_ref / "C_sw_mixed.npz", C=C_golden)
    (out / "hw_out").mkdir(exist_ok=True)

    print(f"[gen_mixed] done: hex+golden+viz → {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: smoke 실행**

Run: `python sim/gen_mixed.py --A 8 --seed 0 --out work/mixed_A8`
Expected: 정상 종료. `work/mixed_A8/` 아래에 `hw_input/`, `sw_ref/C_sw_mixed.npz`, `precision_map.png`, `precision_map.txt`, `hw_out/` 디렉토리 생성.

- [ ] **Step 3: 산출물 확인**

Run: `ls work/mixed_A8/hw_input/ && head -1 work/mixed_A8/precision_map.txt`
Expected:
- `a_input_BS_mixed.hex  a_scale_mixed.hex  b_input_mixed_A8.hex  b_scale_mixed.hex  w_prec_per_block.hex`
- 첫 line: `# precision_map  A_PREC=8  seed=0`

- [ ] **Step 4: Commit**

```bash
git add sim/gen_mixed.py
git commit -m "feat(gen_mixed): CLI entry point + smoke run"
```

---

## Task 7: `tb/gemm_sram_top_mixed_tb.v` — mixed-aware integration TB

**Files:**
- Create: `tb/gemm_sram_top_mixed_tb.v` (기존 `tb/gemm_sram_top_tb.v` 기반)

**전략:** 기존 9-mode TB 의 INIT/LOAD/CONFIG/DRIVE/DRAIN/DUMP 구조 답습. 변경 부분은 다음 3 곳만:
1. **LOAD**: `w_prec_per_block.hex` 추가 `$readmemh` + `W_PREC_MAP[M][K_T]` array.
2. **CONFIG**: A_PREC 의존 상수 (A_FIRE_DELAY, FIRES_PER_COL, N_T_LOGICAL, A_CTRL_CODE) 만 dispatch. W_CYC / FIRST_FIRE_GLOBAL / TOGGLE_VAL / W_CTRL_CODE 는 iteration 마다 동적 계산.
3. **DRIVE stage 3+4**: inner-loop 가 (n_t, k_t, m_t, m_in, bit_pos) 5-deep. m_in 진입마다 `W_PREC_MAP[m_t*32+m_in][k_t]` lookup → 그 iteration 의 `W_now` + `W_CTRL_CODE_now` 결정 → bit_pos loop W_now cycle.

나머지 (Stage 2-A/2-B, Stage 5 tail, capture/decode, drain FSM, dump) 는 기존과 동일하게 복사 (a_bs_word 인덱싱 식, build_in_b/build_in_scale_act 함수 등 모두 그대로).

**연산 정합성 우선** (memory feedback_spec_scope_computational_correctness.md): TOGGLE_VAL / scale_weight pulse 위치 등 dataflow 디테일은 worst-case 보수적 값으로 설정. random 패턴에서도 chain settle 보장.

- [ ] **Step 1: 기존 TB 복사 + 헤더 한글로 재작성**

```bash
cp tb/gemm_sram_top_tb.v tb/gemm_sram_top_mixed_tb.v
```

이후 `tb/gemm_sram_top_mixed_tb.v` 의 헤더를 다음으로 교체 (한글, 검증 목적/시나리오/동작 의도 — memory feedback_comments_korean.md 룰 준수):

```verilog
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// gemm_sram_top_mixed_tb — mixed-precision (per-block W_PREC) 검증 TB.
//
// 검증 목적:
//   weight 의 (M-row, K-block) 단위 random W_PREC ∈ {2,4,8} mix 가 RTL
//   변경 없이 GEMM+RMW+SRAM end-to-end 에서 FP32 bit-exact 동작함을
//   검증. 9-mode sweep TB 는 행렬 전체 균일 mode 만 검증하므로 본 TB
//   가 cycle 단위 in_Wcontrol 갱신 + Accumulator cnt 의 mixed-mode
//   fire 정합성을 보강.
//
// 검증 시나리오:
//   A_PREC ∈ {2,4,8} 각각 × seed=0 random W_PREC map.
//   bit-exact gate = MXP_Tools compare vs sim/gen_mixed.py 산출 골든.
//
// 동작 의도:
//   DRIVE 의 inner-loop 만 mixed-aware (m_in 마다 W_now lookup).
//   capture/decode/drain/dump 는 기존 9-mode TB 와 동일 — A_PREC 만
//   의존하고 W_PREC 는 fire 횟수에 영향 없음.
//
// Plusargs (xsim -testplusarg "KEY=VAL"):
//   A_PREC    : 2 / 4 / 8 (layer-wide activation 정밀도)
//   WORK_DIR  : sim/gen_mixed.py 의 --out 디렉토리 (필수)
//   DUMP_DIR  : SRAM .mem dump 출력 디렉토리 (필수)
//
// 회귀 게이트: `bash sim/run_mixed_sweep.sh` → "ALL 3 MIXED SCENARIOS PASSED".
//////////////////////////////////////////////////////////////////////////////
```

- [ ] **Step 2: LOAD 단계 변경 — `w_prec_per_block.hex` 추가 로드**

기존 LOAD 블록 (메인 시퀀스의 `[LOAD]` 섹션) 의 끝에 추가:

```verilog
        // mixed-prec: w_prec_per_block.hex 로드.
        // 각 line = m row 의 4 K-block W_CTRL_CODE packed (bit[2k+1:2k]).
        reg [7:0] w_prec_packed [0:127];
        $sformat(in_path_w_prec, "%0s/hw_input/w_prec_per_block.hex", WORK_DIR);
        $readmemh(in_path_w_prec, w_prec_packed);
```

`reg [8*512-1:0] in_path_w_prec;` 도 in_path_* 선언부에 추가.

`w_prec_packed[m][2*k_t+:2]` 로 W_CTRL_CODE lookup. 11=W8, 10=W4, 01=W2.

- [ ] **Step 3: CONFIG 단계 변경 — A_PREC 만 dispatch**

기존 `case ({A_PREC[3:0], B_PREC[3:0]})` 9-entry table 을 A_PREC 만 3-entry case 로 교체:

```verilog
        // mixed-prec CONFIG: A_PREC 의존 상수만 set.
        // W_CYC / FIRST_FIRE_GLOBAL / TOGGLE_VAL / W_CTRL_CODE 는
        // drive_stage_3_4 의 iteration 마다 동적 계산.
        case (A_PREC)
            8: begin
                A_FIRE_DELAY=2; FIRES_PER_COL=2048; N_T_LOGICAL=4;
                A_CTRL_CODE=A_INT8;
            end
            4: begin
                A_FIRE_DELAY=1; FIRES_PER_COL=1024; N_T_LOGICAL=2;
                A_CTRL_CODE=A_INT4;
            end
            2: begin
                A_FIRE_DELAY=0; FIRES_PER_COL=512;  N_T_LOGICAL=1;
                A_CTRL_CODE=A_INT2;
            end
            default: begin
                $display("ERROR: unsupported A_PREC=%0d", A_PREC); $finish;
            end
        endcase
```

- [ ] **Step 4: `drive_stage_3_4` task 재구조 — inner-loop 5-deep**

```verilog
    // mixed-aware MAC sweep. inner-loop:
    //   n_t → k_t → m_t → m_in → bit_pos
    // 각 m_in 진입마다 W_PREC_MAP[m_t*32+m_in][k_t] lookup → W_now / W_ctrl_now.
    // 이 iteration 의 bit_pos loop 길이 = W_now (2/4/8).
    //
    // TOGGLE_VAL / FIRST_FIRE_GLOBAL 은 worst-case (W=8) 기준 보수 — 첫
    // K-tile 진입 후 24 cycle 시점에 station selector toggle, cyc_global=28
    // 시점에 start_accumulate 1 회 펄스. random pattern 의 첫 m_in 의
    // W_now 가 2 여도 chain settle 마진 (≥16 cy) 안에 fire 시작 없음.
    task drive_stage_3_4_mixed;
        integer m_in_idx;
        reg [1:0] w_ctrl_now;
        integer w_now;
        integer bp;
        begin
            cyc_global = 0;
            cur_pp = 1'b1;
            capture_en <= 1'b1;

            for (n_t = 0; n_t < N_T_LOGICAL; n_t = n_t + 1) begin
                for (k_t = 0; k_t < K_T; k_t = k_t + 1) begin
                    for (m_t = 0; m_t < M_T; m_t = m_t + 1) begin
                        for (m_in_idx = 0; m_in_idx < TILE_SIZE; m_in_idx = m_in_idx + 1) begin
                            // (m, k_t) 의 W_CTRL_CODE lookup.
                            w_ctrl_now = w_prec_packed[m_t*TILE_SIZE + m_in_idx][2*k_t +: 2];
                            case (w_ctrl_now)
                                W_INT8: w_now = 8;
                                W_INT4: w_now = 4;
                                W_INT2: w_now = 2;
                                default: begin
                                    $display("FATAL: invalid W_CTRL %b at m=%0d k_t=%0d",
                                             w_ctrl_now, m_t*TILE_SIZE+m_in_idx, k_t);
                                    $finish;
                                end
                            endcase

                            for (bp = w_now - 1; bp >= 0; bp = bp - 1) begin
                                cyc_in_K = cyc_global - (k_t > 0 ? k_t_start_cycle : 0);
                                // (단순화: cyc_in_K 는 k_t 진입 후 cumulative cycle.
                                //  cf. step-5 의 보조 reg k_t_start_cycle.)

                                // weight bit-serial — a_bs[] 의 인덱싱은 W=8 padding
                                // 컨벤션 (gen_mixed.py 의 _emit_a_input_bs 와 일치).
                                //   o_padded = m_in*8 + (7 - bp)   // MSB-first 채움
                                in_a <= a_bs_word(k_t, m_t, m_in_idx*8 + (7 - bp));

                                // start_accumulate 펄스: 첫 K-tile 의 cycle 17 (chain
                                // settle 직후) 한 번. 기존 9-mode 와 동일 시점.
                                if (n_t == 0 && k_t == 0 && cyc_global == 17) begin
                                    in_start_accumulate <= 1'b1;
                                end else begin
                                    in_start_accumulate <= 1'b0;
                                end

                                // Wcontrol: cyc_global=17 전엔 W_IDLE, 이후 w_ctrl_now.
                                if (n_t == 0 && k_t == 0 && cyc_global < 17) begin
                                    in_Wcontrol <= W_IDLE;
                                end else begin
                                    in_Wcontrol <= w_ctrl_now;
                                end

                                // PE in_control toggle: K-tile 진입 첫 cycle.
                                if (cyc_in_K == 0 && !(n_t == 0 && k_t == 0)) begin
                                    cur_pp = ~cur_pp;
                                end
                                in_control <= {32{cur_pp}};

                                // Station selector toggle: K-tile 진입 후 24 cycle
                                // (worst-case W=8 기준 보수, random 패턴에서도 안전).
                                if (cyc_in_K == 24 && !(n_t == 0 && k_t == 0)) begin
                                    cur_st_pp = ~cur_st_pp;
                                end
                                in_station_control <= cur_st_pp;

                                // scale_weight 드라이브: 매 m_in iteration 의 첫 bit
                                // (bp == w_now-1) 시점에 새 scale.
                                if (bp == w_now - 1) begin
                                    in_scale_weight <= a_scale[
                                        k_t * M_T * TILE_SIZE +
                                        m_t * TILE_SIZE + m_in_idx];
                                end

                                // Prefetch (기존 9-mode 와 동일 위치: m_t=1, m_in<32 의
                                // 첫 bit 시점).  bit_pos==w_now-1 일 때만 다음 tile
                                // B/scale 한 col 분 driving.
                                if (m_t == 1 && bp == w_now - 1) begin
                                    if (k_t < K_T - 1) begin
                                        drive_prefetch(n_t, k_t + 1, m_in_idx);
                                    end else if (n_t < N_T_LOGICAL - 1) begin
                                        drive_prefetch(n_t + 1, 0, m_in_idx);
                                    end else begin
                                        in_Station_control  <= A_IDLE;
                                        in_loadEN           <= 0;
                                        in_station_loadEN   <= 0;
                                        in_b                <= 0;
                                        in_Scale_Activation <= 0;
                                    end
                                end else begin
                                    in_Station_control  <= A_IDLE;
                                    in_loadEN           <= 0;
                                    in_station_loadEN   <= 0;
                                    in_b                <= 0;
                                    in_Scale_Activation <= 0;
                                end

                                @(posedge clk);
                                cyc_global = cyc_global + 1;
                            end // bp loop
                        end // m_in loop
                    end // m_t loop
                    k_t_start_cycle = cyc_global; // 다음 K-tile 의 cyc_in_K 기준점.
                end // k_t loop
            end // n_t loop
        end
    endtask
```

`integer k_t_start_cycle = 0;` 을 module 상위에 선언. 호출 부 (메인 시퀀스의 `drive_stage_3_4;` → `drive_stage_3_4_mixed;` 로 교체).

- [ ] **Step 5: 기존 task 들 (drive_stage_2a/2b, drive_stage_5_tail, capture always, drain FSM, dump) 은 그대로 유지 — `W_CYC` 직접 참조 부분만 worst-case 8 로 치환**

`drive_stage_5_tail` 의 scale_weight 펄스 조건 `(cyc_global - FIRST_FIRE_GLOBAL) % W_CYC == 0` 를 다음으로 교체:

```verilog
            // mixed-prec: tail 의 in-flight fire 들도 worst W=8 기준 conservative
            // scale_weight 펄스 (8 cycle 마다). chain settle 후 trailing fire 들은
            // 이미 driving 단계에서 scale_weight 가 latch 되었으므로 tail 의 펄스는
            // 안전망 — 잘못된 scale 이 새로 들어가지는 않음.
            if (cyc_global >= FIRST_FIRE_TAIL_BASE &&
                ((cyc_global - FIRST_FIRE_TAIL_BASE) % 8) == 0) begin
                // tail 구간은 마지막 in-flight 만이라 scale 변경 안 됨. no-op 가능.
            end
```

`FIRST_FIRE_GLOBAL` 은 mixed 에서도 28 (cyc_global=17 펄스 후 + chain delay) — 기존과 동일.

- [ ] **Step 6: elab smoke**

`sim/run_top_elab.sh` 를 참고해 임시 smoke 스크립트 작성 (아니면 직접 명령):

```bash
mkdir -p sim/build/mixed_elab
cd sim/build/mixed_elab
xvlog -sv \
    ../../../third_party/berkeley-hardfloat/HardFloatBundle.v \
    ../../../gemm_sram.srcs/sources_1/imports/Desktop/MXP/MXP.srcs/sources_1/new/*.v \
    ../../../gemm_sram.srcs/sources_1/imports/Desktop/sram/rtl/*.v \
    ../../../gemm_sram.srcs/sources_1/new/int_to_fp32.v \
    ../../../gemm_sram.srcs/sources_1/new/fp32_adder.v \
    ../../../gemm_sram.srcs/sources_1/new/RMW.v \
    ../../../gemm_sram.srcs/sources_1/new/sram_1rw_banked_mp.v \
    ../../../gemm_sram.srcs/sources_1/new/GEMM.v \
    ../../../gemm_sram.srcs/sources_1/new/gemm_sram_top.v \
    ../../../tb/gemm_sram_top_mixed_tb.v && \
xelab -L work gemm_sram_top_mixed_tb -snapshot smoke_snap
```

Expected: warning 만, error 0.

- [ ] **Step 7: Commit**

```bash
git add tb/gemm_sram_top_mixed_tb.v
git commit -m "feat(tb): mixed-precision integration TB (per-block W_PREC dispatch)"
```

---

## Task 8: `sim/run_mixed_one.sh` — A_PREC 1 시나리오 end-to-end

**Files:**
- Create: `sim/run_mixed_one.sh`

- [ ] **Step 1: 작성**

```bash
#!/bin/bash
# sim/run_mixed_one.sh — mixed-precision 단일 시나리오 (1 A_PREC).
#
# Usage:   bash sim/run_mixed_one.sh <A_PREC>
# Example: bash sim/run_mixed_one.sh 8
#
# 절차: gen_mixed.py → xsim → MXP_Tools compare (vs C_sw_mixed.npz).
set -e
cd "$(dirname "$0")/.."

A_PREC="${1:-8}"
LABEL="mixed_A${A_PREC}"
WORK="work/${LABEL}"
DUMP="${WORK}/hw_out"
BUILD="sim/build/${LABEL}"

mkdir -p "$DUMP" "$BUILD"

# 1) Python: hex + golden + viz.
python sim/gen_mixed.py --A "$A_PREC" --seed 0 --out "$WORK"

# 2) HW sim.
SRC_ROOT="gemm_sram.srcs/sources_1"
HF_ROOT="third_party/berkeley-hardfloat"
(cd "$BUILD" && \
    xvlog -sv \
        ../../../$HF_ROOT/HardFloatBundle.v \
        ../../../$SRC_ROOT/imports/Desktop/MXP/MXP.srcs/sources_1/new/*.v \
        ../../../$SRC_ROOT/imports/Desktop/sram/rtl/*.v \
        ../../../$SRC_ROOT/new/int_to_fp32.v \
        ../../../$SRC_ROOT/new/fp32_adder.v \
        ../../../$SRC_ROOT/new/RMW.v \
        ../../../$SRC_ROOT/new/sram_1rw_banked_mp.v \
        ../../../$SRC_ROOT/new/GEMM.v \
        ../../../$SRC_ROOT/new/gemm_sram_top.v \
        ../../../tb/gemm_sram_top_mixed_tb.v && \
    xelab -L work gemm_sram_top_mixed_tb -snapshot mixed_snap && \
    cmd //c "xsim mixed_snap -runall -testplusarg \"A_PREC=$A_PREC\" -testplusarg \"WORK_DIR=../../../$WORK\" -testplusarg \"DUMP_DIR=../../../$DUMP\"")

# 3) bit-exact compare.
BANKS=$(printf "../work/${LABEL}/hw_out/bank%d.mem " {0..31})
(cd MXP_Tools && \
    python -m mxp_tools compare \
        --ref ../work/${LABEL}/sw_ref/C_sw_mixed.npz \
        --hw-banks ${BANKS} \
        --layout interleaved_row_major_32bank)

echo "mixed_A${A_PREC}: PASS"
```

- [ ] **Step 2: 실행 권한 부여 + A=8 실행**

```bash
chmod +x sim/run_mixed_one.sh
bash sim/run_mixed_one.sh 8
```

Expected (정상 시): 마지막 line `mixed_A8: PASS`. compare subcommand 가 32 bank 의 FP32 값이 골든과 비트단위 일치 확인.

**실패 시 디버그 절차** (TB 의 cycle 타이밍이 mixed pattern 과 안 맞는 경우):
1. `bash sim/run_mixed_one.sh 8` 의 xsim stdout 에서 `captured` / `drained` 카운트 확인 — 65536 아니면 capture 또는 drain 의 timing 문제.
2. `work/mixed_A8/hw_out/bank0.mem` 의 처음 32 line 과 `C_sw_mixed.npz` 의 C[0:32, 0] 을 직접 비교 — 첫 fire 가 잘못된 위치/값.
3. cycle timing 가 의심되면 `tb/gemm_sram_top_mixed_tb.v` 의 `drive_stage_3_4_mixed` 의 `$display` 디버그 추가 (`if (cyc_global < 100) $display(...);`).

- [ ] **Step 3: Commit**

```bash
git add sim/run_mixed_one.sh
git commit -m "feat(sim): run_mixed_one.sh — A_PREC 단일 시나리오 end-to-end"
```

---

## Task 9: `sim/run_mixed_sweep.sh` — 3 시나리오 sweep + gate

**Files:**
- Create: `sim/run_mixed_sweep.sh`
- Modify: `.gitignore`

- [ ] **Step 1: sweep 스크립트 작성**

```bash
#!/bin/bash
# sim/run_mixed_sweep.sh — A_PREC ∈ {2,4,8} × seed=0 직렬 sweep.
#
# 절차: 각 A 별로 sim/run_mixed_one.sh 실행 → bit-exact compare.
# 마지막에 "ALL 3 MIXED SCENARIOS PASSED" 출력 (전체 PASS 시).
#
# Note: set -e 사용 안 함 — 한 시나리오 fail 해도 나머지 시도.
cd "$(dirname "$0")/.."

PASSED=()
FAILED=()

for A in 2 4 8; do
  echo ""
  echo "==================== mixed_A${A} ===================="
  if bash sim/run_mixed_one.sh "${A}"; then
    PASSED+=("mixed_A${A}")
  else
    FAILED+=("mixed_A${A}")
  fi
done

echo ""
echo "===================================="
echo "MIXED SWEEP RESULT: ${#PASSED[@]}/3 PASS"
echo "PASSED: ${PASSED[*]}"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi
echo "ALL 3 MIXED SCENARIOS PASSED"
```

- [ ] **Step 2: `.gitignore` 갱신**

`.gitignore` 끝에 추가:

```
work/mixed_A*/
```

- [ ] **Step 3: sweep 실행**

```bash
chmod +x sim/run_mixed_sweep.sh
bash sim/run_mixed_sweep.sh
```

Expected: 마지막 line `ALL 3 MIXED SCENARIOS PASSED`.

- [ ] **Step 4: Commit**

```bash
git add sim/run_mixed_sweep.sh .gitignore
git commit -m "feat(sim): run_mixed_sweep.sh — 3-scenario sweep + ALL PASSED gate"
```

---

## Task 10: 회귀 게이트 — 기존 9-mode sweep + 새 mixed sweep 둘 다 통과 확인

**Files:** (변경 없음 — 검증만)

- [ ] **Step 1: 기존 9-mode sweep 재실행**

Run: `bash sim/run_integration_sweep.sh`
Expected: `ALL 9 MODES PASSED` — mixed TB 도입이 기존 회귀를 깨지 않았는지 확인.

- [ ] **Step 2: 새 mixed sweep 재실행**

Run: `bash sim/run_mixed_sweep.sh`
Expected: `ALL 3 MIXED SCENARIOS PASSED`.

- [ ] **Step 3: 모든 회귀 통과 확인 후 CLAUDE.md next-session 섹션 갱신 commit (옵션)**

`CLAUDE.md` 의 "Next session kickoff" 섹션에 mixed-prec TB 완성 사실 추가.

```bash
git add CLAUDE.md
git commit -m "docs: mixed-precision TB 완성 (3-scenario sweep PASS)"
```

---

## Self-Review

- [x] **Spec coverage** — spec 의 § 3 (architecture), § 4 (hex), § 5 (golden), § 6 (gate), § 7 (검증 항목), § 9 (visualize), § 10 (file layout) 모두 task 1~10 에 매핑됨. § 8 (non-goals) 는 본질적으로 task 없음 (= 본 plan 범위 외).
- [x] **Placeholder scan** — "TBD" / "fill in details" / 빈 code block 없음. Task 7 의 task 5 에서 `FIRST_FIRE_TAIL_BASE` 같은 신호 이름이 등장하는데 의도적으로 — 기존 TB 의 `FIRST_FIRE_GLOBAL` 이름을 답습. Step 5 의 변경 자체가 단순 주석 추가라 신규 신호 정의는 없음.
- [x] **Type consistency** — `build_w_prec_map` (uint8) / `prec_map` arg / `quantize_weight_mixed` 의 `prec_map` 모두 (M, K_T) shape 일관. `W_int` / `A_int` 는 int8, scale 들은 uint8. golden `C` 는 float32.
- [x] **명령 일관성** — Task 6/8/9 의 `python sim/gen_mixed.py --A {2,4,8} --seed 0 --out work/mixed_A{P}` 인자 일관. compare 의 `--ref` 파일명 `C_sw_mixed.npz` 도 gen_mixed.py 산출 이름과 일치.

---

## Risk callouts (implementation 진행 중 확인 필요)

1. **Task 4 의 dequant 식**: mxp_tools 의 정확한 식과 일치 안 하면 첫 테스트가 fail. 그 경우 `MXP_Tools/mxp_tools/gemm.py` 본문 읽어 누적 순서 + 정밀도 변환식 맞춤.
2. **Task 5 의 hex layout**: 기존 9-mode sweep 의 `b_input_mxint{P}.hex` 의 byte ordering 과 100% 일치해야 TB 의 `build_in_b` 가 정상 풀어냄. uniform mode (모든 W_PREC=8) 로 산출한 mixed hex 가 기존 9-mode sweep 의 hex 와 동일한지 단위 검증 추가 권장.
3. **Task 7 의 TOGGLE_VAL=24**: random 패턴의 첫 K-tile 의 첫 m_in 의 W_PREC 가 2 일 때 chain settle 마진 ≥ 16 cycle 안에 fire 발생 안 함을 확인. 만약 fire 가 너무 빨리 발생하면 TOGGLE_VAL 을 더 보수적으로 (e.g., 30) 조정.
4. **Task 8 의 실패 시 디버그**: TB 의 cycle 타이밍이 mixed pattern 의 worst-case 와 안 맞으면 `drive_stage_5_tail` 의 TAIL_CYC=45 가 부족할 수 있음. 첫 시도 fail 시 60 으로 늘려 재시도.
