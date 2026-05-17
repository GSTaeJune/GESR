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
import sys
from pathlib import Path

import numpy as np

# MXP_Tools 를 import path 에 추가 (sim/gen_mixed.py 기준 ../MXP_Tools/)
_MXP_TOOLS = Path(__file__).resolve().parent.parent / "MXP_Tools"
if str(_MXP_TOOLS) not in sys.path:
    sys.path.insert(0, str(_MXP_TOOLS))

from mxp_tools.quant import quantize_block_mx, quantize_matrix_mx  # noqa: E402
from mxp_tools.const import BLOCK_SIZE, IMPLICIT_SCALE_EXP  # noqa: E402


def build_w_prec_map(seed: int = 0, M: int = 128, K_T: int = 4) -> np.ndarray:
    """(M, K_T) shape 의 W_PREC array ∈ {2,4,8} 균등 random.

    seed 고정 시 재현 가능. dtype=uint8.
    """
    rng = np.random.default_rng(seed)
    choices = np.array([2, 4, 8], dtype=np.uint8)
    return choices[rng.integers(0, 3, size=(M, K_T), dtype=np.int64)]


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
        # Set Agg only if no backend has been chosen yet — avoids "Backend already
        # set" warning when caller imported matplotlib.pyplot first.
        if not matplotlib.get_backend() or "agg" not in matplotlib.get_backend().lower():
            try:
                matplotlib.use("Agg", force=False)
            except (ImportError, ValueError):
                pass  # backend already locked in; we'll render with whatever it is.
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


def compute_golden_mixed(
    W_int: np.ndarray,
    W_scale: np.ndarray,
    A_int: np.ndarray,
    A_scale: np.ndarray,
    prec_map: np.ndarray,
    prec_b: int = 8,
) -> np.ndarray:
    """FP32 GEMM 골든 — K-block 단위 누적, mxint_gemm_golden 의 accumulation 미러.

    mxint_gemm_golden 과 동일한 방식으로 누적하되 weight 의 implicit_scale 이
    (M, K_T) block 단위로 달라지는 점만 다름.

    Args:
      W_int    : int8,  (M, K)   — weight (mxp_tools 의 int_A 역할)
      W_scale  : uint8, (M, K_T) — per-block E8M0 weight scale
      A_int    : int8,  (K, N)   — activation (mxp_tools 의 int_B 역할)
      A_scale  : uint8, (K_T, N) — per-block E8M0 activation scale
      prec_map : uint8, (M, K_T) — per-(row,block) W_PREC ∈ {2,4,8}
      prec_b   : int            — activation precision (uniform), default 8

    Returns:
      C : float32, (M, N)
    """
    M, K = W_int.shape
    _, N = A_int.shape
    K_T = K // BLOCK_SIZE
    assert prec_map.shape == (M, K_T), f"prec_map shape {prec_map.shape} != ({M}, {K_T})"
    assert W_scale.shape == (M, K_T)
    assert A_scale.shape == (K_T, N)

    # E8M0 → FP32: 2^(e - 127) — mirrors mxint_gemm_golden._e8m0_to_fp32
    w_scale_fp = (2.0 ** (W_scale.astype(np.float64) - 127.0)).astype(np.float32)  # (M, K_T)
    a_scale_fp = (2.0 ** (A_scale.astype(np.float64) - 127.0)).astype(np.float32)  # (K_T, N)

    # Activation implicit_scale is uniform (same prec_b for all blocks).
    impl_b = IMPLICIT_SCALE_EXP[prec_b]

    C = np.zeros((M, N), dtype=np.float32)
    for blk in range(K_T):
        sl = slice(blk * BLOCK_SIZE, (blk + 1) * BLOCK_SIZE)
        # int32 block matmul — shape (M, N)
        block_int = W_int[:, sl].astype(np.int32) @ A_int[sl, :].astype(np.int32)  # (M, N)

        # Per-row implicit scale for weight: depends on per-block prec_map[m, blk].
        # Build a (M, 1) FP32 array of 2^-(impl_a[m] + impl_b).
        impl_a_vec = np.array(
            [IMPLICIT_SCALE_EXP[int(prec_map[m, blk])] for m in range(M)],
            dtype=np.float64,
        )
        # FP32 scalar per row: mirrors np.float32(2.0 ** -(impl_a + impl_b))
        implicit_scale_vec = (2.0 ** -(impl_a_vec + impl_b)).astype(np.float32)  # (M,)

        block_fp = (
            block_int.astype(np.float32)
            * w_scale_fp[:, blk : blk + 1]       # (M, 1) broadcast
            * a_scale_fp[blk : blk + 1, :]       # (1, N) broadcast
            * implicit_scale_vec[:, np.newaxis]   # (M, 1) broadcast
        )
        C += block_fp
    return C
