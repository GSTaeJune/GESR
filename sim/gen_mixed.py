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

    Debug overrides (priority 순):
      MIXED_W_UNIFORM=<2|4|8>     — 전체 (M, K_T) 균일 W. isolation test 용.
      MIXED_W_K_TILE=<v0,v1,v2,v3> — K-tile 별 W 지정. 한 K-tile 안의 모든 M-row 는 같은 W.
                                     예: "8,4,2,8" → K-tile 0=W=8, 1=4, 2=2, 3=8.
                                     mixed-W granularity 분기 테스트 용.
    둘 다 unset 이면 random (M, K_T).
    """
    import os
    uniform = os.environ.get("MIXED_W_UNIFORM")
    if uniform is not None:
        p = int(uniform)
        assert p in (2, 4, 8), f"MIXED_W_UNIFORM must be 2/4/8, got {p}"
        return np.full((M, K_T), p, dtype=np.uint8)
    k_tile = os.environ.get("MIXED_W_K_TILE")
    if k_tile is not None:
        vals = [int(v) for v in k_tile.split(",")]
        assert len(vals) == K_T, f"MIXED_W_K_TILE must have {K_T} vals, got {len(vals)}"
        for v in vals:
            assert v in (2, 4, 8), f"W must be 2/4/8, got {v}"
        # 한 K-tile 안의 128 M-row 가 모두 같은 W (M-row 차원 mix 없음).
        return np.tile(np.array(vals, dtype=np.uint8), (M, 1))
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


# ---------------------------------------------------------------------------
# Hex emit helpers
# ---------------------------------------------------------------------------

# W_CTRL_CODE lookup (precision_modes_protocol.md §1 과 일치)
_W_CTRL_LUT = {8: 0b11, 4: 0b10, 2: 0b01}


def emit_hex_w_prec_only(prec_map: np.ndarray) -> str:
    """prec_map (M, K_T) → 'hh\\n' 행 누적 문자열. 단위 검증용.

    각 row m 에 대해 1 byte (8-bit) 를 생성:
      bit[2*kt+1 : 2*kt] = _W_CTRL_LUT[prec_map[m, kt]]  for kt in 0..K_T-1.
    """
    out = []
    for m in range(prec_map.shape[0]):
        byte = 0
        for kt in range(prec_map.shape[1]):
            byte |= _W_CTRL_LUT[int(prec_map[m, kt])] << (2 * kt)
        out.append(f"{byte:02x}\n")
    return "".join(out)


def _emit_a_input_bs(W_int: np.ndarray, out_path: Path,
                     K_T: int = 4, M_T: int = 4, TILE: int = 32, W_PAD: int = 8):
    """weight bit-serial hex. W=8 padding 컨벤션 — 모든 entry 8 cycle/row.

    한 워드 (32-bit) = 한 TB cycle 의 32 K-row 의 1 bit.
    파일 인덱싱: a_bs[k_t * M_T * TILE * W_PAD + m_t * TILE * W_PAD + m_in * W_PAD + bit_pos]
    bit_pos ∈ 0..W_PAD-1, bit_pos=0 이 MSB (즉 MSB first emit).
    W_PAD=8 보다 낮은 precision 의 경우 MSB 쪽 bit_pos 가 0 → word=0 (dot-product 기여 없음).
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
                  N_T: int = 4, K_T: int = 4, TILE: int = 32):
    """activation 256-bit 워드 hex.

    한 워드 = 한 (n_t, k_t, col) 의 32 K-row × 8-bit activation.
    파일 인덱싱: b_bp[n_t * K_T * TILE + k_t * TILE + col]
    A_PREC=4: 8-bit 안에 INT4 두 개 packed (TB 의 pack_int4_n_pair 가 풀어냄).
    A_PREC=2: 8-bit 안에 INT2 네 개 packed.

    emit 단계는 A_PREC 무관 INT8 raw 8-bit 그대로. TB 의 V3 pack 함수가
    mode 별로 4-bit/2-bit 추출. (기존 9-mode sweep 의 b_input_mxint8.hex 가
    A_PREC=2/4 모드에서도 동일 layout 으로 쓰이고 TB 가 분해하는 패턴 답습.)
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
                  K_T: int = 4, M_T: int = 4, TILE: int = 32):
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
                  N_T: int = 4, K_T: int = 4, TILE: int = 32):
    """activation scale — (n_t, k_t, col) 순서.
    파일 인덱싱: b_scale[n_t * K_T * TILE + k_t * TILE + col]"""
    lines = []
    for n_t in range(N_T):
        for k_t in range(K_T):
            for col in range(TILE):
                n = n_t * TILE + col
                lines.append(f"{int(A_scale[k_t, n]):02x}\n")
    out_path.write_text("".join(lines))


def emit_hex(W_int: np.ndarray, W_scale: np.ndarray,
             A_int: np.ndarray, A_scale: np.ndarray,
             prec_map: np.ndarray, A_PREC: int,
             out_dir: "Path | str") -> None:
    """5 개 hex 파일을 out_dir 에 emit.

    파일 목록:
      a_input_BS_mixed.hex    — weight bit-serial (W=8 padding)
      b_input_mixed_A{P}.hex  — activation 256-bit packed
      a_scale_mixed.hex       — weight scale (E8M0)
      b_scale_mixed.hex       — activation scale (E8M0)
      w_prec_per_block.hex    — per-block W_PREC 2-bit code packed
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _emit_a_input_bs(W_int, out_dir / "a_input_BS_mixed.hex")
    _emit_b_input(A_int, out_dir / f"b_input_mixed_A{A_PREC}.hex", A_PREC)
    _emit_a_scale(W_scale, out_dir / "a_scale_mixed.hex")
    _emit_b_scale(A_scale, out_dir / "b_scale_mixed.hex")
    (out_dir / "w_prec_per_block.hex").write_text(emit_hex_w_prec_only(prec_map))


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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

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

    # FP32 reference matmul — unquantized truth for compare's 3-way diff.
    # Use single-precision matmul to match the dtype convention used by
    # mxp_tools.cli.cmd_gen (which also stores C_fp32 as float32).
    C_fp32_truth = (W_fp.astype(np.float32) @ A_fp.astype(np.float32)).astype(np.float32)

    emit_hex(W_int, W_scale, A_int, A_scale, prec_map, A_PREC=args.A, out_dir=out / "hw_input")
    visualize_prec_map(prec_map, A_PREC=args.A, out_dir=out)

    sw_ref = out / "sw_ref"
    sw_ref.mkdir(exist_ok=True)
    np.savez(sw_ref / "C_sw_mixed.npz", C_sw=C_golden, C_fp32=C_fp32_truth)
    (out / "hw_out").mkdir(exist_ok=True)

    print(f"[gen_mixed] done: hex+golden+viz → {out}")


if __name__ == "__main__":
    main()
