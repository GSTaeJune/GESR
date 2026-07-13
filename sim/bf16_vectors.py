"""ml_dtypes oracle: emit $readmemh vector files for the bf16 primitive TBs.

Each line is space-separated hex words: inputs then expected output. TBs read
with $readmemh into a reg array and split fields by bit-slicing. ASCII-only.

int_to_bf16 vectors cover the full signed 9-bit scale domain (negative scales -> underflow flush) plus directed subnormal-boundary binades (recoded exp 115..133).
"""
import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

BLOCK = 32
IMPLICIT = {8: 6, 4: 2, 2: 0}   # mirrors mxp_tools.const.IMPLICIT_SCALE_EXP


def _u32(x_f32):
    return int(np.float32(x_f32).view(np.uint32))


def _u16_bf16(x_f32):
    return int(np.float32(x_f32).astype(bfloat16).view(np.uint16))


def gen_fp32_to_bf16(path, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    # random normals across magnitudes
    vals += list(rng.standard_normal(50000).astype(np.float32) * rng.uniform(1e-3, 1e3, 50000).astype(np.float32))
    # directed: ties, subnormal band, overflow band, zeros, inf
    vals += [0.0, -0.0, 1.0, -1.0, 1.0 + 2.0**-8, 1.0 + 3.0 * 2.0**-8,
             256.0, 257.0, 3.3e38, -3.3e38, np.float32(np.inf), np.float32(-np.inf)]
    vals += list(rng.uniform(2.0**-140, 2.0**-125, 20000))
    with open(path, "w") as f:
        for v in vals:
            x = np.float32(v)
            f.write(f"{_u32(x):08x} {_u16_bf16(x):04x}\n")
    return len(vals)


def gen_int_to_bf16(path, seed=1):
    # Model int_to_bf16(in_int, scale) = bf16(int -> bf16) * 2^(scale-127), matching
    # the golden's int->bf16 then exponent-shift. scale is the 9-bit combined value,
    # SIGNED: [-256, 255]. Negative scales arise in real workloads (act/weight
    # E8M0 < 127) and force deep underflow, which must flush to +-0 exactly like
    # the golden's ldexp -> fp32 -> bf16 chain (spec 6.1 carry-over #1). The
    # boundary section densely covers the bf16 subnormal band (recoded exp
    # 122..129), where v1's FNFromRecFN truncation diverged from RNE.
    rng = np.random.default_rng(seed)
    rows = []

    def expect(iv, sc):
        with np.errstate(over="ignore"):
            r = np.float32(iv).astype(bfloat16)          # int -> bf16 (RNE), exact int32<2^24
            val = np.ldexp(np.float64(r), sc - 127)      # * 2^(scale-127)
            return int(np.float32(val).astype(bfloat16).view(np.uint16))

    # realizable block_int magnitudes (|.| < 2^20) and E8M0-derived combined scales.
    ints = list(rng.integers(-(2**19), 2**19, 4000)) + [0, 1, -1, 127, -128,
             2**19 - 1, -(2**19), 255, 256, 257, 2**15, 2**15 + 1]
    # combined scale: full 9-bit signed domain, incl. the negative underflow band
    # (un-tested in Phase 2a -- this is what gates the v2 subnormal/flush path).
    scales = (list(rng.integers(40, 200, 60))
              + list(rng.integers(-256, 40, 60))
              + [127, 0, 1, 254, 255, 60, 62, 70, 100, 150, 200,
                 -1, -2, -64, -127, -128, -200, -255, -256])
    for iv in ints:
        for sc in rng.choice(scales, size=8, replace=False):
            iv = int(iv); sc = int(sc)
            rows.append((iv & 0xFFFFFFFF, sc & 0x1FF, expect(iv, sc)))

    # Boundary-directed: land the RESULT exponent on the flush / round-up /
    # deep-subnormal binades around bf16 min subnormal 2^-133 (recoded exp
    # band ~115..133). Includes the exact tie 2^-134 (iv=1, target -134 -> +0
    # ties-to-even) and the round-up 1.5*2^-134 (iv=3 -> min subnormal).
    # sc chosen so r * 2^(sc-127) has exponent `target`.
    for iv in [1, -1, 3, -3, 5, 255, -255, 257, 2**19 - 1, -(2**19), 127, 129]:
        r = float(np.float32(iv).astype(bfloat16))
        e_r = int(np.floor(np.log2(abs(r))))
        for target in range(-141, -123):
            sc = target - e_r + 127
            if -256 <= sc <= 255:
                rows.append((int(iv) & 0xFFFFFFFF, sc & 0x1FF, expect(int(iv), sc)))

    with open(path, "w") as f:
        for a, s, o in rows:
            f.write(f"{a:08x} {s:03x} {o:04x}\n")
    return len(rows)


def gen_bf16_add(path, seed=2):
    rng = np.random.default_rng(seed)
    def rbf16(n):
        return (rng.standard_normal(n).astype(np.float32) * rng.uniform(1e-2, 1e2, n).astype(np.float32)).astype(bfloat16)
    def bp(bits):
        # bf16 scalar from a raw bit pattern
        return np.array([bits], dtype=np.uint16).view(bfloat16)[0]
    A = list(rbf16(200000))
    B = list(rbf16(200000))
    # directed edges appended
    inf = np.float32(np.inf).astype(bfloat16)
    edges = [(inf, -inf), (inf, inf), (bfloat16(1.0), bfloat16(-1.0)),
             (bfloat16(2.0**-133), bfloat16(2.0**-133)), (bfloat16(0.0), bfloat16(-0.0))]
    # Native-adder directed edges (2026-07-13 A6 rewrite). Witnesses for the
    # hand-written paths: finite overflow -> inf, subnormal/normal boundary
    # crossings, d>11 sticky collapse (incl. sub + renorm + round-carry chain),
    # exact-zero signs, RNE ties. Expected values still come from numpy below.
    mx = bp(0x7F7F)                      # max finite bf16
    edges += [
        (mx, mx),                        # finite overflow -> +inf
        (bp(0xFF7F), bp(0xFF7F)),        # -> -inf
        (mx, bp(0x7F00)),                # near-overflow rounding
        (bp(0x7F00), mx),                # commuted (swap path)
        (bp(0x0080), bp(0x8001)),        # min normal - min subnormal -> 0x007F
        (bp(0x0080), bp(0x0001)),        # min normal + min subnormal -> 0x0081 (exact)
        (bp(0x3F80), bp(0x3380)),        # 1.0 + 2^-24: d=24 pure-sticky -> 1.0
        (bp(0x3F80), bp(0xB380)),        # 1.0 - 2^-24: sub + shift + round-carry -> 1.0
        (bp(0x0001), bp(0x8001)),        # x + (-x), subnormal -> +0
        (bp(0x4321), bp(0xC321)),        # x + (-x), normal -> +0
        (bp(0x8000), bp(0x8000)),        # -0 + -0 -> -0
        (bp(0x3F80), bp(0x3B80)),        # 1.0 + 2^-8: tie -> 1.0 (ties-to-even)
        (bp(0x3F81), bp(0x3B80)),        # (1+2^-7) + 2^-8: tie, odd lsb -> round up
        (bp(0x0003), bp(0x0005)),        # subnormal + subnormal (exact grid)
        (bp(0x00FF), bp(0x0001)),        # max subnormal + min subnormal -> min normal
        (bp(0x0100), bp(0x8080)),        # 2^-125 - 2^-126 -> 2^-126 (d=1 borrow)
    ]
    for a, b in edges:
        A.append(a); B.append(b)
    with open(path, "w") as f:
        with np.errstate(over="ignore"):  # (max+max) legitimately overflows fp32 -> inf
            for a, b in zip(A, B):
                s = (np.float32(a) + np.float32(b)).astype(bfloat16)  # ml_dtypes-equivalent bf16 add
                a16 = int(np.float32(a).astype(bfloat16).view(np.uint16))
                b16 = int(np.float32(b).astype(bfloat16).view(np.uint16))
                s16 = int(s.view(np.uint16))
                # NaN policy: any NaN result canonicalized to 0x7FC0 for comparison
                if (s16 & 0x7F80) == 0x7F80 and (s16 & 0x7F):
                    s16 = (s16 & 0x8000) | 0x7FC0
                f.write(f"{a16:04x} {b16:04x} {s16:04x}\n")
    return len(A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/bf16_vec")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n1 = gen_fp32_to_bf16(os.path.join(args.out, "fp32_to_bf16.mem"), args.seed)
    n2 = gen_int_to_bf16(os.path.join(args.out, "int_to_bf16.mem"), args.seed + 1)
    n3 = gen_bf16_add(os.path.join(args.out, "bf16_add.mem"), args.seed + 2)
    print(f"bf16_vectors: fp32_to_bf16={n1} int_to_bf16={n2} bf16_add={n3} -> {args.out}")


if __name__ == "__main__":
    main()
