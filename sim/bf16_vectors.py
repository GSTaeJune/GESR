"""ml_dtypes oracle: emit $readmemh vector files for the bf16 primitive TBs.

Each line is space-separated hex words: inputs then expected output. TBs read
with $readmemh into a reg array and split fields by bit-slicing. ASCII-only.
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
    # the golden's int->bf16 then exponent-shift. scale is the 9-bit combined value.
    rng = np.random.default_rng(seed)
    rows = []
    # realizable block_int magnitudes (|.| < 2^20) and E8M0-derived combined scales.
    ints = list(rng.integers(-(2**19), 2**19, 4000)) + [0, 1, -1, 127, -128,
             2**19 - 1, -(2**19), 255, 256, 257, 2**15, 2**15 + 1]
    # combined scale = e_a + e_b - impl_a - impl_b (still a 9-bit signed value fed as
    # the RMW `scale` port, centered at 127). Sample a wide range incl. subnormal-inducing.
    scales = list(rng.integers(40, 200, 60)) + [127, 0, 1, 254, 60, 62, 70, 100, 150, 200]
    for iv in ints:
        for sc in rng.choice(scales, size=6, replace=False):
            iv = int(iv); sc = int(sc)
            r = np.float32(iv).astype(bfloat16)          # int -> bf16 (RNE), exact int32<2^24
            val = np.ldexp(np.float64(r), sc - 127)      # * 2^(scale-127)
            out = int(np.float32(val).astype(bfloat16).view(np.uint16))
            rows.append((iv & 0xFFFFFFFF, sc & 0x1FF, out))
    with open(path, "w") as f:
        for a, s, o in rows:
            f.write(f"{a:08x} {s:03x} {o:04x}\n")
    return len(rows)


def gen_bf16_add(path, seed=2):
    rng = np.random.default_rng(seed)
    def rbf16(n):
        return (rng.standard_normal(n).astype(np.float32) * rng.uniform(1e-2, 1e2, n).astype(np.float32)).astype(bfloat16)
    A = list(rbf16(200000))
    B = list(rbf16(200000))
    # directed edges appended
    inf = np.float32(np.inf).astype(bfloat16)
    edges = [(inf, -inf), (inf, inf), (bfloat16(1.0), bfloat16(-1.0)),
             (bfloat16(2.0**-133), bfloat16(2.0**-133)), (bfloat16(0.0), bfloat16(-0.0))]
    for a, b in edges:
        A.append(a); B.append(b)
    with open(path, "w") as f:
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
