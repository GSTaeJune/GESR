"""RMW test vector generator (bf16 accumulator datapath). Dumps
(in_GEMM, scale, in_SRAM, expected_out) as hex files.

Semantics mirror RMW.v after the Phase-2b bf16 hard-swap (unchanged by the
2026-07-13 native rewrite -- v3 implements the same numeric contract natively):
  dq  = bf16( bf16(in_GEMM) * 2^(scale-127) )      # int_to_bf16 (r8 + exp shift + flush)
  out = bf16_RNE( in_SRAM + dq )                   # bf16_adder (true bf16 add, RNE)

in_SRAM.hex / expected_out.hex hold 16-bit bf16 words; in_GEMM stays INT32;
scale is the 9-bit signed combined scale. Requires ml_dtypes (same optional
extra as the bf16 golden; lazy import). Project-only file (preserve on
MXP_Tools upstream syncs)."""
import argparse
import os

import numpy as np


def int32_to_hex(x):
    return f"{int(x) & 0xFFFFFFFF:08x}"


def scale9_to_hex(x):
    return f"{int(x) & 0x1FF:03x}"


def bf16_to_hex(x_bf16):
    return f"{int(x_bf16.view(np.uint16)):04x}"


def _expected(iv, sc, sr_f32):
    """bf16 RMW model: int_to_bf16 then bf16 add (both RNE, ml_dtypes).

    Returns (sr_bf16, out_bf16). sr is quantized to bf16 first because the
    SRAM word IS a bf16 -- the TB drives exactly these 16 bits.
    """
    from ml_dtypes import bfloat16
    with np.errstate(over="ignore"):
        r = np.float32(iv).astype(bfloat16)                       # INToRecFN int->bf16
        dq = np.float32(np.ldexp(np.float64(r), sc - 127)).astype(bfloat16)
        sr = np.float32(sr_f32).astype(bfloat16)                  # prior psum is bf16
        out = (np.float32(sr) + np.float32(dq)).astype(bfloat16)  # fp32 add + narrow
    return sr, out


def gen_vectors(n_random=64, seed=0):
    rng = np.random.default_rng(seed)
    v = []
    # Directed cases (kept from the fp32 era; still meaningful in bf16)
    v += [(1, 127, 0.0),
          (-1, 127, 0.0),
          (1, 127, 1.5),
          (0, 127, 3.14),
          (1_000_000, 107, 0.0),
          (127, 0, 0.0),
          (12345, 127, -100.0)]
    # bf16-specific directed
    v += [(12345, -100, 1.5),       # deep underflow: dq flushes to +0
          (-12345, -100, 1.5),      # deep underflow: dq flushes to -0
          (1, -7, 0.0),             # dq = 2^-134 exact tie -> +0 (ties-to-even)
          (3, -8, 0.0),             # dq = 1.5*2^-134 -> rounds up to min subnormal
          (1, 0, 0.0),              # dq = 2^-127 (bf16 subnormal)
          (1, 127, 256.0),          # 256+1=257: tie at ulp=2 grid -> 256 (even)
          (3, 127, 256.0),          # 256+3=259: tie -> 260 (even mantissa)
          (-1, 127, 1.0),           # exact cancellation -> +0
          (2**19 - 1, 255, 0.0),    # scale saturation -> inf
          (0, 40, 5.0)]             # zero passthrough with nonzero scale
    for _ in range(n_random):
        iv = int(rng.integers(-(1 << 20), (1 << 20)))
        sc = int(rng.integers(127 - 30, 127 + 30))
        sr = float(rng.uniform(-1e6, 1e6))
        v.append((iv, sc, sr))
    # full signed 9-bit scale domain (incl. the negative underflow band)
    for _ in range(32):
        iv = int(rng.integers(-(1 << 20), (1 << 20)))
        sc = int(rng.integers(-256, 256))
        sr = float(rng.uniform(-1e3, 1e3))
        v.append((iv, sc, sr))
    out = []
    for iv, sc, sr in v:
        sr_bf16, exp = _expected(iv, sc, sr)
        out.append((iv, sc, sr_bf16, exp))
    return out


def write_vectors(out_dir, vectors):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "in_GEMM.hex"), "w") as f:
        for t in vectors:
            f.write(int32_to_hex(t[0]) + "\n")
    with open(os.path.join(out_dir, "scale.hex"), "w") as f:
        for t in vectors:
            f.write(scale9_to_hex(t[1]) + "\n")
    with open(os.path.join(out_dir, "in_SRAM.hex"), "w") as f:
        for t in vectors:
            f.write(bf16_to_hex(t[2]) + "\n")
    with open(os.path.join(out_dir, "expected_out.hex"), "w") as f:
        for t in vectors:
            f.write(bf16_to_hex(t[3]) + "\n")
    with open(os.path.join(out_dir, "N.txt"), "w") as f:
        f.write(str(len(vectors)) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(prog="rmw-gen")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    vectors = gen_vectors(args.n, args.seed)
    write_vectors(args.out, vectors)
    print(f"rmw-gen: wrote {len(vectors)} vectors to {args.out}/")


if __name__ == "__main__":
    main()
