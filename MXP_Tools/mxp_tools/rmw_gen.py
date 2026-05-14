"""RMW test vector generator. Dump (in_GEMM, scale, in_SRAM, expected_out) as
hex files. expected_out is np.float32(in_SRAM) + np.float32(in_GEMM) * 2^(scale-127),
matching RMW's IEEE-754 RNE semantics."""
import argparse
import os
import struct

import numpy as np


def fp32_to_hex(x):
    return f"{struct.unpack('<I', struct.pack('<f', np.float32(x)))[0]:08x}"


def int32_to_hex(x):
    return f"{int(x) & 0xFFFFFFFF:08x}"


def scale9_to_hex(x):
    return f"{int(x) & 0x1FF:03x}"


def gen_vectors(n_random=64, seed=0):
    rng = np.random.default_rng(seed)
    v = []
    # Directed cases
    v += [(1, 127, np.float32(0.0)),
          (-1, 127, np.float32(0.0)),
          (1, 127, np.float32(1.5)),
          (0, 127, np.float32(3.14)),
          (1_000_000, 107, np.float32(0.0)),
          (127, 0, np.float32(0.0)),
          (12345, 127, np.float32(-100.0))]
    for _ in range(n_random):
        iv = int(rng.integers(-(1 << 20), (1 << 20)))
        sc = int(rng.integers(127 - 30, 127 + 30))
        sr = np.float32(rng.uniform(-1e6, 1e6))
        v.append((iv, sc, sr))
    out = []
    for iv, sc, sr in v:
        dq = np.float32(iv) * np.float32(2.0) ** (np.int32(sc) - 127)
        out.append((iv, sc, sr, np.float32(sr) + dq))
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
            f.write(fp32_to_hex(t[2]) + "\n")
    with open(os.path.join(out_dir, "expected_out.hex"), "w") as f:
        for t in vectors:
            f.write(fp32_to_hex(t[3]) + "\n")
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
