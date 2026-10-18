"""The whole-model gate's numbers from ref/gate.py's chains and tf-dsv41-forward's logits.bin.

  python gate_compare.py GATE_DIR

NRMS(a, b) = ||a - b|| / ||b|| over every position's logits; the floor F = NRMS(fp32, fp64); each engine run's
NRMS against fp64 and its ratio to F, argmax agreement with fp64 by position, and the same against the bf16 chain.
"""

import os
import sys

import numpy as np

V = 129280


def engine(path: str) -> np.ndarray:
    raw = np.fromfile(os.path.join(path, "logits.bin"), dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(-1, V)


def nrms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)) / np.sqrt(np.sum(b**2)))


def main():
    d = sys.argv[1]
    ref = {w: np.load(os.path.join(d, f"ref-{w}.npy")) for w in ("bf16", "fp32", "fp64")}
    runs = {"engine, one call": engine(os.path.join(d, "engine-all")), "engine, a row a call": engine(os.path.join(d, "engine-row"))}
    f = nrms(ref["fp32"], ref["fp64"])
    print(f"floor F = NRMS(fp32, fp64) = {f:.6%}; 1.1 F = {1.1 * f:.6%}")
    top = ref["fp64"].argmax(-1)
    for name, x in [("bf16 reference", ref["bf16"])] + list(runs.items()):
        agree = int((x.argmax(-1) == top).sum())
        print(f"{name}: NRMS vs fp64 {nrms(x, ref['fp64']):.6%} ({nrms(x, ref['fp64']) / f:.2f} F), "
              f"max |d| {np.abs(x - ref['fp64']).max():.4f}, argmax {agree}/{len(top)}; "
              f"vs bf16 reference {nrms(x, ref['bf16']):.6%}, argmax {int((x.argmax(-1) == ref['bf16'].argmax(-1)).sum())}/{len(top)}")
        for i in np.nonzero(x.argmax(-1) != top)[0]:
            s = np.sort(ref["fp64"][i])
            print(f"  position {i}: {x[i].argmax()} vs {top[i]} (fp64 margin top1-top2 {s[-1] - s[-2]:.4f})")
    print(f"per position, engine one call vs fp64: {[round(nrms(runs['engine, one call'][i], ref['fp64'][i]) * 100, 3) for i in range(len(top))]} %")
    print(f"per position, bf16 reference vs fp64: {[round(nrms(ref['bf16'][i], ref['fp64'][i]) * 100, 3) for i in range(len(top))]} %")
    print(f"per position, fp32 vs fp64: {[round(nrms(ref['fp32'][i], ref['fp64'][i]) * 100, 4) for i in range(len(top))]} %")


if __name__ == "__main__":
    main()
