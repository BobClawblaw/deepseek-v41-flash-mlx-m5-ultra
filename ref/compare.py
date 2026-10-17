"""The native forward (tf-dsv41-forward's out dir) against the CPU reference (run.py teacher --dump-layers, run with
the converted checkpoint's widths: --experts qBgG --dense qBgG).

  python compare.py REF.pt NATIVE_DIR [--case 0]

Per block: the relative error of the four streams and the worst element; then the logits' relative error and
top-1 agreement over every position.
"""

import argparse
import os

import numpy as np
import torch

D, S, V = 5120, 4, 129280


def bf16(path, shape):
    raw = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32).reshape(shape))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("native")
    p.add_argument("--case", type=int, default=0)
    a = p.parse_args()
    ref = torch.load(a.ref)[a.case]
    n = len(ref["ids"])
    if ref["dump"]:
        for i in range(40):
            path = os.path.join(a.native, f"layer{i}.bin")
            if not os.path.exists(path):
                break
            x = bf16(path, (n, S, D))
            r = ref["dump"][f"layer{i}"][0].float()
            rel = ((x - r).norm() / r.norm()).item()
            worst = (x - r).abs().max().item()
            print(f"layer {i:2d}: rel {rel:.2e}  max |d| {worst:.3e}  |ref| {r.abs().max().item():.3e}")
    logits = bf16(os.path.join(a.native, "logits.bin"), (n, V))
    r = ref["logits"]
    rel = ((logits - r).norm(dim=-1) / r.norm(dim=-1))
    top1 = (logits.argmax(-1) == r.argmax(-1)).float()
    print(f"logits: rel mean {rel.mean().item():.3e} max {rel.max().item():.3e}; top-1 {top1.mean().item():.3f} ({int(top1.sum())}/{n})")


if __name__ == "__main__":
    main()
