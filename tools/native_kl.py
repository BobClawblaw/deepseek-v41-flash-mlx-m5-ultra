"""The native engine against the study's fp4 reference: teacher-forced logits over the study cases' replies, top-1
agreement and mean KL(fp4 || native) per case, as ref/study.py compare reports them for the CPU reference.

  python tools/native_kl.py MODEL_DIR [--forward BIN] [--study work/study] [--out work/native-kl/NAME]
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ref"))
from compare import bf16  # noqa: E402

VOCAB = 129280


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--forward", default=os.path.expanduser("~/tf-checks/deepseek-v41/zig-out/bin/tf-dsv41-forward"))
    p.add_argument("--study", default=os.path.join(HERE, "..", "work", "study"))
    p.add_argument("--out", default=os.path.join(HERE, "..", "work", "native-kl", "run"))
    a = p.parse_args()
    cases = json.load(open(os.path.join(a.study, "cases.json")))
    ref = torch.load(os.path.join(a.study, "logits-fp4.pt"))
    rows, kls, tops = [], [], []
    for case, base in zip(cases, ref["logits"]):
        d = os.path.join(a.out, case["name"])
        os.makedirs(d, exist_ok=True)
        ids = np.array(case["ids"][:-1], dtype=np.uint32)
        np.save(os.path.join(d, "ids.npy"), ids)
        subprocess.run([a.forward, os.path.expanduser(a.model), os.path.join(d, "ids.npy"), d, "1"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        got = bf16(os.path.join(d, "logits.bin"), (len(ids), VOCAB)).float()[case["prompt_len"] - 1 :]
        lb, lg = base.log_softmax(-1), got.log_softmax(-1)
        kl = (lb.exp() * (lb - lg)).sum(-1).mean().item()
        top = (base.argmax(-1) == got.argmax(-1)).float().mean().item()
        kls.append(kl), tops.append(top)
        rows.append({"name": case["name"], "n": len(got), "top1": round(top, 4), "kl": round(kl, 4)})
        print(f"{case['name']:6s} n={len(got):3d} top1={top:.3f} kl={kl:.4f}", flush=True)
    print(f"mean   top1={sum(tops) / len(tops):.3f} kl={sum(kls) / len(kls):.4f}")
    json.dump(rows, open(os.path.join(a.out, "kl.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
