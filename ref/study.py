"""Requant study: what affine experts cost against the shipped mxfp4 ones.

  python study.py replies CKPT OUT_DIR [--max-new 64]       # fp4 greedy replies to study_prompts.json
  python study.py logits  CKPT OUT_DIR --experts fp4|qBgG    # teacher-forced logits over prompt + reply
  python study.py compare OUT_DIR                            # every config against fp4

`logits` runs all cases as one right-padded batch, so each layer's experts are dequantized (and requantized) once.
"""

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run  # noqa: E402
from load import load  # noqa: E402

PAD = 2


def replies(a):
    model, tok, _ = load(a.ckpt, 4096)
    cases = []
    for c in json.load(open(os.path.join(HERE, "study_prompts.json"))):
        ids = run.encode(tok, c["prompt"], thinking=False)
        print(f"== {c['name']} ({len(ids)} prompt tokens)", file=sys.stderr)
        out = run.greedy(model, tok, ids, a.max_new)
        cases.append({"name": c["name"], "prompt_len": len(ids), "ids": ids + out})
    json.dump(cases, open(os.path.join(a.out, "cases.json"), "w"))


def logits(a):
    tag = a.experts
    if a.experts != "fp4":
        bits, group = a.experts[1:].split("g")
        layer_bits = {}
        for part in filter(None, a.layer_bits.split(",")):
            rng, b = part.split(":")
            lo, _, hi = rng.partition("-")
            for i in range(int(lo), int(hi or lo) + 1):
                layer_bits[i] = int(b)
        run.affine_experts(int(bits), int(group), layer_bits)
        if a.layer_bits:
            tag += "-L" + a.layer_bits.replace(":", "b").replace(",", "_")
    if a.dense != "fp8":
        bits, group = a.dense[1:].split("g")
        run.affine_dense(int(bits), int(group))
        tag += "-dense" + a.dense
    cases = json.load(open(os.path.join(a.out, "cases.json")))
    n = max(len(c["ids"]) for c in cases)
    model, _, _ = load(a.ckpt, 4096, batch=len(cases))
    ids = torch.tensor([c["ids"] + [PAD] * (n - len(c["ids"])) for c in cases])
    out = run.forward_all(model, ids)
    keep = [out[i, c["prompt_len"] - 1 : len(c["ids"]) - 1].float() for i, c in enumerate(cases)]
    torch.save({"names": [c["name"] for c in cases], "logits": keep}, os.path.join(a.out, f"logits-{tag}.pt"))


def compare(a):
    base = torch.load(os.path.join(a.out, "logits-fp4.pt"))
    rows = []
    for f in sorted(os.listdir(a.out)):
        if not f.startswith("logits-") or f == "logits-fp4.pt":
            continue
        cfg = torch.load(os.path.join(a.out, f))
        for name, x, y in zip(base["names"], base["logits"], cfg["logits"]):
            p, q = torch.log_softmax(x, -1), torch.log_softmax(y, -1)
            kl = (p.exp() * (p - q)).sum(-1).mean().item()
            top1 = (x.argmax(-1) == y.argmax(-1)).float().mean().item()
            rel = ((x - y).norm(dim=-1) / x.norm(dim=-1)).mean().item()
            rows.append({"config": f[7:-3], "case": name, "positions": x.shape[0], "top1": top1, "kl": kl, "rel": rel})
    for r in rows:
        print(f"{r['config']:8s} {r['case']:5s} n={r['positions']:3d} top1={r['top1']:.3f} kl={r['kl']:.4f} rel={r['rel']:.3f}")
    json.dump(rows, open(os.path.join(a.out, "compare.json"), "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["replies", "logits", "compare"])
    p.add_argument("ckpt", nargs="?")
    p.add_argument("out")
    p.add_argument("--experts", default="fp4")
    p.add_argument("--dense", default="fp8")
    p.add_argument("--layer-bits", default="", help="expert widths by layer over --experts', e.g. 0-9:3")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--threads", type=int, default=24)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    {"replies": replies, "logits": logits, "compare": compare}[a.mode](a)


if __name__ == "__main__":
    main()
