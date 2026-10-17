"""The CPU reference: greedy replies, teacher-forced logits, per-layer dumps, and the expert requant study.

  python run.py greedy  CKPT "prompt" [--max-new 32] [--thinking]
  python run.py teacher CKPT CASES.json OUT.pt [--experts fp4|q2g64|q3g64|...] [--dump-layers]

`teacher` runs each case's token ids in one forward and saves every position's logits (and, with --dump-layers,
each block's hyper-connection stream) for the engine checks and the requant comparison. `--experts qBgG` swaps
every routed expert for MLX's affine B-bit, group-G quantization of the dequantized fp4 weights, multiplied with
unquantized bf16 activations as the Metal engine will.
"""

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "encoding"))

import model as M  # noqa: E402
from encoding import encode_messages  # noqa: E402
from kernel import dequant_fp4  # noqa: E402
from load import load  # noqa: E402


def mlx_affine(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """tools/convert.py's codes for an exact bf16 weight, dequantized to fp32: mx.quantize of the bf16 values."""
    import mlx.core as mx
    import numpy as np

    wb = mx.array(w.float().numpy()).astype(mx.bfloat16)
    wq, s, b = mx.quantize(wb, group_size=group, bits=bits)
    return torch.from_numpy(np.array(mx.dequantize(wq, s, b, group_size=group, bits=bits).astype(mx.float32)))


def affine_experts(bits: int, group: int):
    """Route model.linear's fp4 expert weights through the converter's affine codes, bf16 activations."""
    plain = M.linear

    def linear(x, weight, bias=None):
        if weight.dtype != torch.float4_e2m1fn_x2:
            return plain(x, weight, bias)
        wd = mlx_affine(dequant_fp4(weight, weight.scale), bits, group)
        return (x.float() @ wd.T).to(torch.get_default_dtype())

    M.linear = linear


def affine_dense(bits: int, group: int):
    """The fp8 projections (all but wo_a, which model.py keeps in bf16) and the head through the converter's
    affine codes, bf16 activations: what the Metal engine multiplies."""
    from kernel import dequant_fp8

    plain = M.linear

    def linear(x, weight, bias=None):
        if weight.dtype != torch.float8_e4m3fn:
            return plain(x, weight, bias)
        w = mlx_affine(dequant_fp8(weight, weight.scale), bits, group)
        return (x.float() @ w.T).to(torch.get_default_dtype())

    M.linear = linear

    def head(self, x, full_logits=False):
        if not hasattr(self, "_affine"):
            self._affine = mlx_affine(self.weight.bfloat16(), bits, group)
        if not full_logits:
            x = x[:, -1]
        return x.float() @ self._affine.T

    M.ParallelHead.forward = head


@torch.inference_mode()
def forward_all(model, ids: torch.Tensor, start_pos: int = 0, dump: dict | None = None):
    """Transformer.forward with every position's logits (and the streams after each block when dumping)."""
    hashes = model.engram_hash(ids, start_pos, None) if model.engram_hash is not None else None
    h = model.embed(ids)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, model.hc_mult)
    if dump is not None:
        dump["embed"] = h[:, :, 0].clone()
    for i, layer in enumerate(model.layers):
        t = time.time()
        if layer.engram is not None:
            h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :], None)
        h, pre_mix = layer(h, start_pos, pre_mix, None)
        if dump is not None:
            dump[f"layer{i}"] = h.clone()
        print(f"  layer {i} {time.time() - t:.1f}s", file=sys.stderr, flush=True)
    h = layer.hc_pre(h, pre_mix)
    return model.head(model.norm(h), full_logits=True)


def encode(tok, prompt: str, thinking: bool) -> list[int]:
    text = encode_messages([{"role": "user", "content": prompt}], thinking_mode="thinking" if thinking else "chat")
    return tok.encode(text, add_special_tokens=False).ids


def greedy(model, tok, ids: list[int], max_new: int) -> list[int]:
    out = []
    x = torch.tensor([ids])
    logits = forward_all(model, x)[:, -1]
    pos = len(ids)
    for _ in range(max_new):
        nxt = int(logits.argmax(-1))
        out.append(nxt)
        print(tok.decode([nxt], skip_special_tokens=False), end="", flush=True)
        if nxt == 1:
            break
        logits = forward_all(model, torch.tensor([[nxt]]), pos)[:, -1]
        pos += 1
    print()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["greedy", "teacher"])
    p.add_argument("ckpt")
    p.add_argument("arg")
    p.add_argument("out", nargs="?")
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--experts", default="fp4")
    p.add_argument("--dense", default="fp8", help="fp8, or qBgG: the converter's affine projections and head")
    p.add_argument("--dump-layers", action="store_true")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.experts != "fp4":
        bits, group = a.experts[1:].split("g")
        affine_experts(int(bits), int(group))
    if a.dense != "fp8":
        bits, group = a.dense[1:].split("g")
        affine_dense(int(bits), int(group))
    t = time.time()
    model, tok, _ = load(a.ckpt, a.max_seq_len)
    print(f"loaded in {time.time() - t:.1f}s", file=sys.stderr)

    if a.mode == "greedy":
        ids = encode(tok, a.arg, a.thinking)
        print(f"{len(ids)} prompt tokens", file=sys.stderr)
        out = greedy(model, tok, ids, a.max_new)
        print(json.dumps({"prompt_ids": ids, "reply_ids": out}))
        return

    cases = json.load(open(a.arg))
    result = []
    for case in cases:
        ids = torch.tensor([case["ids"]])
        dump = {} if a.dump_layers else None
        logits = forward_all(model, ids, 0, dump)
        result.append({"name": case["name"], "ids": case["ids"], "logits": logits[0].float(), "dump": dump})
    torch.save(result, a.out)


if __name__ == "__main__":
    main()
