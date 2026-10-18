"""The whole-model gate's reference chains: model.py over a converted checkpoint's own affine codes, in one width.

  python gate.py CKPT CONVERTED IDS.json OUT.npy --width bf16|fp32|fp64

CKPT is the original release (tokenizer, engram tables, the unquantized tensors); CONVERTED is tools/convert.py's
output, whose stored codes replace every fp8/fp4 projection, wo_a and the head, so both sides read the same weights.
bf16 is the reference's own arithmetic (bf16 activations, fp32 projection sums), the engine's golden; fp32 and fp64
keep the equations (the cache's fp8/fp4 round trips included) with every activation and sum in that width.
Writes every position's logits as float64 [n, vocab].
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "encoding"))

import model as M  # noqa: E402
from load import _bind, _freqs, load  # noqa: E402

QUANT = (torch.float8_e4m3fn, torch.float4_e2m1fn_x2)
NP = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U8": np.uint8}


class Converted:
    """The converted checkpoint's affine tensors, dequantized on demand (mx.dequantize of the stored codes)."""

    def __init__(self, d: str):
        self.d = d
        self.where = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        q = json.load(open(os.path.join(d, "config.json")))["quantization"]
        self.default = (q["bits"], q["group_size"])
        self.widths = {k: (v["bits"], v["group_size"]) for k, v in q.items() if isinstance(v, dict)}
        self.headers, self.maps = {}, {}

    def raw(self, name: str) -> np.ndarray:
        f = os.path.join(self.d, self.where[name])
        if f not in self.headers:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.headers[f] = (json.loads(fh.read(n)), 8 + n)
            self.maps[f] = np.memmap(f, dtype=np.uint8, mode="r")
        header, base = self.headers[f]
        info = header[name]
        lo, hi = info["data_offsets"]
        return self.maps[f][base + lo : base + hi].view(NP[info["dtype"]]).reshape(info["shape"]), info["dtype"]

    def weight(self, base: str, expert: int | None = None) -> torch.Tensor:
        import mlx.core as mx

        bits, group = self.widths.get(base, self.default)
        parts = []
        for suffix in (".weight", ".scales", ".biases"):
            a, dtype = self.raw(base + suffix)
            a = np.ascontiguousarray(a[expert] if expert is not None else a)
            t = mx.array(a)
            parts.append(t.view(mx.bfloat16) if dtype == "BF16" else t)
        w = mx.dequantize(*parts, group_size=group, bits=bits).astype(mx.float32)
        return torch.from_numpy(np.array(w))


def install(conv: Converted, acc: torch.dtype):
    """Every fp8/fp4 projection and the head through the converted codes, summed in `acc`."""

    def linear(x, weight, bias=None):
        assert bias is None
        if weight.dtype not in QUANT:
            return x.to(acc) @ weight.to(acc).T if acc == torch.float64 else plain(x, weight)
        name = weight.tf_name
        if ".ffn.experts." in name:
            ffn, _, e, w = name.rsplit(".", 3)
            wd = conv.weight(f"{ffn}.experts.{w}", int(e))
        else:
            wd = conv.weight(name)
        return (x.to(acc) @ wd.to(acc).T).to(torch.get_default_dtype())

    plain = M.linear
    M.linear = linear

    def head(self, x, full_logits=False):
        if not hasattr(self, "_affine"):
            self._affine = conv.weight("head").to(acc)
        if not full_logits:
            x = x[:, -1]
        return x.to(acc) @ self._affine.T

    M.ParallelHead.forward = head


def widen(model, args, w: torch.dtype):
    """Every float parameter and buffer to `w`, and every later bf16/fp32 cast in model.py and kernel.py with it."""
    floats = (torch.bfloat16, torch.float16, torch.float32, torch.float64)
    for mod in model.modules():
        for key, p in list(mod.named_parameters(recurse=False)):
            if p.dtype in floats:
                _bind(mod, key, p.data.to(w))
        for key, b in list(mod.named_buffers(recurse=False)):
            if b.dtype in floats:
                mod.register_buffer(key, b.to(w), persistent=False)
    torch.set_default_dtype(w)
    torch.Tensor.float = lambda self, *a, **k: self.to(w)
    torch.Tensor.bfloat16 = lambda self, *a, **k: self.to(w)
    torch.bfloat16 = w
    torch.float32 = w
    for mod in model.modules():  # the rotary tables again, in the new width
        if "freqs_cis" in dict(mod.named_buffers(recurse=False)):
            mod.register_buffer("freqs_cis", _freqs(args, mod.compress_ratio), persistent=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("converted")
    p.add_argument("ids")
    p.add_argument("out")
    p.add_argument("--width", choices=["bf16", "fp32", "fp64"], required=True)
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    w = {"bf16": torch.float32, "fp32": torch.float32, "fp64": torch.float64}[a.width]
    conv = Converted(a.converted)
    install(conv, w)
    t = time.time()
    ids = json.load(open(a.ids))
    model, _, args = load(a.ckpt, max(64, len(ids) + 8))
    for name, mod in model.named_modules():
        weight = getattr(mod, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.dtype in QUANT:
            weight.tf_name = name
    for name, mod in model.named_modules():
        if name.endswith(".attn") and hasattr(mod, "wo_a"):
            wo_a = conv.weight(f"{name}.wo_a").reshape(mod.wo_a.weight.shape)
            _bind(mod.wo_a, "weight", wo_a.to(mod.wo_a.weight.dtype))
    if a.width != "bf16":
        widen(model, args, w)
    print(f"loaded in {time.time() - t:.1f}s ({a.width})", file=sys.stderr)
    from run import forward_all

    logits = forward_all(model, torch.tensor([ids]), 0)
    np.save(a.out, logits[0].to(torch.float64).numpy())
    print(f"done in {time.time() - t:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
