"""DeepSeek-V4.1-Flash (HF, fp8 + mxfp4) -> the TensorFold `deepseek_v41` checkpoint.

  python tools/convert.py HF_DIR OUT_DIR [--expert-bits 2] [--expert-group 64] [--dense-bits 8]
                          [--layer-bits 0-1:3,39:3] [--engram link|copy]

Output, all little-endian safetensors:
- `model-NNNNN.safetensors` + `model.safetensors.index.json`, one shard per block (plus `model-head`, `model-mtp`,
  `model-vision`): DeepSeek's tensor names. Projections are MLX affine (`.weight` u32 codes, `.scales`/`.biases`
  bf16); routed experts are stacked per layer as `layers.N.ffn.experts.w{1,2,3}` [E, out, in]. Norms, routers,
  compressors, the indexers' key and weight projections and the hyper-connection tensors keep their checkpoint
  dtypes (bf16 / f32), wo_a included as bf16 [8192, 4096] — the reference's own dtypes.
- `engram/layer{L}.safetensors`: `rows` u8 [N, 264], each row the table's 256 fp8 e4m3 bytes then its 8 e8m0
  scales (one per 32), so one lookup reads one span; `engram/hash.safetensors`: the hash state engram.py derives
  from the tokenizer (token_map, multipliers, primes, offsets). These files are mapped, never loaded.
- `chat_template.jinja`: encoding.py's encode_messages as a chat template (template/check.py: byte-identical).
- `config.json`: the checkpoint's config flattened (`model_type` deepseek_v41), with MLX's `quantization` block
  (global width = dense width, a key per module that differs).

Affine codes come from mx.quantize on the exact bf16 dequantization (fp8 x 2^k and fp4 x 2^k are exact in bf16).
"""

import argparse
import json
import os
import shutil
import struct
import sys
import time

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ref"))

FP4 = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=mx.float32)
DENSE_AFFINE = ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b", "attn.indexer.wq_b", "ffn.shared_experts.w1",
                "ffn.shared_experts.w2", "ffn.shared_experts.w3", "engram.wkv", "main_proj")


class Source:
    """The HF shards by tensor name, as numpy views of the mapped files."""

    NP = {"F8_E4M3": np.uint8, "F8_E8M0": np.uint8, "I8": np.uint8, "U8": np.uint8, "BF16": np.uint16,
          "F32": np.float32, "I64": np.int64, "I32": np.int32}

    def __init__(self, d):
        self.dir = d
        self.where = {}
        self.maps = {}
        for name, f in json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].items():
            self.where[name] = f
        self.headers = {}
        for f in sorted(set(self.where.values())):
            with open(os.path.join(d, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            h.pop("__metadata__", None)
            self.headers[f] = (h, 8 + n)

    def info(self, name):
        h, _ = self.headers[self.where[name]]
        return h[name]

    def np(self, name):
        f = self.where[name]
        h, base = self.headers[f]
        if f not in self.maps:
            self.maps[f] = np.memmap(os.path.join(self.dir, f), dtype=np.uint8, mode="r")
        i = h[name]
        a, b = i["data_offsets"]
        return self.maps[f][base + a : base + b].view(self.NP[i["dtype"]]).reshape(i["shape"])

    def names(self, prefix=""):
        return [n for n in self.where if n.startswith(prefix)]


def e8m0(a: np.ndarray) -> mx.array:
    return mx.power(2.0, mx.array(a.astype(np.int32)) - 127).astype(mx.float32)


def bf16(a: np.ndarray) -> mx.array:
    return mx.array(a.view(np.uint16)).view(mx.bfloat16)


def fp8_e4m3(a: np.ndarray) -> mx.array:
    """e4m3fn bytes to fp32 (no infinities; 0x7f/0xff are NaN and never appear in weights)."""
    u = mx.array(a.astype(np.uint32))
    sign = mx.where((u & 0x80) != 0, -1.0, 1.0)
    e = (u >> 3) & 0xF
    m = (u & 0x7).astype(mx.float32)
    normal = (1.0 + m / 8.0) * mx.power(2.0, e.astype(mx.float32) - 7.0)
    sub = (m / 8.0) * (2.0**-6)
    return sign * mx.where(e == 0, sub, normal)


def dequant_fp8(w: np.ndarray, s: np.ndarray, block=32) -> mx.array:
    o, i = w.shape
    sc = e8m0(s)
    sc = mx.repeat(mx.repeat(sc, block, axis=0)[:o], block, axis=1)[:, :i]
    return (fp8_e4m3(w) * sc).astype(mx.bfloat16)


def dequant_fp4(w: np.ndarray, s: np.ndarray, block=32) -> mx.array:
    """[..., out, in/2] packed e2m1 (low nibble first), scales [..., out, in/32] -> bf16 [..., out, in]."""
    u = mx.array(w.view(np.uint8))
    vals = mx.stack([FP4[(u & 0x0F).astype(mx.int32)], FP4[(u >> 4).astype(mx.int32)]], axis=-1)
    vals = vals.reshape(*w.shape[:-1], w.shape[-1] * 2)
    return (vals * mx.repeat(e8m0(s), block, axis=-1)).astype(mx.bfloat16)


def quant(w: mx.array, bits: int, group: int, out: dict, name: str):
    q, s, b = mx.quantize(w, group_size=group, bits=bits)
    out[name + ".weight"], out[name + ".scales"], out[name + ".biases"] = q, s, b
    mx.eval(q, s, b)


def parse_layer_bits(spec: str) -> dict[int, int]:
    out = {}
    for part in filter(None, spec.split(",")):
        rng, bits = part.split(":")
        lo, _, hi = rng.partition("-")
        for i in range(int(lo), int(hi or lo) + 1):
            out[i] = int(bits)
    return out


class Writer:
    def __init__(self, out_dir):
        self.out = out_dir
        self.weight_map = {}
        self.bytes = 0
        index = os.path.join(out_dir, "model.safetensors.index.json")
        if os.path.exists(index):  # a partial run (--only) replaces its shards' entries, keeps the others
            self.weight_map = json.load(open(index))["weight_map"]

    def save(self, shard: str, tensors: dict):
        path = os.path.join(self.out, f"model-{shard}.safetensors")
        mx.save_safetensors(path, tensors, metadata={"format": "mlx"})
        name = os.path.basename(path)
        self.weight_map = {k: v for k, v in self.weight_map.items() if v != name}
        for n in tensors:
            self.weight_map[n] = name

    def finish(self):
        self.bytes = sum(os.path.getsize(os.path.join(self.out, f)) for f in set(self.weight_map.values()))
        json.dump({"metadata": {"total_size": self.bytes}, "weight_map": self.weight_map},
                  open(os.path.join(self.out, "model.safetensors.index.json"), "w"), indent=1)


def block(src: Source, prefix: str, n_experts: int, expert_bits: int, a, quant_cfg: dict) -> dict:
    """One block's (or DSpark stage's) tensors: affine projections, stacked experts, the rest as stored."""
    t = {}
    for name in sorted(src.names(prefix)):
        rest = name[len(prefix):]
        if ".engram.embed." in name or ".ffn.experts." in name or rest.endswith(".scale"):
            continue
        mod = rest.rsplit(".", 1)[0]
        info = src.info(name)
        if rest.endswith(".weight") and any(mod.endswith(d) for d in DENSE_AFFINE):
            w = dequant_fp8(src.np(name), src.np(name[: -len("weight")] + "scale"))
            quant(w, a.dense_bits, a.dense_group, t, prefix + mod)
        elif rest == "attn.wo_a.weight":
            t[name] = dequant_fp8(src.np(name), src.np(prefix + "attn.wo_a.scale"))
        elif info["dtype"] == "BF16":
            t[name] = bf16(src.np(name))
        elif info["dtype"] == "F32":
            t[name] = mx.array(src.np(name))
        elif info["dtype"] == "F8_E4M3":  # an fp8 tensor outside the affine list: keep it exact in bf16
            t[name] = dequant_fp8(src.np(name), src.np(name[: -len("weight")] + "scale"))
        else:
            raise SystemExit(f"unhandled tensor {name} {info['dtype']}")
    for w in ("w1", "w2", "w3"):
        codes, scales, biases = [], [], []
        for e0 in range(0, n_experts, 32):
            ws = np.stack([src.np(f"{prefix}ffn.experts.{e}.{w}.weight") for e in range(e0, min(e0 + 32, n_experts))])
            ss = np.stack([src.np(f"{prefix}ffn.experts.{e}.{w}.scale") for e in range(e0, min(e0 + 32, n_experts))])
            q, s, b = mx.quantize(dequant_fp4(ws, ss), group_size=a.expert_group, bits=expert_bits)
            mx.eval(q, s, b)
            codes.append(q), scales.append(s), biases.append(b)
        base = f"{prefix}ffn.experts.{w}"
        t[base + ".weight"] = mx.concatenate(codes)
        t[base + ".scales"] = mx.concatenate(scales)
        t[base + ".biases"] = mx.concatenate(biases)
        if expert_bits != a.dense_bits or a.expert_group != a.dense_group:
            quant_cfg[base] = {"bits": expert_bits, "group_size": a.expert_group}
    return t


def engram(src: Source, out_dir: str, layer: int, chunk_rows: int = 1 << 20):
    """layers.L.engram.embed.{weight,scale} -> engram/layer{L}.safetensors rows of 256 + 8 bytes."""
    w = src.np(f"layers.{layer}.engram.embed.weight")
    s = src.np(f"layers.{layer}.engram.embed.scale")
    rows, dim = w.shape
    width = dim + s.shape[1]
    header = json.dumps({"rows": {"dtype": "U8", "shape": [rows, width], "data_offsets": [0, rows * width]}}).encode()
    header += b" " * (-(len(header)) % 8)
    path = os.path.join(out_dir, "engram", f"layer{layer}.safetensors")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for r0 in range(0, rows, chunk_rows):
            r1 = min(rows, r0 + chunk_rows)
            f.write(np.concatenate([w[r0:r1], s[r0:r1]], axis=1).tobytes())
    return rows, width


def engram_hash(hf_dir: str, out_dir: str, cfg: dict):
    import torch
    from tokenizers import Tokenizer

    from engram import EngramLayout, NgramHashState, build_compressed_token_map, compute_hash_multipliers
    from load import TokenizerShim

    class Args:
        pass

    args = Args()
    for k in ("engram_layer_ids", "engram_num_embeddings", "engram_max_ngram_size", "engram_vocab_size",
              "engram_n_heads", "engram_head_dim", "engram_compressed_vocab_size"):
        setattr(args, k, cfg[k])
    layout = EngramLayout.from_args(args)
    tok = TokenizerShim(Tokenizer.from_file(os.path.join(hf_dir, "tokenizer.json")))
    token_map, size = build_compressed_token_map(tok)
    assert size == cfg["engram_compressed_vocab_size"], size
    flat = [[p for per in layer for p in per] for layer in layout.primes]
    offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64)
    mult = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, size).numpy().astype(np.int64)
    tensors = {
        "token_map": mx.array(np.array(token_map, dtype=np.int32)),
        "multipliers": mx.array(mult),
        "primes": mx.array(np.array(layout.primes, dtype=np.int64)),
        "offsets": mx.array(offsets),
        "pad": mx.array(np.array([token_map[cfg["engram_pad_token_id"]]], dtype=np.int32)),
    }
    mx.save_safetensors(os.path.join(out_dir, "engram", "hash.safetensors"), tensors)
    del torch, NgramHashState


def main():
    p = argparse.ArgumentParser()
    p.add_argument("hf")
    p.add_argument("out")
    p.add_argument("--expert-bits", type=int, default=2)
    p.add_argument("--expert-group", type=int, default=64)
    p.add_argument("--dense-bits", type=int, default=8)
    p.add_argument("--dense-group", type=int, default=64)
    p.add_argument("--layer-bits", default="", help="expert widths by layer, e.g. 0-1:3,39:3")
    p.add_argument("--mtp-expert-bits", type=int, default=None)
    p.add_argument("--only", default="", help="debug: comma-separated parts (blocks,head,mtp,vision,engram,hash)")
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "engram"), exist_ok=True)
    src = Source(a.hf)
    hf_cfg = json.load(open(os.path.join(a.hf, "config.json")))
    text = hf_cfg["text_config"]
    parts = set(a.only.split(",")) if a.only else {"blocks", "head", "mtp", "vision", "engram", "hash"}
    layer_bits = parse_layer_bits(a.layer_bits)
    quant_cfg = {"group_size": a.dense_group, "bits": a.dense_bits, "mode": "affine"}
    wr = Writer(a.out)
    t0 = time.time()

    if "blocks" in parts:
        for i in range(text["num_hidden_layers"]):
            bits = layer_bits.get(i, a.expert_bits)
            wr.save(f"{i:05d}", block(src, f"layers.{i}.", text["n_routed_experts"], bits, a, quant_cfg))
            print(f"layer {i} ({bits}-bit experts) {time.time() - t0:.0f}s", flush=True)
    if "mtp" in parts:
        for i in range(text["num_nextn_predict_layers"]):
            bits = a.mtp_expert_bits or a.expert_bits
            wr.save(f"mtp{i}", block(src, f"mtp.{i}.", text["dspark_n_routed_experts"], bits, a, quant_cfg))
        print(f"dspark {time.time() - t0:.0f}s", flush=True)
    if "head" in parts:
        t = {"embed.weight": bf16(src.np("embed.weight")), "norm.weight": bf16(src.np("norm.weight"))}
        quant(bf16(src.np("head.weight")), a.dense_bits, a.dense_group, t, "head")
        wr.save("head", t)
    if "vision" in parts:
        # the tower's and the aligner's projections affine at the dense width (the tensor units take them); the
        # patch embedding (588 inputs, no group of 64 divides it), biases, norms and the span embeddings as stored
        t = {}
        for n in src.names("vision.") + src.names("aligner.") + ["image_start", "image_end", "image_newline"]:
            info = src.info(n)
            v = bf16(src.np(n)) if info["dtype"] == "BF16" else mx.array(src.np(n))
            mod = n.rsplit(".", 1)[0]
            if n.endswith(".weight") and mod.rsplit(".", 1)[-1] in ("wqkv", "wo", "w1", "w2") and v.ndim == 2:
                quant(v, a.dense_bits, a.dense_group, t, mod)
            else:
                t[n] = v
        wr.save("vision", t)
    if parts & {"blocks", "head", "mtp", "vision"}:
        wr.finish()
    if "engram" in parts:
        for layer in text["engram_layer_ids"]:
            rows, width = engram(src, a.out, layer)
            print(f"engram layer {layer}: {rows} rows of {width} bytes {time.time() - t0:.0f}s", flush=True)
    if "hash" in parts:
        engram_hash(a.hf, a.out, text)

    cfg = dict(text)
    cfg["model_type"] = "deepseek_v41"
    cfg["architectures"] = hf_cfg["architectures"]
    for k in ("bos_token_id", "eos_token_id", "pad_token_id", "image_token_id"):
        cfg[k] = hf_cfg[k]
    cfg["vision_config"] = hf_cfg["vision_config"]
    old_cfg = os.path.join(a.out, "config.json")
    if "blocks" not in parts and os.path.exists(old_cfg):  # a partial run keeps the experts' widths it did not write
        kept = json.load(open(old_cfg))["quantization"]
        quant_cfg = {**kept, **{k: v for k, v in quant_cfg.items() if k not in ("bits", "group_size", "mode")}}
    cfg["quantization"] = quant_cfg
    cfg["engram_files"] = {str(layer): f"engram/layer{layer}.safetensors" for layer in text["engram_layer_ids"]}
    cfg["source"] = {"repo": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "expert_bits": a.expert_bits,
                     "layer_bits": a.layer_bits, "dense_bits": a.dense_bits}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "LICENSE"):
        if os.path.exists(os.path.join(a.hf, f)):
            shutil.copyfile(os.path.join(a.hf, f), os.path.join(a.out, f))
    # the checkpoint ships its prompt format as Python (encoding/encoding.py); template/check.py holds this to it
    shutil.copyfile(os.path.join(HERE, "..", "template", "chat_template.jinja"), os.path.join(a.out, "chat_template.jinja"))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
