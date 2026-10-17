"""DeepSeek-V4.1-Flash on the CPU: model.py's Transformer with every weight a view of the memory-mapped shards.

The model is built on torch's meta device, then each parameter is bound to its checkpoint tensor without a copy
(fp8, fp4, e8m0, bf16 and fp32 all view the file's bytes), so the 510 GB checkpoint costs only the pages a forward
touches. The few weights model.py keeps in another dtype than the file (wo_a dequantized to bf16, the head and the
compressors' pooling projections in fp32) are converted on load. Runtime buffers (caches, rope tables, the engram
hash state) are then rebuilt on the CPU.
"""

import json
import math
import os
import struct
from glob import glob

import numpy as np
import torch
from tokenizers import Tokenizer

import model as M
from kernel import dequant_fp8
from engram import NgramHashState

_NP = {"F8_E4M3": np.uint8, "F8_E8M0": np.uint8, "I8": np.int8, "BF16": np.int16, "F32": np.float32, "I32": np.int32,
       "I64": np.int64, "U8": np.uint8}
_TORCH = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "BF16": torch.bfloat16}


class Shards:
    """name -> zero-copy torch tensor over the shard's mmap."""

    def __init__(self, ckpt: str):
        self.where: dict[str, tuple[str, dict, int]] = {}
        self.maps: dict[str, np.memmap] = {}
        for path in sorted(glob(os.path.join(ckpt, "*.safetensors"))):
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            header.pop("__metadata__", None)
            for name, info in header.items():
                self.where[name] = (path, info, 8 + n)

    def __contains__(self, name):
        return name in self.where

    def get(self, name: str) -> torch.Tensor:
        path, info, base = self.where[name]
        if path not in self.maps:
            self.maps[path] = np.memmap(path, dtype=np.uint8, mode="r")
        start, end = info["data_offsets"]
        raw = self.maps[path][base + start : base + end]
        arr = raw.view(_NP[info["dtype"]]).reshape(info["shape"])
        t = torch.from_numpy(arr)  # read-only mmap: torch warns, nothing writes these
        return t.view(_TORCH[info["dtype"]]) if info["dtype"] in _TORCH else t


class TokenizerShim:
    """What engram.build_compressed_token_map reads from a transformers tokenizer."""

    def __init__(self, tok: Tokenizer):
        self.backend_tokenizer = tok

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


def model_args(ckpt: str, max_seq_len: int, vision: bool, batch: int = 1) -> M.ModelArgs:
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(os.path.join(here, "inference_config.json")))
    cfg.update(max_batch_size=batch, max_seq_len=max_seq_len)
    if not vision:
        cfg["vision_n_layers"] = 0
    fields = M.ModelArgs.__dataclass_fields__
    return M.ModelArgs(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items() if k in fields})


def _bind(module: torch.nn.Module, key: str, value: torch.Tensor):
    setattr(module, key, torch.nn.Parameter(value, requires_grad=False))


def load(ckpt: str, max_seq_len: int = 4096, vision: bool = False, temperature: float = 0.0, batch: int = 1):
    import warnings

    warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
    torch.set_default_dtype(torch.bfloat16)
    args = model_args(ckpt, max_seq_len, vision, batch)
    args.temperature = temperature
    tok = Tokenizer.from_file(os.path.join(ckpt, "tokenizer.json"))
    shards = Shards(ckpt)

    with torch.device("meta"):
        model = M.Transformer.__new__(M.Transformer)
        # the engram hash state holds real data, so it is built outside the meta context
        layout = M.EngramLayout.from_args(args)
        _orig = M.NgramHashState
        M.NgramHashState = lambda *a, **k: None
        try:
            M.Transformer.__init__(model, args, None)
        finally:
            M.NgramHashState = _orig
    model.engram_hash = NgramHashState(args, layout, TokenizerShim(tok)) if layout is not None else None

    missing = []
    for mod_name, mod in model.named_modules():
        for key, p in list(mod.named_parameters(recurse=False)):
            name = f"{mod_name}.{key}" if mod_name else key
            if name not in shards:
                missing.append(name)
                continue
            t = shards.get(name)
            if p.dtype == torch.float4_e2m1fn_x2:
                t = t.view(torch.float4_e2m1fn_x2)
            elif name.endswith("wo_a.weight"):
                t = dequant_fp8(t, shards.get(name.replace("weight", "scale"))).bfloat16()  # exact: e4m3 x 2^k fits bf16
            elif p.dtype != t.dtype:
                t = t.to(p.dtype)
            assert tuple(t.shape) == tuple(p.shape), (name, t.shape, p.shape)
            _bind(mod, key, t)
        if isinstance(mod, M.Linear) and mod.scale is not None:
            mod.weight.scale = mod.scale
    assert not missing, f"not in the checkpoint: {missing[:20]}"

    for mod in model.modules():
        for key, b in list(mod.named_buffers(recurse=False)):
            if not b.is_meta:
                continue
            if key == "score_state":
                new = torch.full(b.shape, -math.inf, dtype=b.dtype)
            elif key == "freqs_cis":
                new = _freqs(args, mod.compress_ratio)
            else:
                new = torch.zeros(b.shape, dtype=b.dtype)
            mod.register_buffer(key, new, persistent=False)
    return model, tok, args


def _freqs(args, compress_ratio):
    if compress_ratio:
        original, theta = args.original_seq_len, args.compress_rope_theta
    else:
        original, theta = 0, args.rope_theta
    M.precompute_freqs_cis.cache_clear()
    return M.precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, original, theta, args.rope_factor, args.beta_fast, args.beta_slow
    )
