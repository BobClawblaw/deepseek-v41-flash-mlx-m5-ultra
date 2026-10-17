"""Plain-torch stand-ins for DeepSeek's tilelang kernels (kernel_tilelang.py), same names and signatures.

Run on the CPU, so model.py works on a Mac. The quantizers copy the tilelang rounding exactly: power-of-two (ue8m0)
scales from ceil(log2(amax / max)), fp8 e4m3 and fp4 e2m1 round to nearest even, values clamped before the cast.
GEMMs dequantize both sides and multiply in fp32. Attention is an exact fp32 softmax with the sink rather than the
kernel's blocked online softmax, which rounds its probabilities to bf16 before the value product.
"""

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
# e2m1 rounding: thresholds between neighbours, and whether a tie there rounds up (to the even code)
_FP4_BOUNDS = ((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False))
_FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _pow2_scale(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    """2 ** ceil(log2(amax * max_inv)), from the fp32 bits as fast_round_scale does."""
    v = (amax * max_inv).float()
    bits = v.view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).int()
    return ((exp + 127) << 23).view(torch.float32)


def fp4_round(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values already inside [-6, 6] to the e2m1 grid, nearest even."""
    a = x.abs()
    idx = torch.zeros_like(a, dtype=torch.long)
    for bound, tie_up in _FP4_BOUNDS:
        idx += (a >= bound) if tie_up else (a > bound)
    return torch.copysign(_FP4_GRID[idx], x)


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    return x.float().unflatten(-1, (-1, block))


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_scale(amax, 1 / FP8_MAX) if scale_fmt is not None else amax / FP8_MAX
    y = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((y.float() * s).flatten(-2).to(x.dtype))
        return x
    return y.flatten(-2), s.squeeze(-1).to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6 * 2**-9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        s = _pow2_scale(amax.clamp_min(6 * 2**-126), 1 / FP4_MAX)
    y = fp4_round((xb / s).clamp(-FP4_MAX, FP4_MAX))
    assert inplace, "only the fused quantize-dequantize form is used"
    x.copy_((y * s).flatten(-2).to(x.dtype))
    return x


def dequant_fp8(w: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    """fp8 weight with one scale per block x block tile, as fp32."""
    o, i = w.shape
    sf = s.float().repeat_interleave(block, 0)[:o].repeat_interleave(block, 1)[:, :i]
    return w.float() * sf


def dequant_fp4(w: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Packed e2m1 weight ([out, in/2] bytes, low nibble first) with one scale per 32 along `in`, as fp32."""
    b = w.view(torch.uint8)
    vals = torch.stack([FP4_TABLE[(b & 0x0F).long()], FP4_TABLE[(b >> 4).long()]], dim=-1).flatten(-2)
    return vals * s.float().repeat_interleave(block, -1)


def _act(x, s, block):
    return x.float() * s.float().repeat_interleave(block, -1)


def fp8_gemm(x, s, w, ws, scale_dtype, block_size=32):
    return (_act(x, s, block_size) @ dequant_fp8(w, ws, block_size).T).to(torch.get_default_dtype())


def fp4_gemm(x, s, w, ws, scale_dtype, act_block_size=32):
    return (_act(x, s, act_block_size) @ dequant_fp4(w, ws).T).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q [b, m, h, d], kv [b, n, d], topk_idxs [b, m, k] (-1 = nothing); one shared key/value head."""
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    sink = attn_sink.float()
    for bi in range(b):
        for start in range(0, m, 64):
            idx = topk_idxs[bi, start : start + 64].long()
            valid = idx >= 0
            g = kv[bi].float()[idx.clamp_min(0)]  # [r, k, d]
            g = g * valid.unsqueeze(-1)
            scores = torch.einsum("rhd,rkd->rhk", q[bi, start : start + 64].float(), g) * softmax_scale
            scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
            mx = scores.amax(-1, keepdim=True).clamp_min(-1e30)
            p = torch.exp(scores - mx)
            denom = p.sum(-1, keepdim=True) + torch.exp(sink.view(1, h, 1) - mx)
            out[bi, start : start + 64] = (torch.einsum("rhk,rkd->rhd", p, g) / denom).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (m[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
