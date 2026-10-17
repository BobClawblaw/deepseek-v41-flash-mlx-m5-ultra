# DeepSeek-V4.1-Flash-1017 on one M5 Ultra (256 GB), TensorFold native Metal

Work plan and status. The README is written once the recipe serves; until then this file is the source of truth.

## The model (deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e10, MIT)

- 40 blocks, hidden 5120, hyper-connections x4 (Sinkhorn 20), vocab 129,280, window 1,048,576 (YaRN x16 on 65,536).
- Attention: 64 heads of 512 over one shared KV head, rope on the last 64 dims (inverse rope on the output), 128-token
  sliding window with sinks, grouped low-rank output (8 groups, rank 1024). Layers 0-1 window only; 2-19 add KV
  compressed 2:1; 20-39 add compressed 1:1. Only layers 2, 8, 14, 20 compress (the rest read their cache); indexers
  (32 heads of 128, top 512) run on 2, 8, 14, 20, 24, 28, 32, 36; layer 20 also picks 2,048 candidate blocks of 8 that
  the later indexers score inside.
- MoE: 384 routed experts (top 6, sqrt-softplus, bias-steered, route scale 1.5), one shared expert, SwiGLU clamp 10.
  Experts ship mxfp4 (e8m0 per 32); everything else fp8 e4m3 with e8m0 per 32x32 block.
- Engram at layers 1 and 14: n-gram (2-4) hashes over a compressed 99,092-token vocab, 8 heads, 24 rows of 256 per
  token from two 384M-row fp8 tables (~203 GB together), gated into each hyper-connection copy.
- DSpark: 3 draft stages (128 experts, top 3) reading the attention input of layers 37-39, 5-token blocks, Markov
  head and confidence head. Vision: 32-layer ViT + aligner, up to 1,024 image tokens.

## Memory budget (256 GB Mac)

| Part | As shipped | Plan |
| --- | --- | --- |
| Routed experts | 289 GB (mxfp4) | affine 2-bit g64, sensitive layers 3-bit: ~170-185 GB resident |
| Engram tables | 203 GB (fp8) | stay fp8 on SSD, mmap'd; only hashed rows are read (known from token ids before the forward) |
| Attention, shared experts, embed/head, DSpark, vision | ~15 GB | affine 4/8-bit or bf16: ~10 GB |
| KV | 4 compressed caches + 40 windows | ~1 KB a token: a 1M window is ~1 GB |

Go/no-go before engine work: the quality cost of 2-bit experts, measured with the torch reference (teacher-forced
top-1 agreement and logit error against the shipped fp4 weights).

## Phases

1. `ref/`: pure-torch CPU port of DeepSeek's inference code (tilelang kernels replaced by exact torch equivalents),
   weights streamed from the shards per layer. Golden dumps per layer and logits for short prompts.
2. Requant study: experts at 2/3-bit affine (and mixed) through the reference; pick the layout.
3. `convert`: HF checkpoint -> TensorFold checkpoint (MLX affine per-module bits + engram tables as fp8 files).
4. TensorFold `deepseek_v41` Metal family on the tf-checks/deepseek-v41 worktree (branch deepseek-v41 off origin/main):
   config, weights, hc/sinkhorn, attention (window + compressed + indexer + candidates), MoE, engram, head; checked
   layer by layer against the reference; then the native server hookup.
5. DSpark drafts, prompt cache, concurrency, vision, long context; harness speed runs; README.

## Engine notes (from the code map of origin/main 6ee828c2)

- Template: `native/lfm2_moe.zig` + `families/lfm2_moe` (LaneHost + `lanes.backend.Backend`: prefill, first, queue,
  read, verify, keep, draft, release; prompt cache via `api.prompt_cache.Store`). Register in `native/metal.zig`
  `families` + `open()`, and `tensorfold.zig`.
- Kernels to reuse: `core/row_projection` (decode, affine 2-8 bits incl. indexed experts), `core/affine_mm` (M5
  prefill incl. gather-GLU for experts, 2-6/8 bits, g32/64), `core/moe_route` (needs a 384/top-6 sqrt-softplus
  shape: it is sigmoid+bias today), `core/shared_attention` rope, argmax/sampling from the lane core.
- Hyper-connections differ from V4/GLM: V4.1 collapses each sublayer's input with the *previous* boundary's `pre`
  (the attention uses the last FFN boundary's, the FFN the attention boundary's; the head the last FFN's), and the
  first block uses copy 0. `core/hc_boundary.metal` collapses with its own `pre`, so the family carries a variant
  that takes the previous pre and writes the new one. Mix norm eps is 1e-20 (`norm_eps`).
- Chat template: no Jinja in the checkpoint; ship a native V4.1 encoder (`server/model_text.Text`) or a
  chat_template.jinja that reproduces encoding/encoding.py. DSML tool-call parsing already exists in the server.

## Status

- 2026-10-17: download done (48 shards, 475 GiB). Worktree at 6ee828c2 (branch deepseek-v41).
- ref/: CPU torch reference (DeepSeek's model.py/engram.py verbatim, kernel.py torch stand-ins, memmap loader);
  greedy "capital of France" reply matches expectations on the shipped fp4 weights. `--experts qBgG --dense qBgG`
  reproduce the converter's affine codes exactly (bf16 activations), so the reference is the engine's golden.
  2-bit g64 experts + 8-bit dense reproduce the fp4 greedy reply on that prompt (24/24 top-1).
- tools/convert.py: 2-bit g64 experts / 8-bit dense conversion running to ~/models/DeepSeek-V4.1-Flash-1017-TF-e2g64-d8.
- Engine (worktree): families/deepseek_v41 (config, weights, engram (mmap, CPU hash+gather), state, kernels,
  forward, backend), kernels/metal/deepseek/v41.metal, native/deepseek_v41.zig, tf-dsv41-forward check program.
  Builds; not yet run on the GPU.
- Chat template: the checkpoint ships none (encoding/encoding.py only); the server needs a chat_template.jinja.
- tools/yield.sh pauses the converter/study while another session runs a TensorFold engine or benchmark.
