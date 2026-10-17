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
- Chat template: template/chat_template.jinja renders byte-identical to encode_messages on 15 cases (4 DeepSeek
  fixtures + 11 request shapes), in jinja2 and in TensorFold's own engine (template/check.py --native: 30/30);
  converter copies it into the checkpoint. Server parses V4.1's spaced DSML tool calls (worktree commit 8af7f39e).
- GPU (2026-10-17, 8888 unloaded for it): the native forward matches the CPU reference block by block (each
  block's own error ~0.1-1% of the non-outlier channels, one bf16 step; isolated spikes = near-tied expert routes),
  top-1 23/24 on the check case (the miss a three-way near tie in the prompt), every reply position equal.
  Chunked prefill and one-row decode give bit-identical logits to the one-call forward.
- Served (tensorfold-native serve, port 8890): chat with thinking on/off correct, reasoning split; first unoptimized
  decode ~14 tok/s (curl timing, not a harness number), first token 3.0 s on 25 prompt tokens (row kernels).
- Decode speed (tf-dsv41-forward, 24 one-row steps, DS_PROFILE for sections): 61 ms -> 30.9 ms a step after
  parallel routing (22 -> 1.3 ms) and parallel hyper-connection mixes (7.5 -> 4.5 ms). GPU ms a step then:
  boundaries 4.5, attn proj 4.1, compress+index 1.7, attend 4.9, attn out 5.3, route 1.3, experts 3.7,
  shared 2.2, engram 0.4, head 0.6. A shared-key-tile attention (8 heads a threadgroup) was slower (17.9 ms:
  8 threadgroups a decode step) and was reverted; next: split keys across threadgroups (flash decoding).
- Prompt chunks: tensor units (core/affine_mm dense; experts sorted + gathered), shared-tile attention for prompt
  rows, engram pages prefetched (madvise WILLNEED: the gather was 984 ms of a 256-row chunk, now ~80): a 256-token
  chunk 2.8 s -> 0.85 s (~300 tok/s); decode 27.7 ms a step (~36 tok/s) on 24 tokens, 44 ms averaged over 1,500.
- Long context (1,500 tokens of README + code, real indexer selection past 512): next-token accuracy native 75.25%
  (chunks and decode alike) vs the reference's 75.12%; logit rel error flat 0.13-0.14 over positions. Layer 36's
  picks vs the reference fed the native streams: 99.5% mean overlap (min 97.9%), every row under 512 exact.
- DSpark drafts (2026-10-17): drafted replies equal one-row ones (tf-dsv41-forward DS_DRAFT, 128/128 tokens on
  chat, code, math, zh; and through the server: same reply hashes). Tokens a round (greedy chain of 5): code 4.27,
  math 4.27, zh 2.67, chat 1.97. Load-timed windows 1/2/4/6 rows: 35.7/39.8/52.7/64.8 ms, a draft round 4.4 ms; the
  lane core picks depths (speculate on). Through the server: code ~60 tok/s, chat ~38 tok/s (curl timings).
- Tools: chat and thinking modes call get_weather with the right arguments and answer from the tool's result.
- Prompt cache: a repeated 1,733-token prompt 8.7 s -> 1.0 s (1,536 cached), a follow-up turn 0.9 s; replies equal
  with the cache on and off (same hashes). Budget is the 70% cap's leftover: 2.2 GiB at 8k context.
- Requant study (teacher-forced on fp4 greedy replies, 4 prompts x 64 tokens): mean top-1 / KL vs fp4:
  q2g64 0.894 / 0.289, q2g32 0.894 / 0.260, q3g64 0.940 / 0.120 (zh worst at every width). 3-bit everywhere does
  not fit (~238 GB); a 2/3 mix of ~20 layers at 3-bit would (~208 GB); needs a layer-sensitivity pass.
- Mixed checkpoint (2026-10-17): layers 10-29 3-bit experts (quarter sweep: mean KL vs fp4 by 3-bit quarter
  0-9 0.277, 10-19 0.230, 20-29 0.173, 30-39 0.280, none 0.289, all 0.120), ~/models/...-mix23-d8, 206 GiB in use.
- Vision: preparation bit-exact with image_processor.py (both example images); the tower+aligner vs a reference with
  the same 8-bit codes: most rows cos 0.999-1.000, a few outliers 0.89-0.98; carrots/corn described correctly
  through the server; template 36/36 (image fixture included).
- Streams take turns (--parallel up to 4): two at once reply as alone.
- Recipe: python3 -m mac (up/down/status/logs/build/convert/command/chat/validate/launchd), tests.
- Harness (recipe-harness target deepseek-v41-m5, plan speed, run 20261017-145733): ruler c=1 prose/code/structured/
  list 41.1/56.7/48.4/62.2 tok/s, ttft 0.15-0.22 s; c=2 aggregate 42.7/58.3/49.4/78.5; turns: 4541/4561 cached,
  ttft 0.22 s vs 2.04 s cold, replies equal. Qwen3.8-Flash-Next on 8888 (its 120921 run): 142/210/173/206 at c=1.
- Decode speed (2026-10-17, DS_TIMING GPU ms a step, 24-token prompt, mix23): 27.2 -> 25.6 (attention split by
  keys: 8 heads share each key tile, chunks merged with the sink; at 1,500 tokens 42.0 -> 25.9) -> 23.1 (the
  boundary's Sinkhorn on a simdgroup). Deterministic run to run; these reorder float sums, which moves logits a
  bf16 step at position 0 and up to a few units later (routing near-ties), as the validated split change did.
  DS_ABLATE (each kernel left out): row projections 7.2 ms, routed experts 3.9, gemv bf16 2.8 (wo_a), attention
  2.5; 1,401 dispatches a step at ~4 us each, serial or barrier alike (DS_DISPATCH).
- Dense bytes dominate decode: ~6 GB of 8-bit projections a token (wq_b and wo_b 42M each a layer, the shared
  expert 35M) against <0.1 GB of routed experts; the row kernels run them at ~860 GB/s.
- wo_a: affine, stacked by group [8, 1024, 4096], decode through the indexed expert kernels, prompts through
  affine_mm's denseBatch (convert.py writes it; the engine keeps the bf16 path for older checkpoints).
  Incident: a first in-place rewrite of mix23's shards (mx.load + mx.save_safetensors) wrote 26 MB files under
  full headers; the blocks were reconverted from the HF source (convert.py --only blocks,mtp). Never rewrite a
  checkpoint in place; convert to a new file and check its size against its header first.
- Concurrent encoder: every dispatch behind a barrier unless marked to run beside the one before (q and kv chains,
  routed gate/up, the shared expert beside routing and the routed experts). Fused: kv_norm+RoPE+fp8+ring put
  (ds_kv_finish), the merge's inverse RoPE + fp8 (ds_attend_merge finish).
  Bit-identical to the serial build on e2g64-d8 (logits over 24 and 1,500 tokens byte-equal): 22.6 -> 20.7 ms.
  The sublayer expand deferred into the next boundary's mix sums with the MoE combine inline: also bit-identical,
  no measurable gain (those barriers were not the cost).
- Native quality check (tools/native_kl.py: the engine teacher-forced on the study's fp4 replies, chunk 1):
  mix23 8-bit projections KL 0.136 / top-1 0.905; mix23 with the blocks' projections 4-bit (convert.py
  --block-dense-bits 4, head and tower 8-bit) KL 0.145 / top-1 0.905, decode 21.2 -> 17.7 ms a step at 1,500
  tokens, long next-token accuracy 0.7485, drafts equal plain 128/128 (77 tok/s drafted, 52 plain). Adopted:
  ~/models/DeepSeek-V4.1-Flash-1017-TF-mix23-d8b4 (200.5 GiB of shards).
- Harness speed (run 20261017-175442, engine 2f1c1009): ruler c=1 prose/code/structured/list
  57.7/90.5/69.5/93.1 tok/s (was 41.1/56.7/48.4/62.2), ttft 0.13-0.19 s; c=2 62.6/90.1/74.3/112.6; turns 4541 of
  4569 cached, ttft 0.20 s vs 1.77 s cold, replies equal. No quality suite run (not asked for).
- Structured output: libtfgrammar built by the recipe (xgrammar v0.2.8); the backend hands the lane core its held
  drafts; json_object and json_schema replies parse and fit the schema through the server.
- Served (2026-10-17): the LAN's default server on 8888 (launchd com.tensorfold.deepseek-v41, host 0.0.0.0);
  com.tensorfold.qwen38-vision disabled (launchctl enable gui/$UID/com.tensorfold.qwen38-vision undoes it).
  Through it: text, code, tools, JSON, images (carrots, corn), a repeated image cached, a mirrored image of the same
  size not taken from the cache.
- tools/yield.sh pauses the converter/study while another session runs a TensorFold engine or benchmark.
