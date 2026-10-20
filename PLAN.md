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
- Streams (--parallel up to 4): shared forwards since c4c69219 (below); before that they took turns.
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
- Decode, second round (2026-10-17, long prompt, GPU ms a step): 17.7 -> 13.95. Wins, each bit-identical to the run
  before it except the chunk size: attention in 16-key chunks read once into threadgroup memory (vector loads), 16
  heads a threadgroup, the queries' RoPE as they load; the merge a thread a dim; fp8 rounding a lane a value; topk's
  sums and scans parallel; the indexer's scores a lane a head. Serial loops inside small kernels were the cost.
  Tried, no gain or slower (reverted): twin and wider-row dense kernels for windows, expert-sorted windows, fused
  gate/up/SwiGLU, weight prefetch into cache, a split-k router gemv, 32 heads a threadgroup, per-simdgroup mix parts.
  Checks: prompt chunks bit-identical to the previous commit; drafts equal plain 128/128; KL vs fp4 0.138, top-1 0.909.
- Harness speed (run 20261017-193300, engine fcf9e0b4): ruler c=1 prose/code/structured/list 70.6/96.3/76.1/91.6
  tok/s (was 57.7/90.5/69.5/93.1), c=2 76.6/94.7/77.0/117.6; turns cached 4541 of 4561, replies equal.
- Indexer at depth (2026-10-18, engine profile, 256-row chunk at 1.04M deep): the indexers after layer 20 score only
  their candidate blocks (6f9b7a04: 2,765 -> 1,503 ms a chunk, 35.6 -> 27.8 ms a decode step); the full-context
  scorer on 2-row by 16-key tiles (a7bbd088: 1,503 -> 1,132 ms, 27.8 -> 26.0 ms). Both bit-identical on four prompts
  (24k in chunks of 256, tied scores, 18k in windows of 6, 17k a row at a time). Review pass 544b3a40: dead kernels
  and fields out, stale comments fixed; same four prompts bit-identical.
- Whole-model gate (ref/gate.py: model.py over this checkpoint's own codes in bf16, fp32 and fp64; gist
  5e120cb3f7a7b3c9d7ca9b40ebea29f0): akol1's 8 tokens, floor NRMS(fp32, fp64) 4.22%; engine 8.08% vs fp64 (1.91 F),
  the bf16 reference 9.67% (2.29 F), argmax 8/8. 268-token greedy reply replayed: engine = fp64 argmax at 252/268,
  fp32 chain at 250/268 (disagreements at near-ties, fp64 margins 0.001-1.10); drafts equal plain 268/268.
- Prompt attention on the tensor units (6f4617c9, GLM's sparse kernel adapted: window ring + picks, the sink):
  attention 391 -> 22.5 ms a 256-row chunk; not bit-identical (sums order): 276-token sequence NRMS vs fp64
  16.32% -> 16.16%, argmax 260 -> 263; chunks of 64 and 256 give identical logits.
- Prompt chunks of 1024 rows (6a4d7be7; sweep 256/512/1024/2048 at depth 0: 819/1080/1270/1446 tok/s in the
  profile, scratch 0.21/0.42/0.85/1.69 GiB). Served memory 202.4 GiB once loaded (+0.7).
- Harness prefill (new plan `prefill`: one cold prompt each at 2k/8k/32k/128k, thinking off then on), tok/s:
  544b3a40 384-412; 6f4617c9 887-1073; 6a4d7be7 1093-1392 (128k: 1195 off, 1210 on). Speed run on 6a4d7be7:
  ruler c=1 67.5/87.5/77.6/94.4 (decode unchanged), turns resumed == cold, 4541 of 4561 cached, cold ttft 0.74 s.
- 1M on 6a4d7be7 (2026-10-18, through the server, thinking off): the 1,007,409-token recall prompt in 1,783.5 s
  (565 tok/s; was 6,726 s, 150 tok/s), passphrase right, at least 19.5 GiB free throughout; a follow-up turn
  restored 1,007,407 cached tokens, first token 7.8 s, decoded 197 tokens at 46.3 tok/s (drafts on; one sample).
- Thinking on by default (recipe.toml; a request turns it off with chat_template_kwargs.enable_thinking false).
- Published: BobClawblaw/DeepSeek-V4.1-Flash-1017-TensorFold-e2e3-p4-h8 on the Hub (418 GB; the Engram tables in
  three parts each, tools/split_engram.py, engine 39e5c50a reads them; logits identical to the whole tables).
  The engine branch is not public yet; no harness quality suite run.
- 6a4d7be7 reworded to de6894db (same tree; the binaries differ only in the embedded commit, build paths, object
  timestamps, UUID and signature). Harness runs 20261018-183907-prefill and -184411-speed and the 1M run were made on
  6a4d7be7's build; local.toml and the harness target now pin de6894db.
- Rebased onto origin/main 7d34f3dc (TensorFold 1.1.0, beta 8 app) as 55a9bfed: no conflicts; the logits are
  identical to the pre-rebase build on four prompts (24k in chunks of 1024, tied scores in chunks of 256, 18k in
  windows of 6, 17k a row at a time); test, kernel-contract, golden (306/319, 0 unexpected), fuzz-cortheon, flashnext
  unit tests, fmt and lean (0 new on 503) pass; the Windows job not run. The prompt attention kernel lost its unused
  pre-M5 branches (same logits). local.toml and the harness target pin 55a9bfed.
- Pushed: the engine branch to BobClawblaw/TensorFold deepseek-v41-main at 269f2447 (55a9bfed rebased onto a
  docs-only main commit, the same code); local.toml and the harness target pin it.
- Harness quality on 269f2447 (run 20261018-202801-dsv41-quality, 2026-10-18, the served build at 1M context and
  parallel 1, GLM recipe's quality suite). Thinking off: drafts 12/12, behaviour all pass, tools 58/60 (an empty
  optional `folder`, an extra `label`), json 30/30, ifeval 129/150 loose (122 strict), gsm8k 247/250, mgsm 217/240,
  mmlu 307/342, math500 457/500 (12 without a boxed answer). Thinking on: the harness's problems 10/10, gsm8k
  237/250; re-asked, 9 of its 10 misses hit the suite's 1024-token cap mid-reasoning and one answered wrong. The
  suite's ifeval always sends thinking off (129/150 again).
- Thinking reserve (7837387a; recipe `thinking_reserve = 32768`): with thinking on, the server adds 32,768 tokens to
  a request's max_tokens and the lanes stop the answer after `</think>` at the request's own max_tokens
  (Request.answer_max). Served checks: max_tokens 8 / 40 thinking on gave 2,095 reasoning + exactly 8 / 40 answer
  tokens; thinking off exactly 8 / 40; drafted == plain. The 10 thinking-on GSM8K misses at max_tokens 1024 all
  finish now (779-12,881 tokens), 5 right. zig build test passes (see the 2026-10-19 note on
  its "failed command:" lines).
- Shared rounds and chunked prompts (c4c69219; recipe parallel 4, context 524288 shared by the streams). forward.many
  runs up to four streams' windows (24 rows) on the row kernels; a lone prompt chunk keeps the tensor units. Lone
  logits byte-identical to 7837387a at chunks 1024/256/6/1; DS_SHARED=30: logits 420/420 rows, drafts 120/120.
  Served: four concurrent replies equal alone, a reply during a 60k-token prompt equals alone, prompt chunks and
  decode split 14.6 s / 14.6 s (Info.decode_share 0.5). Harness 20261019-090829-dsv41-speed, wall tok/s: 1 stream
  64.0/86.8/71.4/84.7, 4 streams 93.5/101.4/89.8/102.1. Next for throughput: one weight read per expert for all the
  rows that pick it, each row's arithmetic unchanged.
- Tried and dropped (2026-10-19): routed experts grouped by expert for multi-row forwards (local branch
  grouped-experts, not pushed). A core kernel sorted the slots by expert and read each expert's weights once for
  blocks of 4, 2 and 1 of its slots, each slot's sums as the indexed kernel's: byte-equal at every width
  (tf-row-check), lone logits byte-identical at chunks of 6 and 12, DS_SHARED 420/420 rows and 120/120 drafts. Rows
  share experts (the last MoE layer: 1,272 distinct for 2,520 slots in DS_SHARED's 30 rounds), yet a shared round
  took 95.3 ms against 91.1 ms, and a 6-row chunk's experts 17.28 against 16.51 ms. The experts cost grows with slots
  (4.04 ms at one row, 16.51 at six) whatever is shared: on the row kernels each slot's multiply-adds are the cost,
  not the weight reads. Only the tensor units would cut it, which changes a lone stream's bits against today's builds.
- Also measured (2026-10-19): a tiled version of akol1's family's index scorer (#825), byte-equal to its kernel, is
  1-6% slower on an idle GPU (256 rows at 520k keys: 1,344-1,351 against 1,318-1,319 ms). Earlier timings that showed
  it faster ran beside a quality run. That indexer's time is its merge, one thread a row over every score.
- Tried and dropped (2026-10-19): every forward's routed experts on the tensor units' gathers (local branch
  units-experts, DS_UNITS_EXPERTS). tf-affine-ref shows a one-expert gather gives each row the same bits in windows of
  1 to 65 rows at every width, but at decode sizes it is slower: the experts took 8.48 ms a 1-row call (row kernels
  4.05) and 25.00 ms a 6-row call (16.53), each slot in its own mostly empty 32-row tile (384 experts, few slots
  share one). DS_SHARED with it also failed (SharedForwardDiffers): the many-expert gather as called is not batch
  invariant. Not pursued: the row kernels stay for decode, and four streams stay at 1.17-1.46x one.
- Thinking reserve 32768 -> 49152 (2026-10-19, the recipe default): four thinking requests at max_tokens 32768 reserve
  about 82k tokens each, within the default context 524288. Two GPQA-Diamond questions (8 and 10 in the suite's order)
  had used all 36,864 tokens reasoning; asked again as the suite asks (greedy, max_tokens 4096) with the 48k reserve,
  both used all 53,248 and still gave no answer (202,637 and 226,826 characters of reasoning).
- Tried and dropped (2026-10-19): the expert routing kernel (ds_route) as per-simdgroup picks and one merge, not six
  passes with barriers (local branch fast-route). Skipping ds_route had taken a 1-row forward from 14.10 to 12.46 ms,
  but the rewrite saved only 0.1 ms (14.07 -> 13.97 ms; 3 rows 23.72 -> 23.63 ms): with routing skipped the experts
  read stale picks, so the skip measured more than routing. Other DS_SKIP numbers may overstate a part the same way;
  time a rewrite before trusting them. A 1-row forward is GPU-bound: encode 0.4 ms of 14.7.
- Harness quality, thinking on (run 20261019-034805-dsv41-thinking, 7837387a, 32k reserve): gsm8k 243/250 (237 before
  the reserve), math500 451/500 as graded, at least 469 under the suite's fixed grader (at least 465 thinking off).
  GSM8K's four extra thinking-on misses are ambiguous wording read another way. GPQA-Diamond not run to the end
  (stopped 2026-10-19): two questions loop under greedy reasoning, past 36,864 and 53,248 tokens.
- Long-cycle loop guard (f8a3ce2f; recipe loop_guard = true): the reasoning closes when a passage repeats verbatim, a
  whole period and at least 1,024 tokens, found from where each 32-token run ended (long_loop.zig). GPQA 8 and 10:
  closed at periods 5,352 and 2,996 after 29,359 and 12,122 reasoning tokens, then answered C and A (gold D, D).
  Question 10 drafted == "draft": false (sha 1fd5b84a8c20). Seven GSM8K thinking replies unchanged with the guard.
- zig build test (2026-10-19): it passes here, exit 0 with every step succeeded (f8a3ce2f: 292/292 steps, 3,037 of
  3,044 tests, 7 skipped). Zig 0.17's build runner prints "failed command:" over any step that wrote to stderr, even
  one that passed (lib/compiler/Maker.zig: "No matter the result, we want to display error/warning messages"). Our
  tests log warnings, so those lines read as failures, and earlier notes and commit messages said test-cuda-host
  steps, and later ten binaries, failed. None did. Read the Build Summary line and the exit code, not those lines.
- The loop guard commit was reworded to f8a3ce2f (2026-10-19, the same tree): its message had said ten test binaries
  fail under zig build test, the runner's "failed command:" lines misread. It was ab93aeef.

- Prompt speed (2026-10-20, engine bfad20ae, three changes, each bit-identical to f8a3ce2f on a 1,500-token prompt and
  an 8,192-token random one):
  - Profile first (tf-dsv41-forward DS_DEPTH, a 1,024-row chunk, GPU ms by section): depth 0 809 ms (1,266 tok/s) with
    the routed experts 432 (every expert's weights stream once a chunk, ~16 slots an expert in 32-row tiles);
    compress+index grows 1.7 ms a 1k of depth (the four full-scan indexers): 277 at 131k, 898 at 524k, 1,751 of
    2,546 at 1M (402 tok/s). The profiler cycles a warm prompt: on novel text the host engram gather reads cold table
    pages, 307 ms of a 1,171 ms chunk (DS_PROFILE on an 8,192-token random prompt), not the 21 ms it shows.
  - A lone stream's prompt goes in chunks of 3,968 rows (the ring less the window; 1,024 beside other streams, which
    decode between chunks): 2,585 ms a chunk, 1,535 tok/s at depth 0 (2,048 rows: 1,427). Save marks sit at the
    history boundary, not every chunk, so the prompt cache is unchanged; a short prompt's mark still cuts at 1,024.
  - The next chunk's engram rows are gathered on the host while this one's command buffer runs (forward.runChunk, two
    sets of rows in the scratch): the random prompt's chunks 1,171 -> 782 ms.
  - A lone chunk's index scores on the tensor units (v41_index_nax.metal, GLM's tile: 64 rows by 64 keys), the lane
    kernel's bf16 rounding of each head's term and head-order sum kept; score_budget 256 MiB (64-row slices at 1M).
    Compress+index at 131k/524k/1M: 277/898/1,751 -> 131/270/556 ms; the 1,024-row chunk 931/1,066/1,354 ms. The
    scores are bit-identical because the index queries and keys are fp4: a 128-term dot of 2-bit significands is
    exact in fp32 in any order. DS_SKIP=ds_index_scores_nax changes every row (the path runs).
  - Together, 3,968-row chunks at depths 0/131k/524k/1M: 1,535/1,325/1,132/854 tok/s (was 1,266/953/604/402). What
    is left at 1M is the indexer still (2,145 of 4,644 ms: the kept scorer, block max and the top-k over 1M scores a
    row), then the experts (1,193). zig build test passes (exit 0); fmt clean.
  - Quality with thinking off on bfad20ae (run 20261020-043551-dsv41-quality-off, the fixed MATH-500 grader): GSM8K
    246/250, MATH-500 462/500 (17 without a boxed answer), MGSM 217/240, MMLU 306/342, IFEval 130/150 loose and
    124/150 strict, tools 57/60 (each miss an extra optional argument), json 30/30, drafts and behaviour all pass.
    Against the 2026-10-18 run on 269f2447: one down on GSM8K, MMLU and tools, one up on IFEval, MATH-500 up on the
    grader. The suite sends 16 requests at a time to a four-stream server; which replies share a forward differs
    between runs. Not re-run with thinking on.
  - Harness on 98e41cae (bfad20ae plus a release-path fix: a stream's release drops the chunk gathered ahead for it):
    prefill run 20261020-061318-prefill, thinking off 1,172/1,453/1,430/1,383 tok/s at ~2k/8k/33k/131k (6a4d7be7:
    1,093/1,240/1,256/1,195), thinking on 1,354/1,456/1,431/1,382; speed run 20261020-061904-dsv41-speed, one stream
    70.6/86.4/76.2/92.9 tok/s (prose/code/structured/list), two 86.9/103.1/97.5/117.0, four 107.2/118.5/114.3/123.5.
    The LaunchAgent serves 98e41cae.

- Upstream merged (2026-10-20, engine c14c1848 = 98e41cae + TensorFold 1.1.1, 93 upstream commits): upstream grew its own CUDA
  DeepSeek-V4.1 family in families/deepseek_v41 (about 250 files, four names colliding with ours), so a rebase would have
  replayed 34 commits through add/add conflicts; instead the Metal family moved to families/deepseek_v41_metal (one
  commit) and upstream main was merged (three content conflicts: build.zig, where metalTargets moved to
  zig/build/metal_targets.zig and the tf-dsv41 programs went with it; native/metal.zig's --vision message; cli.zig's
  option line, --thinking-reserve kept beside upstream's --prompt-seed). The registry entry says raw_text = false. A
  lean-rule commit followed (the engram-ahead gather in ahead.zig, comments under 120 columns). Checks: zig build test,
  test-kernel-contract, test-golden, fuzz-cortheon, fmt and the lean check (0 new) pass.
  - Logits bit-identical to 98e41cae: long.npy 1,500 rows, rand8k 8,192 rows in chunks of 3,968 and of 1,024.
  - Harness: prefill run 20261020-082113-prefill, thinking off 1,213/1,444/1,418/1,372 tok/s, on 1,359/1,442/1,422/1,370;
    speed run 20261020-082555-dsv41-speed, one stream 68.8/87.7/82.1/94.5, two 87.2/105.0/97.7/116.5, four
    107.1/119.3/115.5/122.1. Within noise of 98e41cae.
  - community-v2 benchmark (managed, the public e2e3-p4-h8 snapshot, not publishable: 1.1.1 refuses v2 uploads): all
    four cells complete, the code prompt through /v1/chat/completions (receipt route recorded): code T=1 88.69, chat T=1
    70.66, code T=0 95.19, chat T=0 71.45 tok/s. A v1 run on 98e41cae had failed again on code T=1 seeds 1235-1237 (50,
    34, 196 tokens), the same as on 2026-10-18: the seeds are fixed, so a retry cannot pass.
  - Upstream's new server rules: Host/Origin checks only for a loopback bind, private routes gated only with an API key
    configured; the recipe's 0.0.0.0 listener without a key answers the LAN as before (spot check: LAN Host header 200).
    Sampled requests without a seed draw a fresh one (two no-seed replies differed; seed 7 twice gave the same reply).
  - Spot checks through the LaunchAgent (c14c1848): thinking on/off, tool call, json_schema, data-URL image, drafted
    equals plain, LAN Host: pass. Quality suite not re-run on c14c1848 (the forward is unchanged and bit-identical).

- Decode windows (2026-10-20, engine 8557236a): why drafting paid so little, and the fix.
  - The server's logs: thinking-off replies proposed 0.87 drafts a round at 82-88% acceptance; the depth policy chooses
    by tokens a ms from the measured window costs (1 row 14.8 ms, 2 19.7, 4 29.5, 6 39.4: 4.9 ms a row, a third of a
    plain step), so it rarely drafted. Mia's two-Spark recipe (BertholomusAI's CUDA family) runs rounds of 16 rows.
  - Window profile (tf-dsv41-forward DS_DEPTH=256,8192 DS_PROFILE, GPU ms, the same at both depths): 1 row 22.6, 2 30.9,
    4 36.1, 6 45.4. Growth 1 -> 6 rows: experts +12.9 (56%), attn proj + attn out +6.2 (27%), attend +2.8, route +1.5;
    compress+index, boundaries, shared, engram and head flat. (Profiling serialises the sections: the server's own
    plain step is 14.8 ms.) Experts at 6 rows read ~15.9 GB in 16.9 ms (~940 GB/s): near the bandwidth floor.
  - A wide row kernel (batches of 6, 4, 2, 1 rows a weight read; bit-identical to the plain kernel in tf-row-check and
    on the model) was slower: attn proj 4.7 -> 11.4 ms at 6 rows, the server's 6-row window 39.4 -> 59.0 ms. A bench
    streaming 40 weight sets a command buffer (tf-row-check ROW_BENCH, kept out of the tree) showed the plain kernel's
    cost is per-row ALU and latency, not the re-read: preloading 3 or 4 rows' inputs spills registers, and column- or
    row-outer orders that keep one row live re-read inputs or codes from cache and gain nothing. Dropped.
  - core/simd_qmm_bits (8-row simdgroup-matrix tiles, a scalar twin at 1-2 rows, "row-exact": a row's reduction never
    depends on the window's rows) is flat with rows in the same bench: wq_b 32768x1280 8-bit, 40 sets: plain 2.0 / 1.9
    / 2.3 / 3.1 / 3.9 ms at 1 / 2 / 4 / 6 / 8 rows, tiles 2.0 / 2.2 / 2.0 / 2.1 / 2.1; wo_b 5120x8192: plain 2.0 /
    2.4 / 3.9 / 5.2 / 6.5, tiles 2.0 / 2.3 / 2.1 / 2.1 / 2.1; wq_a 1280x5120: plain 1.0 / 0.9 / 1.2 / 1.6 / 1.9,
    tiles 1.3 / 0.7 / 0.8 / 0.7 / 0.8. Adopted for a window's dense projections (rows under 16; prompt chunks keep the
    tensor units), g64 widths with fp32 sums; the twin checked against the tiles at load (first layer and head).
  - On the model: prompt logits bit-identical to c14c1848 (rand8k, chunk 1,024); a 6-row window equals one row at a
    time on 8,192 rows (tiles vs twin); DS_DRAFT 64/64 equal (plain 65.2, drafted 61.8 tok/s on the check's prompt);
    DS_SHARED 108/108 rows, 32/32 drafts (87.5 ms a round shared, 98.7 alone). A window's logits differ from the row
    kernel's (NRMS 0.20 on the random prompt, argmax equal 7,109 of 8,192; the random prompt's logits are flat, as the
    prompt path's differ from the decode path's by the same amount). tools/native_kl.py on the study's fp4 replies (190 tokens, three prompts): tiles KL 0.167 / top-1 0.889, row kernels
    0.164 / 0.893 (code 0.065/0.969 vs 0.063/0.953, math 0.128/0.859 vs 0.134/0.859, zh 0.333/0.823 vs 0.320/0.855).
  - Profile at 6 rows: attn proj 4.7 -> 2.6, attn out 5.6 -> 4.0, route 3.0 -> 2.4 ms; 45.4 -> 40.9. Server costs:
    1 / 2 / 4 / 6 rows 15.1 / 19.8 / 27.1 / 33.9 ms (was 14.8 / 19.7 / 29.5 / 39.4), shared 8 / 16 / 24 rows 45.2 /
    73.3 / 106.2 (was 54.5 / 91.9 / 131.3), a draft round 2.7 (3.2).
  - Harness: prefill run 20261020-091454-prefill 1,222/1,451/1,422/1,380 tok/s thinking off (unchanged); speed run
    20261020-091934-dsv41-speed one stream prose/code/structured/list 70.7/104.9/87.6/103.8 (was 68.8/87.7/82.1/94.5),
    two 96.6/111.6/106.9/150.4, four 126.8/138.5/132.5/146.4 (was 107.1/119.3/115.5/122.1). Spot checks 12/12.
    Checks: zig build test, kernel contract, golden, fuzz, fmt, lean (0 new) pass. Quality suite not re-run.
