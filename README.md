# DeepSeek-V4.1-Flash-1017 on one M5 Ultra

DeepSeek-V4.1-Flash-1017 served on a single 256 GB Mac Studio (M5 Ultra) by TensorFold's native Metal engine.

- **Context:** the full 1,048,576-token window.
- **Features:** DSpark drafts, the prompt cache, images, tools and structured output.
- **Checkpoint:** [`BobClawblaw/DeepSeek-V4.1-Flash-1017-TensorFold-e2e3-p4-h8`](https://huggingface.co/BobClawblaw/DeepSeek-V4.1-Flash-1017-TensorFold-e2e3-p4-h8), 418 GB.
- **Engine:** the `deepseek-v41-main` branch of [BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold/tree/deepseek-v41-main), pinned at `269f2447`.

The engine is not part of upstream TensorFold. The DeepSeek family upstream is being coordinated on
[ashhart/TensorFold#847](https://github.com/ashhart/TensorFold/issues/847).

## Results

Measured on one M5 Ultra (80-core GPU, 256 GB), macOS 27.0.1. Speed comes from
[recipe-harness](https://github.com/BobClawblaw/recipe-harness), target `deepseek-v41-m5`; its `stats/` has every run.

- **Engine for these runs:** the harness and 1M runs used build `6a4d7be7`. That commit was later reworded and rebased
  onto TensorFold 1.1.0 as `269f2447`, with the same DeepSeek code.
- **The rebase was checked:** its logits are byte-identical to the pre-rebase build on four prompts (chunks of 1,024
  and of 256, draft windows of 6, one row at a time).

### Decode, one stream

Harness speed plan, run `20261018-184411-speed`. Drafts on, thinking off, greedy, short prompts.

| Prose | Code | Structured | List | First token |
|---|---|---|---|---|
| 67.5 tok/s | 87.5 tok/s | 77.6 tok/s | 94.4 tok/s | 0.11–0.14 s |

At 1M deep, a follow-up turn on a 1,007,442-token conversation decoded 197 tokens at 46.3 tok/s, drafts on. That
is one sample from a script, not the harness.

### Prompts, cold

Harness prefill plan, run `20261018-183907-prefill`. One fresh prompt per size, timed to the first token (with thinking
on, the first reasoning token).

| Prompt tokens | Thinking off | Thinking on |
|---|---|---|
| ~2k | 1,093 tok/s | 1,339 tok/s |
| ~8k | 1,240 tok/s | 1,392 tok/s |
| ~33k | 1,256 tok/s | 1,334 tok/s |
| ~131k | 1,195 tok/s | 1,210 tok/s |

A 1,007,409-token prompt took 1,783.5 s (565 tok/s on average) through the server.

### Two streams

Streams take turns: the engine runs one stream's chunk or decode round at a time, so two streams together decode at
about the speed of one. Batching them into shared rounds is not done yet.

## What works, and how it was checked

Each row below was checked through the server's chat API on engine `269f2447`, apart from the harness checks, the 1M run and the check program noted, which ran on the same code before the rebase.

| Feature | Check |
|---|---|
| Greedy and sampled text | The ruler's prose, code, structured and list replies; the same seed at temperature 1 gives the same reply twice |
| Thinking | On by default; `chat_template_kwargs: {"enable_thinking": false}` turns it off; the reasoning comes back in `reasoning_content` |
| Long context | A passphrase planted 37% of the way into a 1,007,409-token prompt was recalled exactly |
| Tools | A `get_weather` call comes back as `tool_calls` with the right arguments, with thinking on and with it off |
| Structured output | A `json_schema` reply parses and fits its schema |
| Images | A photo of carrots is answered "Carrot" |
| Drafts | Drafted replies equal `"draft": false` replies (4 of 4); with the check program, a 268-token greedy reply drafted equals it plain |
| Prompt cache | The harness `turns` check: a resumed third turn equals the same turn sent cold, with 4,541 of 4,561 tokens cached; a 1M conversation resumes in 7.8 s |
| Concurrency | At `--parallel 2`, four requests sent two at a time each equal the same request alone |

Two notes on these checks:
- **Images:** data URLs only (`image_url` parts), PNG or JPEG. One image takes at most 1,024 tokens, a request at most
  4,096 image tokens (`vision_image_tokens`), and at most 4 images (`vision_max_images`).
- **Drafts with the check program:** that check ran on `544b3a40`, whose decode path the current engine keeps.

## Quality

- **Against DeepSeek's own fp4 weights:** KL 0.138 and top-1 agreement 0.909, teacher-forced on four prompts of 64
  tokens each (`tools/native_kl.py`, a row a step). Measured on engine `fcf9e0b4`; the current engine keeps that
  decode arithmetic.
- **Against the reference in FP64** (DeepSeek's `model.py` over this checkpoint's own codes; runners in
  [this gist](https://gist.github.com/BobClawblaw/5e120cb3f7a7b3c9d7ca9b40ebea29f0)):
  - **Eight tokens:** the engine is 8.08% NRMS from FP64, against 4.22% between the FP32 and FP64 references, with
    the top token right at 8 of 8.
  - **A 268-token greedy reply:** the engine's tokens are FP64's top token at 252 of 268 positions; FP32's agree at 250.
  - Details are on [#847](https://github.com/ashhart/TensorFold/issues/847#issuecomment-6256853391).
- **Not run:** the harness quality suite, with thinking on or off.

## Requirements

- **Hardware:** a Mac with 256 GB of unified memory. The weights take about 200 GiB; the Engram tables (189 GiB) are
  read from disk where lookups land.
- **Chip:** measured on an M5 Ultra, where prompt chunks use the GPU's tensor units. Older chips take the plain
  kernels with `zig_cpu = "apple_m1"`; not measured.
- **Software:**
  - [Zig 0.17.0](https://ziglang.org/download/) for the engine
  - Python 3.11 or newer
  - `cmake` (for libjpeg-turbo)
  - `git`
  - the `hf` CLI to download the checkpoint
- **Disk:** about 420 GB for the checkpoint, plus the engine build.

## Running it

```sh
git clone https://github.com/BobClawblaw/deepseek-v41-flash-mlx-m5-ultra && cd deepseek-v41-flash-mlx-m5-ultra
git clone -b deepseek-v41-main https://github.com/BobClawblaw/TensorFold ~/github/TensorFold
hf download BobClawblaw/DeepSeek-V4.1-Flash-1017-TensorFold-e2e3-p4-h8 \
    --local-dir ~/models/DeepSeek-V4.1-Flash-1017-TF-mix23-d8b4
cp local.example.toml local.toml      # this Mac's overrides; edit paths and the port there
python3 -m mac validate               # settings, checkpoint and build, without starting anything
python3 -m mac up                     # builds the pinned engine, libjpeg-turbo and libtfgrammar once, then serves
```

The server speaks the OpenAI API at `http://127.0.0.1:8890/v1`, model `DeepSeek-V4.1-Flash`. Other commands:

| Command | What it does |
|---|---|
| `python3 -m mac status` | the server's pid, URL and health |
| `python3 -m mac logs` | the server log's last lines |
| `python3 -m mac chat "..."` | one prompt to the running server |
| `python3 -m mac down` | stops the server |
| `python3 -m mac command` | prints the serve command `up` runs |

**Running at login:** `python3 -m mac launchd --port 8888` writes a launchd unit that serves the LAN at login. Loading
it is up to you:

```sh
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tensorfold.deepseek-v41.plist
```

That unit is kept alive: a killed server restarts 30 s later. Stop it with `launchctl bootout` before running anything
else that loads the model, since two copies do not fit in 256 GB.

**Converting from the release instead of downloading:** with `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e10` in
the Hugging Face cache, run `python3 -m mac convert`, then `python3 tools/split_engram.py MODEL_DIR` to split the
Engram tables as the Hub copy is.

## Settings

All of these are in `recipe.toml`'s `[serve]`. `local.toml` overrides them, and so does an environment variable of
the same name in upper case.

| Key | Default | Meaning |
|---|---|---|
| `context` | 65536 | Positions a stream holds; the model's window is 1,048,576 |
| `parallel` | 2 | Streams at once; they take turns |
| `drafts` | true | The checkpoint's DSpark head drafts; replies equal undrafted ones |
| `thinking` | true | Thinking on unless a request turns it off |
| `prompt_cache_gib` | 6 | Memory for kept prompt states, used past the engine's 70% share (`prompt_cache_over_cap`) |
| `vision`, `vision_image_tokens`, `vision_max_images` | true, 4096, 4 | Image input and its limits |
| `max_tokens`, `temperature` | 32768, 0 | A request's defaults |
| `memory_margin_gib` | 6 | `up` refuses to start when free memory is under the weights plus this |

**For the full window,** set `context = 1048576` and `parallel = 1` (see `local.example.toml`). Measured that way, the
server held 202.4 GiB once loaded. At least 19.5 GiB stayed free through a 1,007,409-token prompt.

## Known limits

- **Streams take turns.** Two streams together are about as fast as one.
- **The leaderboard benchmark doesn't complete yet.** TensorFold's community-v1 `code` prompt is a raw completion with
  no template. At temperature 1 this checkpoint sometimes ends it early, so the receipt is incomplete and is not
  published ([ashhart/TensorFold#1019](https://github.com/ashhart/TensorFold/issues/1019)).
- **Not yet upstream:** the DeepSeek kernels are not registered with TensorFold's kernel contract.

## Layout

| Path | What it holds |
|---|---|
| `mac/` | `python3 -m mac`: settings, the engine build, the server, the launchd unit |
| `template/` | The chat template (renders byte for byte like DeepSeek's `encoding/encoding.py`) and its checks |
| `tools/` | The converter, the Engram split, the KL check against the fp4 release |
| `ref/` | A CPU reference: DeepSeek's `model.py` and `engram.py` (MIT, `LICENSE.deepseek`), torch kernels, the loader, the whole-model gate |
| `PLAN.md` | The work log, with how each number was measured |

## License

MIT (`LICENSE`). DeepSeek's `model.py`, `engram.py`, `image_processor.py` and the encoding under `ref/` keep their own
MIT license and copyright (`ref/LICENSE.deepseek`).
