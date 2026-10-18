# Verification

All runs on one machine: Windows 11 Pro (build 26200), Intel Core Ultra 9 285K, 64 GB RAM, NVIDIA GeForce RTX 5090 (32 GB, driver
610.88). Models: `nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL` at `fd1e77fbe2f6eb96307a2e9512a7201ab7e3408f` and
`nyu-visionx/webssl300m_decoder` at `0e6a28d0744b31ac35624ecabadde20b540d8fa0` (SHA-256 of every file in `tools/download_models.py`).
Official code: `ZitengWangNYU/Scale-RAE` at `dab38002b4b42c2d2262776785276d410599a443`.

Two Python environments:

| | official environment | ComfyUI environment |
| --- | --- | --- |
| Python | 3.10.20 | 3.12.13 |
| torch | 2.8.0+cu128 | 2.14.1+cu130 |
| transformers / tokenizers | 4.37.0 / 0.15.0 (the official pins) | 5.20.1 / 0.23.3 |
| timm / einops | 0.9.16 / 0.6.1 | 1.0.30 / 0.8.2 |
| ComfyUI | — | v0.38.0 (`6b747c0`) |

## 1. The official reference run

The official text-to-image path (`inference/cli.py`: `load_model`, `build_prompt`, `tokenize_prompt`, `model.generate` with the CLI's
generation arguments, `decode_image_embeds`) was run from a small harness that imports those functions. Differences from
`python cli.py t2i`, none of which touches text-to-image: local folders instead of Hub downloads; the WebSSL **encoder** is built
from its `config.json` with random weights instead of downloading its CC-BY-NC-4.0 weights (text-to-image never runs it; the
decoder only needs its config and image statistics); several jobs in one process with the CLI's `set_seed` before each.

Jobs (`cfg_mode = official-cli` and `image_start = model` on the pack side, i.e. the released behaviour), all 50 steps:

| id | prompt | seed | guidance |
| --- | --- | --- | --- |
| garden_s42_g1 | Can you create an image of a peaceful garden? | 42 | 1 |
| cat_s42_g1 | Can you generate a photo of a cat on a windowsill? | 42 | 1 |
| lake_s42_g1 | Show me a photo of a serene lake at dawn | 42 | 1 |
| robot_s123_g1 | Can you give me a photo of a futuristic robot? | 123 | 1 |
| garden_s42_g3 | (garden) | 42 | 3 |
| cat_s7_g3 | (cat) | 7 | 3 |
| garden_s42_g1_again | (garden, repeated in the same process) | 42 | 1 |

Hashes: SHA-256 (first 16 hex digits) of the float32 bytes of the 256 x 1024 latent, and of the 8-bit RGB pixels as the CLI saves
them (`(clip(x, 0, 1) * 255).astype(uint8)`).

| id | official (official env) latent / image | pack in-process (official env) | pack in-process (ComfyUI env) | pack via ComfyUI queue (ComfyUI env), image |
| --- | --- | --- | --- | --- |
| garden_s42_g1 | `5c39e71fe904e6bb` / `068b7a3635229e68` | `5c39e71fe904e6bb` / `068b7a3635229e68` | `27f5ba271c403d42` / `46e14a26255b0bb3` | `46e14a26255b0bb3` |
| cat_s42_g1 | `6c8a538e789eb955` / `f2c103a14bbacf73` | `6c8a538e789eb955` / `f2c103a14bbacf73` | `125b148040004680` / `ad9d04f6f6ff0b23` | `ad9d04f6f6ff0b23` |
| lake_s42_g1 | `ee85d6dcaa0b9d2d` / `aefc9e1d39d14597` | `ee85d6dcaa0b9d2d` / `aefc9e1d39d14597` | `90c780ff811c058b` / `3138495cb3ccf1df` | `3138495cb3ccf1df` |
| robot_s123_g1 | `bb022a839334cc3e` / `af0602602994c215` | `bb022a839334cc3e` / `af0602602994c215` | `16bdd7bba7a9261a` / `81758fd93959e01d` | `81758fd93959e01d` |
| garden_s42_g3 | `8822a6b291b8f743` / `1233a8becee8d285` | `8822a6b291b8f743` / `1233a8becee8d285` | `e2aeb911d28683c8` / `088ddb4f8c80f3e1` | `088ddb4f8c80f3e1` |
| cat_s7_g3 | `dd71733b5c74d562` / `c3d97ddcf2738d39` | `dd71733b5c74d562` / `c3d97ddcf2738d39` | `7646cd58b695732b` / `7995b014a3a9fe71` | `7995b014a3a9fe71` |
| garden_s42_g1_again | `5c39e71fe904e6bb` / `068b7a3635229e68` | `5c39e71fe904e6bb` / `068b7a3635229e68` | `27f5ba271c403d42` / `46e14a26255b0bb3` | `46e14a26255b0bb3` |

- **Same environment: 7 of 7 latents and 7 of 7 images identical** between the official code and the pack.
- **Across environments** (official env vs ComfyUI env): maximum pixel difference 1 of 255 on every job, PSNR 64.98 to 71.53 dB,
  relative L2 difference of the latents 3.8e-5 to 5.3e-5. The prompt ids are identical in both environments, so the difference is
  floating-point rounding of the different torch / CUDA versions.
- **Queue vs in-process** (ComfyUI env): 7 of 7 images identical; also in reverse job order, after a server restart, and after the
  later code changes (the last run is in the table).
- The model's text before the image (greedy): 5 tokens ("Here's the image:\n") or 7 tokens ("Here's the image you requested:\n").

### Two numerical details that the pack had to match

- **requires_grad of the weights.** With guidance > 1 the DiT runs a two-row batch. PyTorch folds the batched matmul differently when
  the weight does not require grad, and the guidance-3 jobs then differed from the official run in the last bits (the latents of the
  guidance-1 jobs were identical). The official model keeps `requires_grad=True` on its parameters; the pack does the same.
- **Loading inside `torch.inference_mode()`.** ComfyUI runs nodes under inference mode; weights created there are inference tensors,
  which behave like `requires_grad=False` above, and the queue's guidance-3 jobs differed from the in-process run. The pack loads with
  inference mode switched off (tested in `tests/test_runtime.py`).

## 2. Prompt handling

`tests/data/token_ids.json` holds the chat text and token ids of five prompts produced by the official `build_prompt` +
`tokenize_prompt` with transformers 4.37 (English request, English request with more words, Japanese, leading / trailing whitespace
with a newline and a tab, an emoji plus a literal `<|im_start|>`). The pack's `chat_text` and `build_input_ids` give the same text and
ids with transformers 4.37, 4.50.3 (CI), 5.20.1 and the latest release (CI). The special token ids (`<im_start>` 151665,
`<|im_end|>` 151645) are checked too.

## 3. Guidance

- Official code, guidance 3 vs 1 (garden, seed 42): PSNR 70.25 dB, maximum pixel difference 1, latent relative L2 2.8e-5 — the
  guidance option of the released code has no effect (upstream issue #7: the default `cfg_interval = [-1e4, 1e4]` makes
  `forward_with_cfg` return the conditional prediction). The pack's `official-cli` mode reproduces this bit for bit (table above).
- With the two-value fix proposed upstream (ZitengWangNYU/Scale-RAE#10: projector default `cfg_interval = [-1e4, -1e4]`), the
  official code gives bit-identical results for the five guidance-1 jobs and applies guidance at 3 (PSNR 14.06 / 17.16 dB against
  the unpatched guidance-3 images); the pack's `standard` mode gives the same latents and images as the patched official code, bit for
  bit (2 of 2 guidance-3 jobs, official environment).
- `standard` mode, guidance 3 vs 1 (cat, seed 42): PSNR 14.54 dB — guidance takes effect (`docs/images/guidance.png`, guidance 1, 2,
  3, 5 for two prompts). Guidance 1 is identical in both modes (`tests/test_runtime.py::test_guidance_modes`).

## 4. dtype

Same seven jobs in the ComfyUI env, compared with float32:

| dtype | peak VRAM allocated | per image, guidance 1 / 3 | PSNR vs float32 | latent rel. L2 vs float32 | model's text before the image |
| --- | --- | --- | --- | --- | --- |
| float32 | 17.36 / 17.42 GiB | 2.20 to 2.69 s / 3.77 to 3.83 s | — | — | — |
| float16 | 9.49 / 9.52 GiB | 1.39 to 2.44 s / 2.31 to 2.37 s | 37.28 to 56.29 dB | 0.0023 to 0.051 | same on 7 of 7 |
| bfloat16 | 9.49 / 9.52 GiB | 1.37 to 2.62 s / 2.34 to 2.38 s | 25.95 to 33.31 dB | 0.052 to 0.147 | different on 4 of 7 (9 tokens instead of 5 or 7) |

The first version ran the RAE decoder in the chosen dtype; in float16 every one of the seven images came out as the same flat
picture (identical hash) although the float16 latents were within 0.2 to 5 % of float32. The decoder now always runs in float32
(1.66 GB of weights, one pass per image); the numbers above are with that change. float32 is unchanged by it (identical hashes).

## 5. CPU

float32 on the CPU (`device = cpu`), garden, seed 42, 50 steps: 52 s for the image after a 3.3 s load (files in the OS cache).
The image is a valid garden but differs from the GPU image, because the sampler's starting noise (`torch.randn_like` on the device)
differs between the CPU and CUDA generators for the same seed. The text before the image was the same (5 tokens).

## 6. Text-only answers and `image_start`

80 prompts through the real queue, `image_start = model`: 20 subjects in four phrasings.

| phrasing | started an image |
| --- | --- |
| "Can you generate a photo of {subject}?" | 20 of 20 |
| "Can you create an image of {subject}?" | 20 of 20 |
| "Generate an image of {subject}." | 20 of 20 |
| "{Subject}" (plain description) | 0 of 20 |

Other prompts answered with text: "Describe a sunny beach.", "A red stop sign on a snowy street.", "Show me a red stop sign on a snowy
street." and "Can you generate a close-up photo of a bowl of ramen?" (the model replies that it cannot generate images). The error
message of `model` mode shows the start of that text.

Through the queue with the real model: a request prompt gives the identical image in `auto` and `model` mode; a plain description
fails in `model` mode and gives the identical image in `auto` and `force` mode (single image and a batch of 3; in `auto` the text is
decoded once per batch). The text-only answer cost about 7 s before the forced image in that example. For six request prompts,
`force` and the official path gave very similar pictures (same composition), and six plain descriptions with `force` gave images of
the described subjects.

## 7. Other real-queue checks

- Interrupt during a batch of 16: the job ends as interrupted after about 4 s; the next job runs normally with the resident model.
- `ScaleRAEUnload` after a job: `unloaded: true`, CUDA memory allocated by the process back to 32 MiB; running it again: `unloaded:
  false`. `keep_loaded = false`: the model is freed after the job (next Unload reports `unloaded: false`).
- A prompt containing `<image>` is refused with a clear message (the official code would turn it into an image slot).
- `workflows/api/t2i_fp16_batch.json`: 4 images, seeds 7 to 10, 9.46 GiB peak.
- Windows commit charge: 25.4 GiB before loading, 47.1 GiB peak during loading, 42.8 GiB with the model resident (float32); the
  ComfyUI process's working set was about 3 GiB.

## 8. Clean install

Fresh ComfyUI v0.38.0 checkout, fresh Python 3.12 virtual environment (torch 2.14.1+cu130, then ComfyUI's requirements, then this
pack's `requirements.txt`: transformers 5.20.1, timm 1.0.30), `git clone` of this repository into `custom_nodes`. The decoder was
downloaded with `tools/download_models.py --only webssl300m_decoder` (53 s, all three files SHA-256 verified); the 16 GB generator
folder was linked in through `extra_model_paths.yaml` and checked with `--verify-only`. ComfyUI loaded the pack without warnings and
listed both folders. Through the queue the seven jobs of section 1 gave 7 of 7 images identical to the development environment
(PSNR 64.98 dB or more against the official run), and `workflows/api/t2i_basic.json` and `t2i_fp16_batch.json` ran unchanged.
`comfy node validate` passed; the Registry archive from `comfy node pack` has 42 files (393 KB: the runtime package, the vendored
code, tools, workflows, docs and license files; no tests, scripts or images).

## 9. Automated tests (CI)

`tests/` (43 tests) runs on the CPU with a tiny random model that has the official file layout (safetensors with the same tensor
names, real tokenizer files, a `model.pt` decoder), so the real loader, sampler and decoder code run: prompt ids against the official
ids, loading checks (missing tensors, wrong encoder space, inference mode, dtype on CPU), sampling (shape, range, reproducibility,
global RNG untouched, step counts, guidance modes, the three `image_start` modes, restoring the DiT after an error), the nodes as
ComfyUI loads them (registration, model folder discovery, ComfyUI's own validation of the shipped workflows, batches, keep-loaded,
unload, interrupt), the vendored code against the upstream commit, and the absence of environment-variable, network and process code
in the installed package. `scripts/ci_queue_check.py` starts a real ComfyUI on the CPU and runs the shipped workflow through the queue
with the tiny model (reproducibility, batch, keep-loaded off, unload). CI matrix: Ubuntu 24.04 and Windows with Python 3.12, torch
2.14.1 and transformers 4.50.3 / latest, plus Ubuntu with Python 3.10, torch 2.8.0 and transformers 4.50.3; all five jobs pass.
