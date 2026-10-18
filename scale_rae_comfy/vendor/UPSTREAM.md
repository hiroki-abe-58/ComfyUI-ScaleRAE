# Vendored upstream code

Source: https://github.com/ZitengWangNYU/Scale-RAE, commit `dab38002b4b42c2d2262776785276d410599a443` (MIT, see `LICENSE` in
this folder: Copyright (c) 2026 Shengbang Tong, Boyang Zheng, Ziteng Wang).

Only the modules that text-to-image needs are vendored: the rectified-flow DiT head (`model/diffusion_loss`) and the RAE pixel
decoder (`model/multimodal_decoder/decoder.py`). The language model is this pack's own `scale_rae_comfy/qwen2.py`, and the chat
template and tokenization are `chat_text` / `build_input_ids` in `scale_rae_comfy/runtime.py` (checked against token ids recorded
from the official CLI, `tests/data/token_ids.json`).

`scripts/vendor_sync.py` rebuilds this folder from an upstream checkout; `tests/test_vendor.py` repeats that in CI and compares every
byte, and `MANIFEST.sha256` lists the SHA-256 of every file here.

## Copied unchanged (byte-identical)

- `scale_rae/model/diffusion_loss/__init__.py`, `diffloss.py`
- `scale_rae/model/diffusion_loss/diffusion/`: `__init__.py`, `diffusion_utils.py`, `gaussian_diffusion.py`, `respace.py`, `rf.py`,
  `simple_diffusion.py`, `timestep_sampler.py`, `transport.py`
- `scale_rae/model/diffusion_loss/models/`: `__init__.py`, `attentions.py`, `model_utils.py`

## Patched (no change to the code that runs)

- `scale_rae/model/diffusion_loss/models/lightningDiT.py`: the 9 comment lines that contain `os.getenv(` (commented-out TPU sharding
  code) are removed. Nothing else changes; the test compares the parsed syntax trees of both versions.
- `scale_rae/model/multimodal_decoder/decoder.py` (upstream keeps two copies of the same code in this file; both get the same edits):
  - `from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer` becomes an import of
    `prune_linear_layer` plus a guarded import of `find_pruneable_heads_and_indices`, which transformers 5 removed. Only
    `prune_heads()` uses it, and inference never calls that.
  - the three lines of the `ViTMAEModel.forward` docstring example that download a COCO image with `requests` are removed
    (documentation only).

## Pack shims (own files, not from upstream)

- `scale_rae/__init__.py`, `scale_rae/model/__init__.py`: package markers. Upstream's versions import the whole training and
  image-understanding stack.
- `scale_rae/utils.py`: `IS_XLA_AVAILABLE = False` (the only name the vendored modules import from upstream's `utils.py`).
- `scale_rae/model/diffusion_loss/models/AdaLNmlp.py`: a stub `SimpleMLPAdaLN` that raises if used. `diffloss.py` imports the name;
  only the MLP diffusion head (`use_mlp=True`) uses the class, and the released text-to-image checkpoints use the LightningDiT
  head.

## Not vendored

Everything else, in particular the language-model wrappers, the vision encoders, training, evaluation and the Gradio demo.
`model/diffusion_loss/utils.py`, `nextDiT.py` and `next_utils.py` are not imported by the vendored modules.
