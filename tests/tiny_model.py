"""A tiny random Scale-RAE generator and RAE decoder in the official folder layout, so the real loading and sampling code runs on the
CPU in seconds. Same architecture and file formats as nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL + webssl300m_decoder, with small
widths and depths; the tokenizer files are the real ones (TOKENIZER_FILES, from the pinned model revision)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

TOKENIZER_FILES = ("added_tokens.json", "merges.txt", "special_tokens_map.json", "tokenizer_config.json", "vocab.json")
GENERATOR = "Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL"  # the names the shipped workflows use
DECODER = "webssl300m_decoder"

GENERATOR_CONFIG = {
    "model_type": "cambrian_qwen", "architectures": ["CambrianQwenForCausalLM"], "hidden_act": "silu", "hidden_size": 64,
    "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 151667,
    "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "rope_scaling": None, "tie_word_embeddings": True,
    "vision_loss": "diffusion-loss", "vision_loss_mode": "query", "mm_vision_tower_aux_list": ["facebook/webssl-dino300m-full2b-224"],
    "vision_tower_aux_token_len_list": [256], "diffusion_model_channels": 1024, "diffusion_model_depth": 2, "diffusion_model_heads": 2,
    "diffusion_model_hidden_size": 64, "diffusion_model_z_channels": 32, "diffusion_class_dropout_prob": 0.1, "diffusion_base_dim": None,
    "diffusion_norm_stats_path": None, "dit_cls": "DiT",
}
DECODER_CONFIG = {
    "model_type": "vit_mae", "hidden_size": 1024, "decoder_hidden_size": 32, "decoder_intermediate_size": 64, "decoder_num_attention_heads": 2,
    "decoder_num_hidden_layers": 1, "hidden_act": "gelu", "image_size": 224, "patch_size": 14, "num_channels": 3, "layer_norm_eps": 1e-12,
    "qkv_bias": True, "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0, "initializer_range": 0.02, "mask_ratio": 0.75,
    "norm_pix_loss": False,
}


def _randomize(module: torch.nn.Module, generator: torch.Generator) -> None:
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                p.fill_(1.0)
            else:
                p.copy_(torch.randn(p.shape, generator=generator) * 0.05)


def make_tiny_models(models_dir: Path, tokenizer_dir: Path, seed: int = 0, generator_name: str = GENERATOR, decoder_name: str = DECODER) -> tuple[Path, Path]:
    """Write <models_dir>/<generator_name>/ and <models_dir>/<decoder_name>/; returns both folders."""
    from safetensors.torch import save_file
    from transformers import ViTMAEConfig

    from scale_rae_comfy import runtime
    from scale_rae_comfy.qwen2 import Qwen2Stack

    runtime.vendored()
    from scale_rae.model.diffusion_loss.diffloss import create_rf_projector
    from scale_rae.model.multimodal_decoder.decoder import GeneralDecoder

    g = torch.Generator().manual_seed(seed)
    gen_dir, dec_dir = Path(models_dir) / generator_name, Path(models_dir) / decoder_name
    gen_dir.mkdir(parents=True, exist_ok=True)
    dec_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(GENERATOR_CONFIG)
    llm = Qwen2Stack(cfg, runtime.NUM_IMAGE_TOKENS)
    diff_head = create_rf_projector(runtime.diff_head_config(cfg))
    projector = torch.nn.Linear(cfg["hidden_size"], cfg["diffusion_model_z_channels"])
    state = {}
    for prefix, module in (("model.", llm), ("diff_head.", diff_head), ("diff_head_projector.", projector)):
        _randomize(module, g)
        state.update({prefix + k: v.detach().clone().contiguous() for k, v in module.state_dict().items()})
    state["model.mm_projector.0.weight"] = torch.zeros(4, 4)  # present in the official file, unused for text-to-image
    save_file(state, str(gen_dir / "model.safetensors"))
    (gen_dir / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    for name in TOKENIZER_FILES:
        shutil.copyfile(Path(tokenizer_dir) / name, gen_dir / name)

    dec_cfg = ViTMAEConfig(**{k: v for k, v in DECODER_CONFIG.items() if k != "model_type"})
    dec_cfg._attn_implementation = "eager"
    decoder = GeneralDecoder(dec_cfg, num_patches=runtime.NUM_IMAGE_TOKENS)
    _randomize(decoder, g)
    torch.save(decoder.state_dict(), str(dec_dir / "model.pt"))
    (dec_dir / "config.json").write_text(json.dumps(DECODER_CONFIG, indent=1), encoding="utf-8")
    return gen_dir, dec_dir
