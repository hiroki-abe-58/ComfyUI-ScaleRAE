"""Loading and text-to-image generation for Scale-RAE (one resident model).

The steps are those of the official `inference/cli.py t2i` with `ScaleRAEQwenForCausalLM.generate(use_customize_greedy=True)`:

1. prompt -> Qwen chat template (`conv_templates["qwen_2"]`) -> token ids (`chat_text`, `build_input_ids`);
2. greedy decoding with the full sequence recomputed each step (no KV cache) until the model emits `<im_start>`;
3. `<im_start>` and the 256 learned latent queries are appended; one more forward pass gives their hidden states;
4. `diff_head_projector`, then the rectified-flow DiT head (`FullSequenceRectifiedFlowProjector.infer`, 50 Euler steps, optional
   classifier-free guidance) turns them into 256 encoder-space tokens (the "RAE latent");
5. the RAE decoder (`GeneralDecoder`) maps those tokens to pixels, which are de-normalised with the encoder's image statistics.

The language model is `qwen2.Qwen2Stack` (plain PyTorch, see its docstring); the DiT head and the decoder are the vendored upstream
modules (vendor/UPSTREAM.md). Nothing here downloads anything.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .qwen2 import Qwen2Stack

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
NUM_IMAGE_TOKENS = 256  # ScaleRAEQwenForCausalLM.num_image_tokens ("fixed")

# The RAE decoders reconstruct pixels from the features of a frozen vision encoder. Only two facts about that encoder are needed:
# its feature width and the image normalisation of its processor (`preprocessor_config.json` of the encoder repository).
ENCODERS = {
    "facebook/webssl-dino300m-full2b-224": {"hidden_size": 1024, "image_mean": (0.485, 0.456, 0.406), "image_std": (0.229, 0.224, 0.225)},
}

_LOCK = threading.Lock()
_SLOT = None


def vendored():
    """The vendored upstream package, registered under its own name `scale_rae` (its modules use absolute imports such as
    `from scale_rae.utils import ...`; submodules then load from the vendored folder). sys.path is not changed."""
    from .vendor import scale_rae as package

    current = sys.modules.setdefault("scale_rae", package)
    if current is not package and [Path(p).resolve() for p in getattr(current, "__path__", [])] != [Path(p).resolve() for p in package.__path__]:
        raise RuntimeError("another package named 'scale_rae' is already imported in this Python process; it conflicts with this node pack")
    return current  # the same folder imported under two package names (e.g. by ComfyUI and by the tests) is the same code


@contextlib.contextmanager
def _skip_init():
    """Construct modules without random initialisation (all weights are overwritten from the checkpoint right after)."""
    names = ["normal_", "trunc_normal_", "uniform_", "xavier_uniform_", "xavier_normal_", "kaiming_uniform_", "kaiming_normal_", "constant_", "zeros_", "ones_"]
    saved = {n: getattr(torch.nn.init, n) for n in names}
    try:
        for n in names:
            setattr(torch.nn.init, n, lambda tensor, *a, **k: tensor)
        yield
    finally:
        for n, f in saved.items():
            setattr(torch.nn.init, n, f)


def diff_head_config(cfg: dict) -> dict:
    """`self.diff_head_config` of ScaleRAEQwenForCausalLM.__init__ for vision_loss 'diffusion-loss' + vision_loss_mode 'query' (upstream
    scale_rae_qwen2.py, the non-DDT branch), with `create_rf_projector`'s `inference_step=50`."""
    if cfg.get("vision_loss", "diffusion-loss") != "diffusion-loss" or cfg.get("vision_loss_mode") != "query":
        raise ValueError(f"unsupported Scale-RAE variant: vision_loss={cfg.get('vision_loss')!r}, vision_loss_mode={cfg.get('vision_loss_mode')!r}")
    if not isinstance(cfg["diffusion_model_hidden_size"], int):
        raise ValueError("DDT-style diffusion heads are not supported")
    out = {
        "diffusion_tokens": cfg.get("vision_tower_aux_token_len_list", [NUM_IMAGE_TOKENS])[0],
        "diffusion_channels": cfg.get("diffusion_model_channels", 1152),
        "z_channels": cfg["hidden_size"] if cfg["diffusion_model_z_channels"] == 0 else cfg["diffusion_model_z_channels"],
        "model_hidden_size": cfg["diffusion_model_hidden_size"],
        "model_depth": cfg["diffusion_model_depth"],
        "model_heads": cfg["diffusion_model_heads"],
        "guidance_scale": 1.0,
        "cond_silu": True,
        "class_dropout_prob": cfg.get("diffusion_class_dropout_prob", 0.0),
    }
    if cfg.get("diffusion_base_dim") is not None:
        out["base_dim"] = cfg["diffusion_base_dim"]
    if cfg.get("diffusion_norm_stats_path"):
        raise ValueError("checkpoints that need diffusion_norm_stats_path are not supported")
    out["use_mlp"] = False
    out["dit_cls"] = cfg.get("dit_cls", "DiT")
    return out


@dataclass
class Loaded:
    key: tuple
    device: torch.device
    dtype: torch.dtype
    tokenizer: object
    llm: Qwen2Stack
    projector: torch.nn.Module | None
    diff_head: torch.nn.Module
    decoder: torch.nn.Module
    image_mean: torch.Tensor
    image_std: torch.Tensor
    start_image_token_id: int
    eos_token_id: int
    info: dict = field(default_factory=dict)


def _read_state(model_dir: Path, device: torch.device):
    from safetensors import safe_open

    index_file = model_dir / "model.safetensors.index.json"
    files = sorted(set(json.loads(index_file.read_text(encoding="utf-8"))["weight_map"].values())) if index_file.exists() else ["model.safetensors"]
    state = {}
    for name in files:
        with safe_open(str(model_dir / name), framework="pt", device=str(device) if device.type == "cuda" else "cpu") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
    return state


def _assign(module: torch.nn.Module, state: dict, prefix: str, dtype: torch.dtype) -> list:
    """Copy `prefix`-ed tensors into `module` (shape-checked); return the module's parameter/buffer names that were not in the file."""
    own = dict(module.named_parameters())
    own.update(dict(module.named_buffers()))
    missing = []
    for name, tensor in own.items():
        key = prefix + name
        if key not in state:
            missing.append(name)
            continue
        src = state[key]
        if tuple(src.shape) != tuple(tensor.shape):
            raise ValueError(f"{key}: checkpoint shape {tuple(src.shape)} vs model {tuple(tensor.shape)}")
        tensor.data = src.to(dtype if tensor.is_floating_point() else tensor.dtype)
    return missing


def make_handle(model_dir: str, decoder_dir: str, dtype: str, device: str, keep_loaded: bool) -> dict:
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {list(DTYPES)}")
    return {"model_dir": model_dir, "decoder_dir": decoder_dir, "dtype": dtype, "device": resolve_device(device), "keep_loaded": bool(keep_loaded)}


def resolve_device(choice: str) -> str:
    if choice == "auto":
        try:
            import comfy.model_management as mm

            return str(mm.get_torch_device())
        except Exception:  # noqa: BLE001 - outside ComfyUI
            return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def release() -> None:
    global _SLOT
    with _LOCK:
        _SLOT = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def acquire(handle: dict) -> Loaded:
    global _SLOT
    key = (handle["model_dir"], handle["decoder_dir"], handle["dtype"], handle["device"])
    with _LOCK:
        if _SLOT is not None and _SLOT.key == key:
            return _SLOT
        _SLOT = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ComfyUI runs nodes under torch.inference_mode(); weights created there would be inference tensors, which PyTorch's matmul
        # treats like requires_grad=False (see _load). Load them as ordinary tensors, as the official loader does.
        with torch.inference_mode(False):
            _SLOT = _load(key)
        return _SLOT


def _load(key) -> Loaded:
    model_dir, decoder_dir, dtype_name, device_name = key
    model_dir, decoder_dir = Path(model_dir), Path(decoder_dir)
    device, dtype = torch.device(device_name), DTYPES[dtype_name]
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32
    started = time.perf_counter()
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    encoder = cfg.get("mm_vision_tower_aux_list", [""])[0].split("-interp")[0]
    if encoder not in ENCODERS:
        raise ValueError(f"unsupported encoder space {encoder!r} (supported: {sorted(ENCODERS)})")
    enc = ENCODERS[encoder]

    vendored()
    from scale_rae.model.diffusion_loss.diffloss import create_rf_projector
    from scale_rae.model.multimodal_decoder.decoder import GeneralDecoder
    from transformers import AutoTokenizer, ViTMAEConfig

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
    state = _read_state(model_dir, device)

    with _skip_init():
        llm = Qwen2Stack(cfg, NUM_IMAGE_TOKENS)
        diff_head = create_rf_projector(diff_head_config(cfg))
        projector = None
        if cfg.get("diffusion_model_z_channels", 0) != 0:
            projector = torch.nn.Linear(cfg["hidden_size"], cfg["diffusion_model_z_channels"])
        dec_cfg = ViTMAEConfig.from_pretrained(str(decoder_dir))
        dec_cfg.hidden_size = enc["hidden_size"]
        dec_cfg._attn_implementation = "eager"  # what transformers 4.37 gives a ViTMAEConfig read on its own
        decoder = GeneralDecoder(dec_cfg, num_patches=NUM_IMAGE_TOKENS)
    # non-persistent buffers (rotary tables, decoder position embedding) were computed during construction, on the CPU in float32
    missing = ["llm." + m for m in _assign(llm, state, "model.", dtype) if not m.endswith(("rotary.cos_cached", "rotary.sin_cached"))]
    missing += ["diff_head." + m for m in _assign(diff_head, state, "diff_head.", dtype)]
    if projector is not None:
        missing += ["diff_head_projector." + m for m in _assign(projector, state, "diff_head_projector.", dtype)]
    if missing:
        raise ValueError(f"the checkpoint lacks {len(missing)} tensors, e.g. {missing[:5]}")
    del state

    dec_state = torch.load(str(decoder_dir / "model.pt"), map_location="cpu", weights_only=True)
    result = decoder.load_state_dict(dec_state, strict=False)
    if result.missing_keys:  # upstream loads with strict=False; the pack insists on a complete decoder
        raise ValueError(f"the decoder checkpoint lacks {result.missing_keys[:5]}")
    del dec_state

    for module in (llm, diff_head, decoder) + ((projector,) if projector is not None else ()):
        # The RAE decoder (0.4 B parameters, one pass per image) always runs in float32: in float16 it overflows and every image comes
        # out as the same flat picture (measured; docs/VERIFICATION.md).
        module.to(device=device, dtype=torch.float32 if module is decoder else dtype)
        module.eval()
        # Parameters keep requires_grad=True, as in the official model. Generation runs under inference_mode, so no autograd state is
        # kept, but PyTorch's matmul folds batched inputs differently when the weight requires grad: with requires_grad=False the
        # two-row classifier-free-guidance batch of the DiT rounds differently from the official run (measured; docs/VERIFICATION.md).
        module.requires_grad_(True)

    start_id = tokenizer.convert_tokens_to_ids("<im_start>")
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    info = {
        "model_dir": model_dir.name,
        "decoder_dir": decoder_dir.name,
        "encoder_space": encoder,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "load_seconds": round(time.perf_counter() - started, 2),
        "parameters_b": round(sum(p.numel() for m in (llm, diff_head, decoder) for p in m.parameters()) / 1e9, 3),
        "unused_checkpoint_tensors": "model.mm_projector.* (image-understanding path)",
        "decoder_unexpected_keys": len(result.unexpected_keys),
    }
    return Loaded(key=key, device=device, dtype=dtype, tokenizer=tokenizer, llm=llm, projector=projector, diff_head=diff_head, decoder=decoder,
                  image_mean=torch.tensor(enc["image_mean"], device=device).view(1, 3, 1, 1),
                  image_std=torch.tensor(enc["image_std"], device=device).view(1, 3, 1, 1),
                  start_image_token_id=int(start_id), eos_token_id=int(eos_id), info=info)


def chat_text(prompt: str) -> str:
    """`cli.build_prompt(prompt, with_image=False)`: upstream `conv_templates["qwen_2"]` (SeparatorStyle.CHATML) with the user message
    and the open assistant turn."""
    return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def build_input_ids(loaded: Loaded, prompt: str) -> torch.Tensor:
    """`cli.tokenize_prompt(cli.build_prompt(prompt, with_image=False))`. Upstream `tokenizer_image_token` splits the text at '<image>'
    and tokenizes the pieces; without '<image>' that is the tokenizer's ids of the whole text (tests/data/token_ids.json has the
    official ids)."""
    if "<image>" in prompt:  # upstream would turn it into an image slot, and text-to-image has no input image
        raise ValueError("the prompt must not contain '<image>'")
    if not prompt:
        raise ValueError("prompt is empty")  # upstream drops an empty message from the template
    return torch.tensor([loaded.tokenizer(chat_text(prompt)).input_ids], dtype=torch.long)


@contextlib.contextmanager
def _seeded(seed: int, device: torch.device):
    """The official `set_seed` (torch + CUDA seeds, deterministic cuDNN) inside a forked RNG, restored afterwards."""
    devices = [device] if device.type == "cuda" else []
    deterministic, benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        try:
            yield
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = deterministic, benchmark


def _set_steps(diff_head, steps: int):
    """Rebuild the projector's inference flow for `steps` sampling steps (the official checkpoint is built with 50)."""
    vendored()
    from scale_rae.model.diffusion_loss.diffusion import create_diffusion

    if len(diff_head.inference_flow.used_timesteps) == steps:
        return
    diff_head.inference_flow = create_diffusion(
        str(steps), noise_schedule="linear", use_kl=False, sigma_small=False, predict_xstart=False, learn_sigma=False,
        rescale_learned_sigmas=False, diffusion_steps=1000, input_base_dimension_ratio=diff_head.inference_flow.size_ratio,
        diffusion_type="rf", use_loss_weighting=False, use_schedule_shift=True, diffusion_kwargs=None)


CFG_MODES = ("standard", "official-cli")
IMAGE_STARTS = ("auto", "model", "force")


class TextAnswer(Exception):
    """The model ended its answer (or ran past max_new_tokens) without starting an image; the text is the exception argument."""


def _decode_until_image(loaded: Loaded, embeds: torch.Tensor, max_new_tokens: int):
    """The official `greedy_decode`: argmax decoding with the full sequence recomputed at every step (no KV cache) until the model emits
    `<im_start>`. Returns the prompt + answer embeddings and the answer token ids."""
    llm = loaded.llm
    generated = []
    while True:
        hidden = llm(embeds)
        token = int(torch.argmax(llm.logits(hidden[:, -1:, :]), dim=-1).item())
        if token == loaded.start_image_token_id:
            return embeds, generated
        generated.append(token)
        if token == loaded.eos_token_id or len(generated) > max_new_tokens:
            raise TextAnswer(loaded.tokenizer.decode(generated, skip_special_tokens=True))
        embeds = torch.cat((embeds, llm.embed_tokens(torch.tensor([[token]], device=embeds.device))), dim=1)


@torch.no_grad()
def generate(loaded: Loaded, prompt: str, seed: int, guidance: float, steps: int = 50, max_new_tokens: int = 512, progress=None,
             cfg_mode: str = "standard", image_start: str = "model") -> dict:
    """Returns {"image": float [1, H, W, 3] in 0..1 on the CPU, "latent": [256, C] float32 on the CPU, "report": dict}.

    cfg_mode: with guidance > 1, "standard" applies classifier-free guidance at every step (uncond + g * (cond - uncond));
    "official-cli" keeps the released default `cfg_interval = [-1e4, 1e4]`, under which `forward_with_cfg` returns the conditional
    prediction at every step, so guidance has no effect beyond rounding (upstream issue ZitengWangNYU/Scale-RAE#7). With guidance 1
    both modes are the official code path.
    image_start: the model is a chat model; it writes a short text and then starts the image itself, or answers with text only (it did
    for every plain description we tried and for none of the requests such as "Can you generate a photo of ...?"; the text part is
    greedy, so this depends on the prompt, not on the seed). "model" = the official behaviour, a text-only answer is an error;
    "force" = the image starts right after the prompt; "auto" = "model", and after a text-only answer the same as "force" (same seed).
    """
    if cfg_mode not in CFG_MODES:
        raise ValueError(f"cfg_mode must be one of {CFG_MODES}")
    if image_start not in IMAGE_STARTS:
        raise ValueError(f"image_start must be one of {IMAGE_STARTS}")
    if not prompt.strip():
        raise ValueError("prompt is empty")
    if steps < 2:
        raise ValueError("steps must be at least 2")
    device = loaded.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    input_ids = build_input_ids(loaded, prompt).to(device)
    llm = loaded.llm
    generated, text_answer = [], None
    calls = {"n": 0}

    dit = loaded.diff_head.model
    dit_forward = dit.forward  # the sampler calls `model.forward` directly (no module hooks), so count through the instance attribute

    def counted_forward(*args, **kwargs):
        calls["n"] += 1
        if progress is not None:
            progress(calls["n"])
        return dit_forward(*args, **kwargs)

    dit.forward = counted_forward
    try:
        with _seeded(int(seed), device), torch.inference_mode():
            _set_steps(loaded.diff_head, int(steps))
            loaded.diff_head.cfg_interval = [-1e4, -1e4] if cfg_mode == "standard" else [-1e4, 1e4]
            embeds = llm.embed_tokens(input_ids)
            if image_start != "force":
                try:
                    embeds, generated = _decode_until_image(loaded, embeds, max_new_tokens)
                except TextAnswer as answer:
                    if image_start == "model":
                        raise ValueError("the model answered with text instead of an image (phrase the prompt as a request such as 'Can you "
                                         "generate a photo of ...?', or set image_start to auto or force): " + repr(str(answer)[:200])) from None
                    text_answer = str(answer)  # auto: start the image right after the prompt (greedy text used no random numbers)
            start = llm.embed_tokens(torch.tensor([[loaded.start_image_token_id]], device=device))
            queries = llm.latent_queries.unsqueeze(0).expand(embeds.size(0), -1, -1)
            hidden = llm(torch.cat((embeds, start, queries), dim=1))
            pred_z = hidden[:, -NUM_IMAGE_TOKENS:, :].squeeze(1)
            if loaded.projector is not None:
                pred_z = loaded.projector(pred_z)
            latent = loaded.diff_head.infer(pred_z, guidance_level=float(guidance))  # [1, 256, C]
            features = torch.cat([torch.zeros((1, 1, latent.shape[-1]), device=device), latent.float()], dim=1)
            out = loaded.decoder(features, drop_cls_token=True)
            pixels = loaded.decoder.unpatchify(out.logits)
            pixels = pixels * loaded.image_std.to(pixels.dtype) + loaded.image_mean.to(pixels.dtype)
            image = pixels.permute(0, 2, 3, 1).clip(0, 1).float().cpu()
    finally:
        del dit.forward  # back to the class method
    elapsed = time.perf_counter() - started
    report = {
        "seed": int(seed),
        "guidance": float(guidance),
        "cfg_mode": cfg_mode,
        "steps": int(steps),
        "image_start": image_start,
        "image_forced": image_start == "force" or text_answer is not None,
        "text_before_image": loaded.tokenizer.decode(generated, skip_special_tokens=True),
        "text_tokens_before_image": len(generated),
        "width": int(image.shape[2]),
        "height": int(image.shape[1]),
        "dit_calls": calls["n"],
        "generation_seconds": round(elapsed, 3),
    }
    if text_answer is not None:
        report["text_only_answer"] = text_answer[:300]
    if device.type == "cuda":
        report["peak_vram_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
    return {"image": image, "latent": latent[0].float().cpu(), "report": report}


def report_json(info: dict, report: dict) -> str:
    return json.dumps({"model": {k: info[k] for k in ("model_dir", "decoder_dir", "dtype", "device")}, **report}, indent=1)
