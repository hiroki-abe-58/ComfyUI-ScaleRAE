"""Runtime on the CPU with a tiny random model in the official layout: prompt handling (against ids recorded from the official CLI),
loading checks, sampling, guidance modes and reproducibility. The trained model is compared with the official code in
docs/VERIFICATION.md."""

import json
from pathlib import Path

import pytest
import torch

from scale_rae_comfy import runtime

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = json.loads((ROOT / "tests" / "data" / "token_ids.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def loaded(tiny_models):
    handle = runtime.make_handle(str(tiny_models["generator"]), str(tiny_models["decoder"]), "float32", "cpu", True)
    yield runtime.acquire(handle)
    runtime.release()


def gen(loaded, **kw):
    args = {"prompt": "Can you create an image of a peaceful garden?", "seed": 1, "guidance": 1.0, "steps": 6, "image_start": "force"}
    args.update(kw)
    return runtime.generate(loaded, **args)


@pytest.mark.parametrize("case", OFFICIAL["prompts"], ids=lambda c: c["prompt"][:24])
def test_prompt_template_and_ids_match_the_official_cli(loaded, case):
    assert runtime.chat_text(case["prompt"]) == case["chat_text"]
    assert runtime.build_input_ids(loaded, case["prompt"])[0].tolist() == case["ids"]


def test_special_token_ids(loaded):
    assert loaded.start_image_token_id == OFFICIAL["special"]["<im_start>"]
    assert loaded.eos_token_id == OFFICIAL["special"]["<|im_end|>"]


def test_prompts_that_the_official_code_cannot_handle_are_rejected(loaded):
    with pytest.raises(ValueError, match="<image>"):
        runtime.build_input_ids(loaded, "draw <image> please")
    with pytest.raises(ValueError, match="empty"):
        gen(loaded, prompt="   ")


def test_image_shape_range_and_report(loaded):
    out = gen(loaded)
    assert out["image"].shape == (1, 224, 224, 3) and out["image"].dtype == torch.float32
    assert 0.0 <= float(out["image"].min()) and float(out["image"].max()) <= 1.0
    assert out["latent"].shape == (256, 1024)
    report = out["report"]
    assert report["dit_calls"] == 5 and report["width"] == 224 and report["height"] == 224 and report["image_forced"] is True
    assert report["text_tokens_before_image"] == 0


def test_same_seed_same_image_other_seed_other_image(loaded):
    a, b, c = gen(loaded, seed=3), gen(loaded, seed=3), gen(loaded, seed=4)
    assert torch.equal(a["image"], b["image"]) and torch.equal(a["latent"], b["latent"])
    assert not torch.equal(a["latent"], c["latent"])


def test_global_random_state_is_left_alone(loaded):
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    gen(loaded)
    assert torch.equal(torch.rand(3), expected)


def test_steps_change_the_number_of_dit_calls(loaded):
    assert gen(loaded, steps=3)["report"]["dit_calls"] == 2
    assert gen(loaded, steps=6)["report"]["dit_calls"] == 5  # and back
    assert len(loaded.diff_head.inference_flow.used_timesteps) == 6
    with pytest.raises(ValueError):
        gen(loaded, steps=1)


def test_guidance_modes(loaded):
    base = gen(loaded, guidance=1.0)["latent"]
    standard = gen(loaded, guidance=4.0, cfg_mode="standard")["latent"]
    official = gen(loaded, guidance=4.0, cfg_mode="official-cli")["latent"]
    # official-cli keeps cfg_interval [-1e4, 1e4]: forward_with_cfg returns the conditional half, so guidance changes nothing but
    # the rounding of a two-row batch (ZitengWangNYU/Scale-RAE#7); standard applies the guidance
    assert torch.allclose(official, base, atol=1e-4)
    assert (standard - base).abs().max() > 1e-2
    assert torch.equal(gen(loaded, guidance=1.0, cfg_mode="official-cli")["latent"], base)
    with pytest.raises(ValueError):
        gen(loaded, cfg_mode="other")


def test_the_dit_forward_is_restored_after_errors(loaded):
    def boom(n):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        gen(loaded, progress=boom)
    assert "forward" not in vars(loaded.diff_head.model)
    assert gen(loaded)["report"]["dit_calls"] == 5


class _Scripted:
    """Stands in for the language model head: returns the given token ids one after the other."""

    def __init__(self, llm, tokens):
        self.llm, self.tokens = llm, list(tokens)

    def __call__(self, hidden):
        logits = torch.zeros(1, 1, self.llm.embed_tokens.num_embeddings)
        logits[0, 0, self.tokens.pop(0)] = 1.0
        return logits


def test_text_before_the_image_is_reported(loaded, monkeypatch):
    hello = loaded.tokenizer("Sure!").input_ids
    monkeypatch.setattr(loaded.llm, "logits", _Scripted(loaded.llm, [*hello, loaded.start_image_token_id]))
    report = gen(loaded, image_start="model")["report"]
    assert report["text_tokens_before_image"] == len(hello) and report["text_before_image"] == "Sure!"


def test_a_text_only_answer_raises_with_a_hint(loaded, monkeypatch):
    hello = loaded.tokenizer("A garden.").input_ids
    monkeypatch.setattr(loaded.llm, "logits", _Scripted(loaded.llm, [*hello, loaded.eos_token_id]))
    with pytest.raises(ValueError, match="image_start"):
        gen(loaded, image_start="model")


def test_auto_falls_back_to_force_after_a_text_only_answer(loaded, monkeypatch):
    forced = gen(loaded, image_start="force")
    answer = loaded.tokenizer("A garden.").input_ids
    monkeypatch.setattr(loaded.llm, "logits", _Scripted(loaded.llm, [*answer, loaded.eos_token_id]))
    out = gen(loaded, image_start="auto")
    assert torch.equal(out["latent"], forced["latent"])  # same seed, image started right after the prompt
    assert out["report"]["image_forced"] is True and out["report"]["text_only_answer"] == "A garden."


def test_auto_is_the_official_path_when_the_model_starts_the_image(loaded, monkeypatch):
    hello = loaded.tokenizer("Sure!").input_ids
    monkeypatch.setattr(loaded.llm, "logits", _Scripted(loaded.llm, [*hello, loaded.start_image_token_id]))
    auto = gen(loaded, image_start="auto")
    monkeypatch.setattr(loaded.llm, "logits", _Scripted(loaded.llm, [*hello, loaded.start_image_token_id]))
    official = gen(loaded, image_start="model")
    assert torch.equal(auto["latent"], official["latent"]) and auto["report"]["image_forced"] is False
    assert "text_only_answer" not in auto["report"]
    with pytest.raises(ValueError):
        gen(loaded, image_start="other")


def test_loading_inside_inference_mode_gives_ordinary_parameters(tiny_models):
    """ComfyUI runs nodes under torch.inference_mode(); the weights must still be ordinary tensors with requires_grad=True (the
    official numerics of the guidance batch depend on it, docs/VERIFICATION.md)."""
    runtime.release()
    handle = runtime.make_handle(str(tiny_models["generator"]), str(tiny_models["decoder"]), "float32", "cpu", False)
    with torch.inference_mode():
        loaded = runtime.acquire(handle)
    params = [p for m in (loaded.llm, loaded.diff_head, loaded.decoder) for p in m.parameters()]
    assert not any(p.is_inference() for p in params) and all(p.requires_grad for p in params)
    runtime.release()


def test_cpu_always_uses_float32(tiny_models):
    runtime.release()
    loaded = runtime.acquire(runtime.make_handle(str(tiny_models["generator"]), str(tiny_models["decoder"]), "bfloat16", "cpu", False))
    assert loaded.dtype == torch.float32 and loaded.info["dtype"] == "float32"
    runtime.release()
    with pytest.raises(ValueError):
        runtime.make_handle("a", "b", "int8", "cpu", True)


def test_incomplete_or_foreign_checkpoints_are_refused(tiny_models, tmp_path):
    import shutil

    from safetensors.torch import load_file, save_file

    gen_dir = tmp_path / "gen"
    shutil.copytree(tiny_models["generator"], gen_dir)
    state = load_file(str(gen_dir / "model.safetensors"))
    del state["model.latent_queries"]
    save_file(state, str(gen_dir / "model.safetensors"))
    runtime.release()
    with pytest.raises(ValueError, match="lacks"):
        runtime.acquire(runtime.make_handle(str(gen_dir), str(tiny_models["decoder"]), "float32", "cpu", False))

    other = tmp_path / "other"
    shutil.copytree(tiny_models["generator"], other)
    cfg = json.loads((other / "config.json").read_text(encoding="utf-8"))
    cfg["mm_vision_tower_aux_list"] = ["google/siglip2-so400m-patch14-224"]
    (other / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="encoder space"):
        runtime.acquire(runtime.make_handle(str(other), str(tiny_models["decoder"]), "float32", "cpu", False))
    runtime.release()


def test_head_configurations_the_pack_does_not_run_are_refused():
    """Normalisation-stats files (the only path where the vendored head would torch.load a file), other loss modes and DDT heads."""
    from tiny_model import GENERATOR_CONFIG

    with pytest.raises(ValueError, match="diffusion_norm_stats_path"):
        runtime.diff_head_config({**GENERATOR_CONFIG, "diffusion_norm_stats_path": "stats.pt"})
    with pytest.raises(ValueError, match="vision_loss_mode"):
        runtime.diff_head_config({**GENERATOR_CONFIG, "vision_loss_mode": "token"})
    with pytest.raises(ValueError, match="DDT"):
        runtime.diff_head_config({**GENERATOR_CONFIG, "diffusion_model_hidden_size": [64, 64]})


def test_vendored_package_is_registered_once():
    assert runtime.vendored() is runtime.vendored()
