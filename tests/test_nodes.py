"""The node pack as ComfyUI loads it: registration, model folder discovery, ComfyUI's own validation of the shipped workflows, and node
execution with the tiny random model on the CPU (the trained model runs in the real-queue checks of docs/VERIFICATION.md)."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NODE_IDS = {"ScaleRAELoader", "ScaleRAETextToImage", "ScaleRAEUnload"}


@pytest.fixture(scope="module")
def comfy_nodes(comfy_dir, tiny_models, tmp_path_factory):
    import folder_paths
    import nodes

    base = tiny_models["base"]
    (base / "incomplete").mkdir(exist_ok=True)
    (base / "incomplete" / "config.json").write_text(json.dumps({"model_type": "cambrian_qwen"}), encoding="utf-8")  # no weights
    (base / "other").mkdir(exist_ok=True)
    (base / "other" / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (base / "other" / "model.safetensors").write_bytes(b"")
    folder_paths.add_model_folder_path("scale_rae", str(base), is_default=True)

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert asyncio.run(nodes.load_custom_node(str(ROOT))) is True
    folder_paths.filename_list_cache.clear()
    return nodes


def node_runtime(comfy_nodes):
    """The runtime module the NODES use. ComfyUI imports the pack under a path-based name, so it is a different module object from
    `scale_rae_comfy.runtime` as the tests import it."""
    module_name = comfy_nodes.NODE_CLASS_MAPPINGS["ScaleRAETextToImage"].__module__.rsplit(".", 1)[0] + ".runtime"
    return sys.modules[module_name]


def test_registered(comfy_nodes):
    assert NODE_IDS <= set(comfy_nodes.NODE_CLASS_MAPPINGS)


def test_model_folders_are_told_apart_by_model_type_and_files(comfy_nodes):
    paths = sys.modules[comfy_nodes.NODE_CLASS_MAPPINGS["ScaleRAELoader"].__module__.rsplit(".", 1)[0] + ".paths"]

    assert paths.generators() == ["Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL"]  # not "incomplete" (no weights) and not "other" (wrong model_type)
    assert paths.decoders() == ["webssl300m_decoder"]
    for bad in ("..", "../x", "incomplete", "missing", "other", "webssl300m_decoder"):
        with pytest.raises(FileNotFoundError):
            paths.generator_path(bad)
    with pytest.raises(FileNotFoundError):
        paths.decoder_path("Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL")


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "workflows" / "api").glob("*.json")))
def test_api_workflows_pass_comfy_validation(comfy_nodes, name):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / name).read_text(encoding="utf-8"))
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert valid, (error, node_errors)


def test_validation_rejects_out_of_range_values(comfy_nodes):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / "t2i_basic.json").read_text(encoding="utf-8"))
    prompt["2"]["inputs"]["steps"] = 1
    valid, _, _, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert not valid and "2" in node_errors


def _handle(comfy_nodes, keep_loaded=True):
    loader = comfy_nodes.NODE_CLASS_MAPPINGS["ScaleRAELoader"]
    return loader.execute("Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL", "webssl300m_decoder", "float32", "cpu", keep_loaded).args[0]


def _t2i(comfy_nodes, handle, **kw):
    args = {"prompt": "Can you create an image of a peaceful garden?", "seed": 5, "guidance": 1.0, "cfg_mode": "standard", "steps": 4,
            "batch_size": 1, "image_start": "force"}
    args.update(kw)
    return comfy_nodes.NODE_CLASS_MAPPINGS["ScaleRAETextToImage"].execute(handle, **args).args


def test_text_to_image_node(comfy_nodes):
    image, report = _t2i(comfy_nodes, _handle(comfy_nodes))
    assert image.shape == (1, 224, 224, 3)
    parsed = json.loads(report)
    assert parsed["seed"] == 5 and parsed["dit_calls"] == 3 and parsed["model"]["device"] == "cpu"
    again, _ = _t2i(comfy_nodes, _handle(comfy_nodes))
    assert torch.equal(image, again)


def test_batch_uses_consecutive_seeds(comfy_nodes):
    handle = _handle(comfy_nodes)
    batch, report = _t2i(comfy_nodes, handle, batch_size=2)
    first, _ = _t2i(comfy_nodes, handle, seed=5)
    second, _ = _t2i(comfy_nodes, handle, seed=6)
    assert batch.shape[0] == 2 and torch.equal(batch[0:1], first) and torch.equal(batch[1:2], second)
    assert [r["seed"] for r in json.loads(report)["images"]] == [5, 6]


def test_batch_seeds_wrap_at_the_largest_seed(comfy_nodes):
    handle = _handle(comfy_nodes)
    batch, report = _t2i(comfy_nodes, handle, seed=2**64 - 1, batch_size=2)
    zero, _ = _t2i(comfy_nodes, handle, seed=0)
    assert [r["seed"] for r in json.loads(report)["images"]] == [2**64 - 1, 0] and torch.equal(batch[1:2], zero)


def test_auto_decodes_the_text_once_per_batch(comfy_nodes, monkeypatch):
    runtime = node_runtime(comfy_nodes)
    calls = {"n": 0}

    def text_only(loaded, embeds, max_new_tokens):
        calls["n"] += 1
        raise runtime.TextAnswer("A sunny beach is a place where the sun shines.")

    monkeypatch.setattr(runtime, "_decode_until_image", text_only)
    handle = _handle(comfy_nodes)
    batch, report = _t2i(comfy_nodes, handle, image_start="auto", batch_size=3, seed=20)
    forced, _ = _t2i(comfy_nodes, handle, image_start="force", batch_size=3, seed=20)
    assert calls["n"] == 1 and torch.equal(batch, forced)
    images = json.loads(report)["images"]
    assert all(r["image_start"] == "auto" and r["image_forced"] and r["text_only_answer"].startswith("A sunny beach") for r in images)
    with pytest.raises(ValueError, match="text instead of an image"):
        _t2i(comfy_nodes, handle, image_start="model")


def test_keep_loaded_false_releases_and_unload_is_safe(comfy_nodes):
    runtime = node_runtime(comfy_nodes)
    _t2i(comfy_nodes, _handle(comfy_nodes, keep_loaded=False))
    assert runtime._SLOT is None
    unload = comfy_nodes.NODE_CLASS_MAPPINGS["ScaleRAEUnload"]
    assert json.loads(unload.execute().args[0])["unloaded"] is False  # safe when nothing is loaded
    _t2i(comfy_nodes, _handle(comfy_nodes))
    assert json.loads(unload.execute().args[0])["unloaded"] is True and runtime._SLOT is None


def test_interrupt_stops_sampling(comfy_nodes, monkeypatch):
    import comfy.model_management as mm

    runtime = node_runtime(comfy_nodes)
    handle = _handle(comfy_nodes)
    calls = {"n": 0}

    def interrupt():
        calls["n"] += 1
        if calls["n"] > 2:
            raise mm.InterruptProcessingException()

    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", interrupt)
    with pytest.raises(mm.InterruptProcessingException):
        _t2i(comfy_nodes, handle)
    monkeypatch.undo()
    assert "forward" not in vars(runtime._SLOT.diff_head.model)
    image, _ = _t2i(comfy_nodes, handle)  # the resident model still works
    assert image.shape == (1, 224, 224, 3)


def test_shipped_gui_workflows_use_known_nodes():
    for path in sorted((ROOT / "workflows" / "gui").glob("*.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        types = {n["type"] for n in graph["nodes"]}
        assert types <= NODE_IDS | {"SaveImage", "PreviewAny", "PreviewImage"}, path.name

