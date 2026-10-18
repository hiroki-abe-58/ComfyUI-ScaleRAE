"""Model locations: `<ComfyUI models>/scale_rae/<name>/` (plus any path added for `scale_rae` in extra_model_paths.yaml).

Two kinds of folders are listed:
* a Scale-RAE generator: an official nyu-visionx/Scale-RAE-* folder (`config.json` with `"model_type": "cambrian_qwen"`, the safetensors
  shards with their index, and the tokenizer files);
* an RAE decoder: an official nyu-visionx/*_decoder folder (`config.json` with `"model_type": "vit_mae"` and `model.pt`).
Workflows choose names from these lists only; nothing is downloaded at run time.
"""

from __future__ import annotations

import json
import os

FOLDER = "scale_rae"
GENERATOR_FILES = ("tokenizer_config.json", "vocab.json", "merges.txt")


def register() -> None:
    import folder_paths

    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER), is_default=False)


def _config(directory: str) -> dict:
    try:
        with open(os.path.join(directory, "config.json"), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def is_generator(directory: str) -> bool:
    has_weights = os.path.isfile(os.path.join(directory, "model.safetensors.index.json")) or os.path.isfile(os.path.join(directory, "model.safetensors"))
    return (_config(directory).get("model_type") == "cambrian_qwen" and has_weights
            and all(os.path.isfile(os.path.join(directory, name)) for name in GENERATOR_FILES))


def is_decoder(directory: str) -> bool:
    return _config(directory).get("model_type") == "vit_mae" and os.path.isfile(os.path.join(directory, "model.pt"))


def _scan(test) -> list[str]:
    import folder_paths

    found = set()
    for base in folder_paths.get_folder_paths(FOLDER):
        if os.path.isdir(base):
            found.update(entry.name for entry in os.scandir(base) if entry.is_dir() and test(entry.path))
    return sorted(found)


def generators() -> list[str]:
    return _scan(is_generator)


def decoders() -> list[str]:
    return _scan(is_decoder)


def _resolve(name: str, test, what: str) -> str:
    """The folder of a listed name. Names come from the directory listing, so they never contain a path separator or '..'; a listed
    folder may itself be a symlink or junction to another drive."""
    import folder_paths

    for base in folder_paths.get_folder_paths(FOLDER):
        if os.path.isdir(base) and name in {entry.name for entry in os.scandir(base) if entry.is_dir()}:
            candidate = os.path.join(base, name)
            if test(candidate):
                return candidate
    raise FileNotFoundError(f"{what} folder {name!r} was not found in models/{FOLDER} (see the README for the download command)")


def generator_path(name: str) -> str:
    return _resolve(name, is_generator, "Scale-RAE generator")


def decoder_path(name: str) -> str:
    return _resolve(name, is_decoder, "RAE decoder")
