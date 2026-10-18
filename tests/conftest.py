"""Test setup. References come from environment variables, never from this repository:

SCALE_RAE_COMFYUI_DIR      ComfyUI checkout (v0.38.0 or later)
SCALE_RAE_UPSTREAM_DIR     ZitengWangNYU/Scale-RAE checkout at the commit in scale_rae_comfy/vendor/UPSTREAM.md
SCALE_RAE_TOKENIZER_DIR    the five tokenizer files of nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL (revision and SHA-256 in
                           tests/tokenizer_files.sha256; about 5 MB, no weights)
SCALE_RAE_REQUIRE_REFERENCES=1  fail instead of skip when one of them is missing (CI sets this)
No trained weights are used: the model tests build a tiny random model in the official file layout and run it on the CPU.
"""

import hashlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
REQUIRE = os.environ.get("SCALE_RAE_REQUIRE_REFERENCES") == "1"


def _path(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _path("SCALE_RAE_COMFYUI_DIR")
UPSTREAM = _path("SCALE_RAE_UPSTREAM_DIR")
TOKENIZER = _path("SCALE_RAE_TOKENIZER_DIR")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (SCALE_RAE_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (SCALE_RAE_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def upstream_dir():
    return Path(_need(UPSTREAM, "Scale-RAE checkout (SCALE_RAE_UPSTREAM_DIR)"))


@pytest.fixture(scope="session")
def tokenizer_dir():
    directory = Path(_need(TOKENIZER, "Scale-RAE tokenizer files (SCALE_RAE_TOKENIZER_DIR)"))
    for line in (ROOT / "tests" / "tokenizer_files.sha256").read_text(encoding="ascii").splitlines():
        if line and not line.startswith("#"):
            digest, _, name = line.partition("  ")
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, f"{name} differs from the pinned revision"
    return directory


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory, tokenizer_dir):
    """<tmp>/scale_rae/<generator>/ and <decoder>/ with tiny random weights."""
    from tiny_model import make_tiny_models

    base = tmp_path_factory.mktemp("models") / "scale_rae"
    gen, dec = make_tiny_models(base, tokenizer_dir)
    return {"base": base, "generator": gen, "decoder": dec}
