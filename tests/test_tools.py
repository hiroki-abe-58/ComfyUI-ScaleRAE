"""tools/download_models.py: the hash manifest covers every file the loader needs and agrees with the tokenizer hashes of the tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import download_models  # noqa: E402

from scale_rae_comfy import paths  # noqa: E402


def test_manifest_covers_what_the_loader_needs():
    files = download_models.expected_files()
    assert set(files) == set(download_models.MODELS)
    generator, decoder = files["Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL"], files["webssl300m_decoder"]
    assert {"config.json", "model.safetensors.index.json", *paths.GENERATOR_FILES} <= set(generator)
    assert {f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)} <= set(generator)
    assert {"config.json", "model.pt"} <= set(decoder)
    assert all(len(d) == 64 for f in files.values() for d in f.values())


def test_tokenizer_hashes_agree_with_the_tests():
    generator = download_models.expected_files()["Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL"]
    for line in (ROOT / "tests" / "tokenizer_files.sha256").read_text(encoding="ascii").splitlines():
        if line and not line.startswith("#"):
            digest, _, name = line.partition("  ")
            assert generator[name] == digest, name
