"""Download the official Scale-RAE text-to-image model and its RAE decoder into ComfyUI's models/scale_rae and verify them.

    python tools/download_models.py --models-dir <ComfyUI>/models/scale_rae

Sources (pinned to the revisions this node pack was verified with; model cards: MIT):
  https://huggingface.co/nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL   about 15.9 GB (float32 safetensors + tokenizer)
  https://huggingface.co/nyu-visionx/webssl300m_decoder                 about 1.7 GB
The WebSSL encoder itself (facebook/webssl-dino300m-full2b-224, CC-BY-NC-4.0) is NOT downloaded: text-to-image never runs it, and the
two numbers the decoder needs from it (feature width, image normalisation) are built into the node pack.
Every file is checked against its SHA-256 in tools/model_files.sha256. Needs `huggingface_hub` (installed with ComfyUI's
transformers). Nothing is downloaded unless you run this script.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

MODELS = {
    "Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL": ("nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL", "fd1e77fbe2f6eb96307a2e9512a7201ab7e3408f"),
    "webssl300m_decoder": ("nyu-visionx/webssl300m_decoder", "0e6a28d0744b31ac35624ecabadde20b540d8fa0"),
}
MANIFEST = Path(__file__).resolve().with_name("model_files.sha256")


def expected_files() -> dict[str, dict[str, str]]:
    """{folder: {file name: sha256}} from tools/model_files.sha256."""
    files = {name: {} for name in MODELS}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if line and not line.startswith("#"):
            digest, _, path = line.partition("  ")
            folder, _, name = path.partition("/")
            files[folder][name] = digest
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(folder: Path, files: dict) -> bool:
    ok = True
    for name, expected in files.items():
        path = folder / name
        good = path.is_file() and sha256(path) == expected
        ok &= good
        print(f"{'ok      ' if good else ('MISMATCH' if path.is_file() else 'MISSING ')} {folder.name}/{name}", flush=True)
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI's models/scale_rae folder")
    parser.add_argument("--only", choices=sorted(MODELS), help="download one of the two folders")
    parser.add_argument("--verify-only", action="store_true", help="check existing folders, download nothing")
    args = parser.parse_args(argv)

    expected = expected_files()
    ok = True
    for name, (repo, revision) in MODELS.items():
        if args.only and name != args.only:
            continue
        target = args.models_dir / name
        if not args.verify_only:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo, revision=revision, local_dir=str(target), allow_patterns=sorted(expected[name]))
        ok &= verify(target, expected[name])
    print("all files verified" if ok else "verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
