"""CPU "real queue" check for CI and for a clean-install test: start a real ComfyUI server with this node pack installed, and run the
shipped text-to-image workflow through POST /prompt with a tiny random model in the official folder layout.

    python scripts/ci_queue_check.py --comfyui <ComfyUI checkout> --tokenizer-dir <the five tokenizer files> [--workdir DIR]

The trained model (about 17.6 GB) cannot run in CI; this check proves that the pack installs into a stock ComfyUI, registers its nodes,
lists the model folders, and that Loader -> Text to Image -> SaveImage executes through the queue, reproducibly, with batches,
keep_loaded=false and the Unload node. The trained-model runs are documented in docs/VERIFICATION.md. The server is started and
stopped by this script only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import queue_client  # noqa: E402
from tiny_model import DECODER, GENERATOR, make_tiny_models  # noqa: E402

NODE_IDS = {"ScaleRAELoader", "ScaleRAETextToImage", "ScaleRAEUnload"}


def wait_up(server: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"ComfyUI exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{server}/system_stats", timeout=3).read()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("ComfyUI did not start in time")


def pixels_sha(path: Path) -> tuple[str, tuple]:
    import numpy as np
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("RGB"))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16], arr.shape


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8193)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-pack", action="store_true", help="use the pack already installed in custom_nodes (clean-install test)")
    args = parser.parse_args(argv)

    comfy = args.comfyui.resolve()
    work = (args.workdir or Path(tempfile.mkdtemp(prefix="scale-rae-ci-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    pack = comfy / "custom_nodes" / "ComfyUI-ScaleRAE"
    if pack.is_symlink() or getattr(os.path, "isjunction", lambda p: False)(pack):  # never delete through a link into a working copy
        raise SystemExit(f"{pack} is a link; use a ComfyUI checkout without it")
    if not args.keep_pack:
        if pack.exists():
            shutil.rmtree(pack)
        shutil.copytree(ROOT, pack, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs"))
    elif not pack.is_dir():
        raise SystemExit(f"--keep-pack: {pack} is not installed")
    models = work / "models"
    make_tiny_models(models / "scale_rae", args.tokenizer_dir)
    yaml = work / "extra_model_paths.yaml"
    yaml.write_text(f"scale_rae_ci:\n  base_path: {models.as_posix()}\n  scale_rae: scale_rae\n", encoding="utf-8")
    output = work / "output"

    log = (work / "comfyui.log").open("w", encoding="utf-8")
    server = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(args.port), "--disable-auto-launch", "--disable-api-nodes",
         "--extra-model-paths-config", str(yaml), "--output-directory", str(output)],
        cwd=comfy, stdout=log, stderr=subprocess.STDOUT,
    )
    failures = []
    try:
        wait_up(server, proc)
        info = json.loads(urllib.request.urlopen(f"{server}/object_info", timeout=30).read())
        print("registered:", sorted(NODE_IDS & set(info)))
        if NODE_IDS - set(info):
            raise SystemExit(f"nodes missing from /object_info: {sorted(NODE_IDS - set(info))}")
        options = info["ScaleRAELoader"]["input"]["required"]
        listed = (options["generator"][1]["options"] if isinstance(options["generator"][1], dict) and "options" in options["generator"][1]
                  else options["generator"][0])
        if GENERATOR not in listed:
            failures.append(f"generator list {listed}")

        template = json.loads((ROOT / "workflows" / "api" / "t2i_basic.json").read_text(encoding="utf-8"))

        def run(tag: str, **overrides):
            wf = json.loads(json.dumps(template))
            wf["1"]["inputs"].update({"generator": GENERATOR, "decoder": DECODER, "device": "cpu", **overrides.pop("loader", {})})
            wf["2"]["inputs"].update({"steps": 4, "image_start": "force", **overrides})
            wf["3"]["inputs"]["filename_prefix"] = f"ci/{tag}"
            result = queue_client.run(server, wf, None, 600)
            line = {"workflow": tag, "status": result["status"], "seconds": result["wall_seconds"], "error": result.get("error")}
            if result["status"] == "success":
                line["images"] = [pixels_sha(output / "ci" / f) for f in result["files"]]
            print(json.dumps(line))
            if result["status"] != "success":
                failures.append(f"{tag}: {result.get('error') or result['status']}")
                return None
            return line

        # seed 11, seed 12, seed 11 again: ComfyUI caches the outputs of an unchanged node only from the previous prompt, so the third
        # run executes the sampler again
        first = run("t2i_a", seed=11)
        other = run("t2i_c", seed=12)
        second = run("t2i_b", seed=11)
        if first and second and other:
            if first["images"][0][1] != (224, 224, 3):
                failures.append(f"image shape {first['images'][0][1]}")
            if first["images"][0][0] != second["images"][0][0]:
                failures.append("same seed gave different images")
            if first["images"][0][0] == other["images"][0][0]:
                failures.append("different seeds gave the same image")
        batch = run("t2i_batch", seed=11, batch_size=2, loader={"keep_loaded": False})
        if batch and first and (len(batch["images"]) != 2 or batch["images"][0][0] != first["images"][0][0]):
            failures.append(f"batch images {batch['images']}")
        unload = queue_client.run(server, {"1": {"class_type": "ScaleRAEUnload", "inputs": {}}}, None, 120)
        print(json.dumps({"workflow": "unload", "status": unload["status"], "text": unload.get("text")}))
        if unload["status"] != "success":
            failures.append(f"unload: {unload.get('error')}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        print((work / "comfyui.log").read_text(encoding="utf-8", errors="replace")[-4000:])
        return 1
    print("queue check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
