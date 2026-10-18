"""Run text-to-image jobs through a RUNNING ComfyUI queue (POST /prompt) with the shipped workflows/api/t2i_basic.json.

    python scripts/queue_jobs.py --server http://127.0.0.1:8188 --comfy-output <ComfyUI/output> --jobs jobs.json --out <dir> [--reverse]

Each job overrides prompt, seed, guidance (and cfg_mode / steps / image_start if given; cfg_mode defaults to "official-cli" and image_start to "model" so the
results compare with the official code). The saved PNG is copied to <out>/<id>.png and its pixel hash written to results.jsonl in the
same form as scripts/run_jobs.py, so scripts/compare_runs.py can compare queue, in-process and official runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--comfy-output", type=Path, required=True)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reverse", action="store_true", help="run the jobs in reverse order")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    if args.reverse:
        jobs = jobs[::-1]
    template = json.loads((ROOT / "workflows" / "api" / "t2i_basic.json").read_text(encoding="utf-8"))
    rows, failed = [], 0
    for job in jobs:
        wf = json.loads(json.dumps(template))
        wf["1"]["inputs"]["dtype"] = args.dtype
        wf["2"]["inputs"].update({"prompt": job["prompt"], "seed": job["seed"], "guidance": job["guidance"], "cfg_mode": job.get("cfg_mode", "official-cli"),
                                  "steps": job.get("steps", 50), "image_start": job.get("image_start", "model")})
        wf["3"]["inputs"]["filename_prefix"] = f"scale_rae_queue/{job['id']}"
        result = queue_client.run(args.server, wf, None, 900)
        row = {**job, "status": result["status"], "error": result.get("error")}
        if result["status"] == "success":
            src = args.comfy_output / "scale_rae_queue" / result["files"][0]
            shutil.copyfile(src, args.out / f"{job['id']}.png")
            arr = np.asarray(Image.open(args.out / f"{job['id']}.png").convert("RGB"))
            report = json.loads(next(v[0] for v in result["text"].values() if v and "{" in v[0]))
            row.update({"image_shape": list(arr.shape), "image_sha": hashlib.sha256(arr.tobytes()).hexdigest()[:16],
                        "generation_seconds": report["generation_seconds"], "peak_vram_allocated_gib": report.get("peak_vram_allocated_gib"),
                        "wall_seconds": result["wall_seconds"]})
        else:
            failed += 1
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("id", "status", "image_sha", "generation_seconds", "wall_seconds", "error")}), flush=True)
    with open(args.out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
