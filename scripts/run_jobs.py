"""Run text-to-image jobs through this pack's runtime (the code behind the nodes), in-process.

    python scripts/run_jobs.py --model <Scale-RAE folder> --decoder <decoder folder> --jobs jobs.json --out <dir> [--dtype float32] [--comfyui <ComfyUI>]

jobs.json: [{"id", "prompt", "seed", "guidance"[, "steps", "cfg_mode", "image_start"]}] (defaults: cfg_mode "official-cli", image_start "model").
Writes <out>/<id>.png, <id>_latent.pt and results.jsonl. The hashes are computed like the official CLI saves its outputs (latent: float32
bytes of the [256, C] tensor; image: `(clip(x, 0, 1) * 255).astype(uint8)`), so they can be compared with a run of the official code (see docs/VERIFICATION.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--decoder", type=Path, required=True)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--comfyui", type=Path, help="ComfyUI checkout, only to make `comfy` importable (optional)")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    if args.comfyui:
        sys.path.insert(1, str(args.comfyui))
    import torch
    from PIL import Image

    from scale_rae_comfy import runtime

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    t0 = time.time()
    loaded = runtime.acquire(runtime.make_handle(str(args.model), str(args.decoder), args.dtype, args.device, True))
    info = {**loaded.info, "torch": torch.__version__, "wall_load_s": round(time.time() - t0, 1)}
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    print(json.dumps(info), flush=True)
    rows = []
    for job in jobs:
        # cfg_mode defaults to the released behaviour so that runs compare with the official code (see runtime.generate)
        out = runtime.generate(loaded, job["prompt"], job["seed"], job["guidance"], job.get("steps", 50),
                               cfg_mode=job.get("cfg_mode", "official-cli"), image_start=job.get("image_start", "model"))
        latent = out["latent"].numpy().astype(np.float32)
        arr = (out["image"][0].numpy() * 255).astype("uint8")
        Image.fromarray(arr).save(args.out / f"{job['id']}.png")
        torch.save(out["latent"], args.out / f"{job['id']}_latent.pt")
        row = {**job, "latent_shape": list(latent.shape), "latent_sha": hashlib.sha256(latent.tobytes()).hexdigest()[:16],
               "image_shape": list(arr.shape), "image_sha": hashlib.sha256(arr.tobytes()).hexdigest()[:16], **out["report"]}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("id", "latent_sha", "image_sha", "generation_seconds", "text_tokens_before_image")}), flush=True)
    with open(args.out / "results.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"info": info}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
