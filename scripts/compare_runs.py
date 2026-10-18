"""Compare two run folders (results.jsonl + <id>.png + <id>_latent.pt), e.g. the official code and this pack, or two environments.

    python scripts/compare_runs.py <dir A> <dir B>

Per job: identical latent / image hashes, max |difference| of the latents, PSNR and max pixel difference of the 8-bit images.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def rows(folder: Path) -> dict:
    out = {}
    for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "id" in r:
            out[r["id"]] = r
    return out


def main() -> int:
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = rows(a_dir), rows(b_dir)
    summary = {"jobs": 0, "identical_latent": 0, "identical_image": 0, "psnr": []}
    for key in sorted(set(a) & set(b)):
        ia = np.asarray(Image.open(a_dir / f"{key}.png").convert("RGB")).astype(np.float64)
        ib = np.asarray(Image.open(b_dir / f"{key}.png").convert("RGB")).astype(np.float64)
        mse = float(((ia - ib) ** 2).mean())
        psnr = math.inf if mse == 0 else 10 * math.log10(255**2 / mse)
        summary["jobs"] += 1
        summary["identical_image"] += a[key]["image_sha"] == b[key]["image_sha"]
        summary["psnr"].append(psnr)
        row = {"id": key, "image_identical": a[key]["image_sha"] == b[key]["image_sha"], "pixel_max_abs": int(np.abs(ia - ib).max()),
               "pixel_mean_abs": round(float(np.abs(ia - ib).mean()), 3), "psnr_db": round(psnr, 2) if psnr != math.inf else "inf"}
        if (a_dir / f"{key}_latent.pt").exists() and (b_dir / f"{key}_latent.pt").exists():  # queue runs have no latent file
            la = torch.load(a_dir / f"{key}_latent.pt").float()
            lb = torch.load(b_dir / f"{key}_latent.pt").float()
            summary["identical_latent"] += a[key]["latent_sha"] == b[key]["latent_sha"]
            row.update({"latent_identical": a[key]["latent_sha"] == b[key]["latent_sha"], "latent_max_abs": round(float((la - lb).abs().max()), 6),
                        "latent_rel_l2": round(float((la - lb).norm() / la.norm()), 6)})
        print(json.dumps(row))
    finite = [p for p in summary["psnr"] if p != math.inf]
    summary["psnr_min"] = round(min(finite), 2) if finite else "inf"
    del summary["psnr"]
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
