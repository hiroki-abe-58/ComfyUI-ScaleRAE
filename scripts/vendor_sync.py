"""Rebuild scale_rae_comfy/vendor/scale_rae from an upstream checkout and rewrite vendor/MANIFEST.sha256 (maintainer tool).

    python scripts/vendor_sync.py --upstream <ZitengWangNYU/Scale-RAE checkout at UPSTREAM_COMMIT> [--check]

COPIED files are taken byte for byte. PATCHED files are the upstream file with the edits of `patch()` below: they change no code that
text-to-image runs (an import that transformers 5 removed, and comment / docstring-example lines). SHIMS are this pack's own small
files that replace upstream modules the pack does not use (their upstream versions import training, XLA or launcher code).
tests/test_vendor.py applies the same functions to the upstream checkout in CI and compares every byte.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "scale_rae_comfy" / "vendor"
UPSTREAM_COMMIT = "dab38002b4b42c2d2262776785276d410599a443"

DIFF = "scale_rae/model/diffusion_loss"
COPIED = [
    f"{DIFF}/__init__.py",
    f"{DIFF}/diffloss.py",
    *(f"{DIFF}/diffusion/{n}.py" for n in ("__init__", "diffusion_utils", "gaussian_diffusion", "respace", "rf", "simple_diffusion",
                                         "timestep_sampler", "transport")),
    *(f"{DIFF}/models/{n}.py" for n in ("__init__", "attentions", "model_utils")),
]
PATCHED = [f"{DIFF}/models/lightningDiT.py", "scale_rae/model/multimodal_decoder/decoder.py"]
SHIMS = ["scale_rae/__init__.py", "scale_rae/utils.py", "scale_rae/model/__init__.py", f"{DIFF}/models/AdaLNmlp.py"]

_PRUNE_OLD = "from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer\n"
_PRUNE_NEW = (
    "from transformers.pytorch_utils import prune_linear_layer\n"
    "try:  # ComfyUI-ScaleRAE patch: removed in transformers 5; only prune_heads() uses it, inference never does\n"
    "    from transformers.pytorch_utils import find_pruneable_heads_and_indices\n"
    "except ImportError:\n"
    "    find_pruneable_heads_and_indices = None\n"
)
# docstring example of transformers' ViTMAEModel.forward that downloads a COCO image; documentation only
_DOC_EXAMPLE_LINES = (
    "        >>> import requests\n",
    '        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n',
    "        >>> image = Image.open(requests.get(url, stream=True).raw)\n",
)


NOTICE = "# This file was changed by ComfyUI-ScaleRAE; scale_rae_comfy/vendor/UPSTREAM.md lists the changes.\n"


def patch(rel: str, text: str) -> str:
    """The pack's edits of one PATCHED upstream file (str -> str, LF line endings). Each patched file starts with NOTICE (after a coding
    declaration, which must stay on the first line)."""
    lines = text.splitlines(keepends=True)
    if rel.endswith("lightningDiT.py"):
        # commented-out TPU sharding code reads a launcher environment variable; drop those comment lines
        kept = [ln for ln in lines if not (ln.lstrip().startswith("#") and "os.getenv(" in ln)]
        if len(lines) - len(kept) != 9:
            raise ValueError(f"{rel}: expected 9 commented os.getenv lines, found {len(lines) - len(kept)}")
        return NOTICE + "".join(kept)
    if rel.endswith("decoder.py"):
        if text.count(_PRUNE_OLD) != 2:
            raise ValueError(f"{rel}: expected the prune import twice")
        text = text.replace(_PRUNE_OLD, _PRUNE_NEW)
        lines = text.splitlines(keepends=True)
        kept = [ln for ln in lines if ln not in _DOC_EXAMPLE_LINES]
        if len(lines) - len(kept) != 6:
            raise ValueError(f"{rel}: expected the docstring example lines twice")
        if not kept[0].startswith("# coding"):
            raise ValueError(f"{rel}: expected a coding declaration on the first line")
        return kept[0] + NOTICE + "".join(kept[1:])
    raise KeyError(rel)


def expected_tree(upstream: Path) -> dict[str, bytes]:
    out = {rel: (upstream / rel).read_bytes() for rel in COPIED}
    for rel in PATCHED:
        out[rel] = patch(rel, (upstream / rel).read_bytes().decode("utf-8")).encode("utf-8")
    return out


def manifest_lines() -> list[str]:
    files = sorted(p for p in (VENDOR / "scale_rae").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.append(VENDOR / "LICENSE")
    return [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(VENDOR).as_posix()}" for p in files]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="only compare, write nothing")
    args = ap.parse_args(argv)
    tree = expected_tree(args.upstream)
    tree["LICENSE"] = (args.upstream / "LICENSE").read_bytes()
    bad = [rel for rel, data in tree.items() if not (VENDOR / rel).is_file() or (VENDOR / rel).read_bytes() != data]
    if args.check:
        print("\n".join(f"DIFFERS {rel}" for rel in bad) or "vendor tree matches the upstream checkout")
        return 1 if bad else 0
    for rel in bad:
        (VENDOR / rel).parent.mkdir(parents=True, exist_ok=True)
        (VENDOR / rel).write_bytes(tree[rel])
        print("wrote", rel)
    keep = set(tree) | set(SHIMS)
    for path in sorted((VENDOR / "scale_rae").rglob("*.py")):
        rel = path.relative_to(VENDOR).as_posix()
        if rel not in keep:
            path.unlink()
            print("removed", rel)
    lines = manifest_lines()
    (VENDOR / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    print(f"MANIFEST.sha256: {len(lines)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
