"""The vendored upstream code must be exactly the pinned upstream files (copied, or with the documented edits of scripts/vendor_sync.py),
and the installed package must not contain code that reads environment variables or opens network connections."""

import hashlib
import re
from pathlib import Path

import vendor_sync

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "scale_rae_comfy" / "vendor"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_matches_every_vendored_file():
    listed = dict(reversed(line.split("  ", 1)) for line in (VENDOR / "MANIFEST.sha256").read_text(encoding="ascii").splitlines())
    actual = {p.relative_to(VENDOR).as_posix(): _sha(p) for p in (VENDOR / "scale_rae").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    actual["LICENSE"] = _sha(VENDOR / "LICENSE")
    assert actual == listed


def test_every_vendored_file_is_accounted_for():
    files = {p.relative_to(VENDOR).as_posix() for p in (VENDOR / "scale_rae").rglob("*.py") if "__pycache__" not in p.parts}
    assert files == set(vendor_sync.COPIED) | set(vendor_sync.PATCHED) | set(vendor_sync.SHIMS)


def test_vendor_notes_pin_the_commit():
    notes = (VENDOR / "UPSTREAM.md").read_text(encoding="utf-8")
    assert vendor_sync.UPSTREAM_COMMIT in notes
    for rel in vendor_sync.PATCHED + vendor_sync.SHIMS:
        assert rel in notes, rel


def test_identical_to_the_upstream_checkout(upstream_dir):
    """CI checks out the pinned commit (LF line endings on every platform) and rebuilds the tree with the documented edits."""
    tree = vendor_sync.expected_tree(upstream_dir)
    for rel, data in tree.items():
        assert (VENDOR / rel).read_bytes() == data, rel
    assert (VENDOR / "LICENSE").read_bytes() == (upstream_dir / "LICENSE").read_bytes()


def test_patches_touch_only_comments_and_one_import(upstream_dir):
    import ast

    for rel in vendor_sync.PATCHED:
        original = (upstream_dir / rel).read_text(encoding="utf-8")
        patched = (VENDOR / rel).read_text(encoding="utf-8")
        a, b = ast.parse(original), ast.parse(patched)
        if rel.endswith("lightningDiT.py"):
            assert ast.dump(a) == ast.dump(b)  # comment lines only
        else:
            def strip(tree):  # docstrings and the prune import aside, the code is unchanged
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and ast.get_docstring(node) is not None:
                        node.body = node.body[1:]
                tree.body = [n for n in tree.body if not ((isinstance(n, ast.ImportFrom) and n.module == "transformers.pytorch_utils")
                                                          or (isinstance(n, ast.Try) and "pruneable" in ast.dump(n)))]
                return ast.dump(tree)

            assert strip(a) == strip(b)


PACKAGE_FILES = [p for p in [ROOT / "__init__.py", *(ROOT / "scale_rae_comfy").rglob("*.py"), *(ROOT / "tools").rglob("*.py")]
                 if "__pycache__" not in p.parts]
FORBIDDEN = re.compile(r"os\.environ|getenv|putenv|urllib|requests\.|http\.client|socket\.|subprocess|os\.system|importlib|(?<![.\w])exec\(|(?<![.\w])eval\(|__import__")


def test_package_has_no_environment_network_or_process_code():
    """The same categories as the Comfy Registry's automated scan (scripts/ and tests/ are not part of the Registry package)."""
    hits = []
    for path in PACKAGE_FILES:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN.search(line):
                hits.append(f"{path.relative_to(ROOT).as_posix()}:{number}: {line.strip()}")
    assert not hits, "\n".join(hits)


def test_comfyignore_keeps_development_files_out_of_the_registry_package():
    ignored = {line.strip() for line in (ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
    assert {"tests/", "scripts/", ".github/"} <= ignored
