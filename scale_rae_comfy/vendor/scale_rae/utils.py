"""ComfyUI-ScaleRAE shim (not an upstream file).

The vendored modules import only `IS_XLA_AVAILABLE` from upstream `scale_rae/utils.py`, which also holds logging, web-demo and video helpers
that this pack does not use. TorchXLA (TPU training) is never used by the pack.
"""

IS_XLA_AVAILABLE = False
