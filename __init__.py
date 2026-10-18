"""Scale-RAE for ComfyUI (unofficial integration of https://github.com/ZitengWangNYU/Scale-RAE)."""

import logging

try:
    from .scale_rae_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API, or a missing dependency
    logging.error("[Scale-RAE] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
