"""ComfyUI-ScaleRAE shim for upstream `models/AdaLNmlp.py`, the MLP diffusion head (`use_mlp=True`, single-token heads).

The released text-to-image checkpoints use the LightningDiT head, and `diffloss.py` only needs this name at import time, so the pack
does not ship the MLP head (its upstream file also carries TPU launcher code).
"""

import torch.nn as nn


class SimpleMLPAdaLN(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("the MLP diffusion head (use_mlp=True) is not supported by ComfyUI-ScaleRAE")
