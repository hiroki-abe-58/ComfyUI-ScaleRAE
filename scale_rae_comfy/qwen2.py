"""The Qwen2 decoder stack of Scale-RAE, in plain PyTorch.

Scale-RAE's language model is Qwen2.5-1.5B run through transformers 4.37's `Qwen2Model` with SDPA attention and no KV cache (the official
`greedy_decode` recomputes the whole sequence every step). transformers 5 changed the internals that the official subclass relies on,
so this module re-states that forward pass with the same operations, in the same order:

- RMSNorm in float32, then cast back (`Qwen2RMSNorm`);
- rotary embedding from a cos/sin table that 4.37 builds once on the CPU in float32 for `max_position_embeddings` positions and moves
  with the module (`Qwen2RotaryEmbedding._set_cos_sin_cache`), indexed by `position_ids` (`apply_rotary_pos_emb`);
- grouped-query attention by repeating the key/value heads (`repeat_kv`) and `scaled_dot_product_attention(..., is_causal=True)` with
  no mask (4.37 drops an all-ones mask for full-length prompts, `_prepare_4d_causal_attention_mask_for_sdpa`);
- SwiGLU MLP (`down(silu(gate(x)) * up(x))`).
Weights are the `model.*` tensors of the official checkpoint; `lm_head` is tied to the embeddings (`tie_word_embeddings`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Rotary(nn.Module):
    def __init__(self, dim: int, max_positions: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_positions, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # built on the CPU in float32 like transformers 4.37, then moved with the module
        self.register_buffer("cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    def forward(self, dtype: torch.dtype, seq_len: int):
        return self.cos_cached[:seq_len].to(dtype=dtype), self.sin_cached[:seq_len].to(dtype=dtype)


class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.hidden = cfg["hidden_size"]
        self.heads = cfg["num_attention_heads"]
        self.kv_heads = cfg["num_key_value_heads"]
        self.head_dim = self.hidden // self.heads
        self.q_proj = nn.Linear(self.hidden, self.heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden, self.kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden, self.kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.heads * self.head_dim, self.hidden, bias=False)
        self.rotary = Rotary(self.head_dim, cfg["max_position_embeddings"], cfg["rope_theta"])

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = x.size()
        q = self.q_proj(x).view(bsz, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(v.dtype, q_len)
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        groups = self.heads // self.kv_heads
        if groups > 1:
            k = k[:, :, None, :, :].expand(bsz, self.kv_heads, groups, q_len, self.head_dim).reshape(bsz, self.heads, q_len, self.head_dim)
            v = v[:, :, None, :, :].expand(bsz, self.kv_heads, groups, q_len, self.head_dim).reshape(bsz, self.heads, q_len, self.head_dim)
        # (4.37 makes q/k/v contiguous only when an attention mask is passed, which never happens here)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=q_len > 1)
        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.gate_proj = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), position_ids)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2Stack(nn.Module):
    """`model.embed_tokens`, `model.layers.*`, `model.norm` and `model.latent_queries` of the checkpoint; `logits` uses the tied head."""

    def __init__(self, cfg: dict, num_latent_queries: int):
        super().__init__()
        if cfg.get("hidden_act", "silu") != "silu":
            raise ValueError(f"unsupported hidden_act {cfg.get('hidden_act')!r}")
        if cfg.get("rope_scaling"):
            raise ValueError("rope_scaling is not supported")
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg["num_hidden_layers"]))
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.latent_queries = nn.Parameter(torch.zeros(num_latent_queries, cfg["hidden_size"]))

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """inputs_embeds [B, T, H] -> final-norm hidden states [B, T, H] (positions 0..T-1, causal)."""
        seq = inputs_embeds.shape[1]
        position_ids = torch.arange(0, seq, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).view(-1, seq)
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, position_ids)
        return self.norm(h)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.embed_tokens.weight).float()
