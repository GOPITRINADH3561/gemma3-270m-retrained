"""
Gemma 3 270M - Complete Implementation from Scratch
====================================================
Pre-trained on TinyStories dataset.

Architecture: Custom Gemma 3 (164.6M parameters)
Features: MQA, Sliding Window, RoPE (dual base), QK-Norm, GeGLU
HuggingFace: https://huggingface.co/G3nadh/gemma3-270m-tinystories
Credits: Vizuara Team - Raj (https://youtu.be/bLDlwcl6hbA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import math
import tiktoken


# ======= Configuration =======

@dataclass
class Gemma3Config:
    vocab_size: int = 50_257
    context_length: int = 32_768
    emb_dim: int = 640
    n_layers: int = 18
    n_heads: int = 4
    head_dim: int = 256
    hidden_dim: int = 2048
    n_kv_groups: int = 1
    qk_norm: bool = True
    query_pre_attn_scalar: int = 256
    rope_base: float = 1_000_000.0
    rope_local_base: float = 10_000.0
    sliding_window: int = 512
    dtype: str = "bfloat16"
    layer_types: List[str] = field(default_factory=lambda: [
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
    ])


# ======= RMSNorm =======

class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x.float() / rms
        return (x_normed * (1.0 + self.weight.float())).to(x.dtype)


# ======= RoPE =======

def precompute_rope_frequencies(head_dim, max_seq_len, rope_base=10_000.0, device=None):
    assert head_dim % 2 == 0
    i = torch.arange(0, head_dim, 2, device=device).float()
    freqs = 1.0 / (rope_base ** (i / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x, freqs_cis):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_cis
    return torch.view_as_real(x_rotated).reshape(*x.shape).to(x.dtype)


# ======= Multi-Query Attention =======

class MultiQueryAttention(nn.Module):
    def __init__(self, config: Gemma3Config, layer_type: str):
        super().__init__()
        self.config = config
        self.layer_type = layer_type
        self.n_heads = config.n_heads
        self.n_kv = config.n_kv_groups
        self.head_dim = config.head_dim
        self.scaling = config.query_pre_attn_scalar ** -0.5

        self.q_proj = nn.Linear(config.emb_dim, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.emb_dim, config.n_kv_groups * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.emb_dim, config.n_kv_groups * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.emb_dim, bias=False)

        if config.qk_norm:
            self.q_norm = RMSNorm(config.head_dim)
            self.k_norm = RMSNorm(config.head_dim)

        self.sliding_window = config.sliding_window if layer_type == "sliding_attention" else None

    def forward(self, x, freqs_cis):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv, self.head_dim)

        if self.config.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        q, k = apply_rope(q, freqs_cis), apply_rope(k, freqs_cis)

        if self.n_kv < self.n_heads:
            k = k.expand(B, S, self.n_heads, self.head_dim)
            v = v.expand(B, S, self.n_heads, self.head_dim)

        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn = self._apply_mask(attn, S)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

    def _apply_mask(self, attn, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=attn.device, dtype=torch.bool), diagonal=1)
        if self.sliding_window is not None:
            mask = mask | torch.tril(torch.ones(seq_len, seq_len, device=attn.device, dtype=torch.bool),
                diagonal=-(self.sliding_window + 1))
        return attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))


# ======= Feed Forward (GeGLU) =======

class FeedForward(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.emb_dim, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.emb_dim, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.emb_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


# ======= Transformer Block =======

class TransformerBlock(nn.Module):
    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.attn_norm = RMSNorm(config.emb_dim)
        self.attention = MultiQueryAttention(config, self.layer_type)
        self.ffn_norm = RMSNorm(config.emb_dim)
        self.ffn = FeedForward(config)

    def forward(self, x, freqs_cis):
        x = x + self.attention(self.attn_norm(x), freqs_cis)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ======= Complete Model =======

class Gemma3Model(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.emb_dim)
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layers)])
        self.final_norm = RMSNorm(config.emb_dim)
        self.output_proj = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        self.register_buffer("freqs_local", precompute_rope_frequencies(
            config.head_dim, config.context_length, config.rope_local_base), persistent=False)
        self.register_buffer("freqs_global", precompute_rope_frequencies(
            config.head_dim, config.context_length, config.rope_base), persistent=False)

        self.emb_scale = config.emb_dim ** 0.5
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids) * self.emb_scale
        for layer in self.layers:
            freqs = self.freqs_local[:S] if layer.layer_type == "sliding_attention" else self.freqs_global[:S]
            x = layer(x, freqs)
        logits = self.output_proj(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=200, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx = input_ids if input_ids.size(1) <= self.config.context_length else input_ids[:, -self.config.context_length:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            input_ids = torch.cat([input_ids, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return input_ids


# ======= Inference Helpers =======

def load_model(checkpoint_path: str, device: str = "cuda"):
    config = Gemma3Config()
    model = Gemma3Model(config)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def generate_text(model, prompt: str, max_tokens=200, temperature=0.7, top_k=50, device="cuda"):
    enc = tiktoken.get_encoding("gpt2")
    input_ids = torch.tensor([enc.encode_ordinary(prompt)], device=device)
    with torch.no_grad(), autocast(device_type=device if device != "cpu" else "cpu", dtype=torch.bfloat16):
        output = model.generate(input_ids, max_tokens, temperature, top_k)
    return enc.decode(output[0].tolist())


if __name__ == "__main__":
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "pytorch_model.bin"

    print(f"Loading model from {ckpt}...")
    model = load_model(ckpt, device)
    print(f"Model loaded! ({sum(p.numel() for p in model.parameters()):,} parameters)")

    prompts = [
        "Once upon a time, there was a little girl named Lily",
        "The big brown dog ran to the park",
        "One day, a magical bird flew into the garden",
    ]

    for prompt in prompts:
        print(f"\nPrompt: \"{prompt}\"")
        print(f"Story:  {generate_text(model, prompt, device=device)}")
        print("-" * 60)
