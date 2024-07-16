# %%
import torch.nn as nn
import torch

from typing import Type
import math

class Transformer(nn.Module):
    def __init__(
        self,
        depth: int = 2,
        embed_dim: int = 768,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionBlock(
                embed_dim,
                num_heads,
                mlp_dim,
                activation,
                attention_downsample_rate
            ) for _ in range(depth)
        ])

        self.final_attn = Attention(embed_dim, num_heads, attention_downsample_rate)
        self.norm_final = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        embed_current: torch.Tensor,
        embed_initial: torch.Tensor,
        embed_current_pe: torch.Tensor,
    ):
        queries = embed_current

        for layer in self.layers:
            queries = layer(queries, embed_initial, embed_current_pe)

        queries += embed_current
        queries += self.final_attn(queries, embed_initial, queries)
        queries = self.norm_final(queries)
        queries = self.output_layer(queries)

        return queries

class AttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 1
    ):
        super().__init__()
        self.self_attn = Attention(embed_dim, num_heads, attention_downsample_rate)
        self.norm_one = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_dim, activation)
        self.norm_two = nn.LayerNorm(embed_dim)
        self.cross_attn = Attention(embed_dim, num_heads, attention_downsample_rate)
        self.norm_three = nn.LayerNorm(embed_dim)

    def forward(
        self,
        current: torch.Tensor,
        encoded: torch.Tensor,
        current_pe: torch.Tensor,
    ) -> torch.Tensor:
        current += current_pe
        current = self.norm_one(current + self.self_attn(current, current, current))
        current = self.norm_two(current + self.mlp(current))
        current += current_pe
        current = self.norm_three(current + self.cross_attn(current, encoded, current))
        return current

class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        downsample_rate: int = 1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.internal_dim = embed_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embed_dim"

        self.q_proj = nn.Linear(embed_dim, self.internal_dim)
        self.k_proj = nn.Linear(embed_dim, self.internal_dim)
        self.v_proj = nn.Linear(embed_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embed_dim)

    def split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def fuse_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self.split_heads(q, self.num_heads)
        k = self.split_heads(k, self.num_heads)
        v = self.split_heads(v, self.num_heads)

        _, _, _, c_head = q.shape
        qk_attn = q @ k.permute(0, 1, 3, 2)
        qk_attn /= math.sqrt(c_head)
        qk_attn = torch.softmax(qk_attn, dim=-1)

        out = qk_attn @ v
        out = self.fuse_heads(out)
        out = self.out_proj(out)

        return out

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

# %%
