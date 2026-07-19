"""
Spatially-Aware Transcriptomic Encoder (E_t).

Implements a GATv2 message-passing layer directly (no torch_geometric
dependency, so this runs anywhere torch runs) following:

    e(v,u)      = a^T LeakyReLU(W1 x_v + W2 x_u)
    beta_{v,u}  = softmax_u( e(v,u) )
    x_v'        = sigma( sum_h sum_u beta_{v,u}^h * W2^h x_u )

Graphs are built by radius connectivity (default 25um, matching the
manuscript's stated cellular-adjacency radius) over 2D spatial coordinates.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_radius_graph(coords: torch.Tensor, radius: float) -> torch.Tensor:
    """Build an undirected radius graph.

    Args:
        coords: (N, 2) spatial coordinates (same units as `radius`).
        radius: connectivity radius.

    Returns:
        edge_index: (2, E) long tensor of directed edges (both directions
        included, self-loops excluded).
    """
    with torch.no_grad():
        dist = torch.cdist(coords, coords)
        mask = (dist <= radius) & (dist > 0)
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    return edge_index


class GATv2Layer(nn.Module):
    """Single multi-head GATv2 layer, message-passing implemented via
    scatter-softmax over an explicit edge list (dense-safe for the graph
    sizes used in per-neighborhood ST windows; not intended for
    whole-slide-scale graphs without a sparse backend)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.heads = heads
        self.head_dim = out_dim // heads
        self.out_dim = out_dim

        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.att)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_dim)
        edge_index: (2, E), row 0 = source u, row 1 = target v  (message u -> v)
        returns: (N, out_dim)
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        h1 = self.W1(x).view(N, self.heads, self.head_dim)  # W1 x_v, indexed by dst
        h2 = self.W2(x).view(N, self.heads, self.head_dim)  # W2 x_u, indexed by src

        # e(v,u) = a^T LeakyReLU(W1 x_v + W2 x_u), per head
        e = self.leaky_relu(h1[dst] + h2[src])          # (E, heads, head_dim)
        e = (e * self.att.unsqueeze(0)).sum(-1)          # (E, heads)

        # softmax over incoming edges per (dst, head) -- numerically stable
        e = e - e.max(dim=0, keepdim=True).values
        e_exp = e.exp()
        denom = torch.zeros(N, self.heads, device=x.device).index_add_(0, dst, e_exp) + 1e-16
        beta = e_exp / denom[dst]                        # (E, heads)
        beta = self.dropout(beta)

        msg = beta.unsqueeze(-1) * h2[src]                # (E, heads, head_dim)
        out = torch.zeros(N, self.heads, self.head_dim, device=x.device)
        out = out.index_add_(0, dst, msg)
        return out.reshape(N, self.out_dim)


class SpatialTranscriptomicEncoder(nn.Module):
    """4-layer, 8-head GATv2 stack, hidden dim 512, matching Sec. 5.3 of the
    manuscript. Input: 480-gene normalized transcript vectors per node."""

    def __init__(
        self,
        gene_dim: int = 480,
        hidden_dim: int = 512,
        n_layers: int = 4,
        heads: int = 8,
        radius: float = 25.0,
    ):
        super().__init__()
        self.radius = radius
        self.input_proj = nn.Linear(gene_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GATv2Layer(hidden_dim, hidden_dim, heads=heads) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        x: (N, gene_dim) transcript vectors
        coords: (N, 2) spatial coordinates in the same units as self.radius
        returns: (N, hidden_dim) node embeddings after the final GNN layer
        """
        edge_index = build_radius_graph(coords, self.radius)
        h = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + F.gelu(layer(h, edge_index)))  # residual + norm
        return h

    def pool_neighborhood(self, node_embeds: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
        """Masked mean-pool of node embeddings into a single per-neighborhood
        vector z_t,i, as used to pair against z_m,i in the contrastive loss.

        window_mask: (N,) boolean mask selecting nodes in neighborhood i.
        """
        if window_mask.sum() == 0:
            raise ValueError("Empty neighborhood window.")
        return node_embeds[window_mask].mean(dim=0)
