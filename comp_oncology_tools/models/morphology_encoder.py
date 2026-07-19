"""
Morphology Encoder (E_m).

NOTE ON HONESTY: the manuscript initializes this encoder from UNI's
pretrained ViT-H/14 weights. This sandbox has no network access to
model hubs, so `backbone` here defaults to a small randomly-initialized
ViT for testing the *fusion math* end-to-end. Swap in a real pretrained
ViT-H/14 (e.g. via timm) before using this for anything beyond
architecture verification -- random weights will not produce meaningful
embeddings.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PatchEmbedViT(nn.Module):
    """Minimal ViT patch encoder: patchify -> linear embed -> transformer
    encoder -> CLS token. Stands in for a UNI/ViT-H backbone at each
    magnification scale."""

    def __init__(self, patch_size: int = 14, in_ch: int = 3, embed_dim: int = 1024,
                 depth: int = 4, heads: int = 8, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) -> (B, embed_dim) CLS embedding."""
        B = img.size(0)
        x = self.proj(img).flatten(2).transpose(1, 2)          # (B, n_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.norm(x[:, 0])                                # CLS token


class MultiScaleFusion(nn.Module):
    """Cross-scale attention fusion of {5x, 20x, 40x} patch embeddings for a
    local (~50um) neighborhood, exactly as specified in the math supplement:

        alpha_{i,l} = softmax_l( (Wq h_i^40x)^T (Wk h_i^l) / sqrt(d) )
        z_{m,i}     = sum_l alpha_{i,l} * Wv h_i^l
    """

    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.d = embed_dim
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, h_40x: torch.Tensor, h_20x: torch.Tensor, h_5x: torch.Tensor) -> torch.Tensor:
        """All inputs: (B, embed_dim). Returns fused z_m: (B, embed_dim)."""
        q = self.Wq(h_40x)                                        # (B, d)
        scales = torch.stack([h_5x, h_20x, h_40x], dim=1)         # (B, 3, d)
        k = self.Wk(scales)                                       # (B, 3, d)
        v = self.Wv(scales)                                       # (B, 3, d)

        logits = torch.einsum("bd,bld->bl", q, k) / math.sqrt(self.d)  # (B, 3)
        alpha = torch.softmax(logits, dim=-1)                     # (B, 3)
        z_m = torch.einsum("bl,bld->bd", alpha, v)                # (B, d)
        return z_m, alpha


class MorphologyEncoder(nn.Module):
    """Full E_m: three per-scale ViT encoders (weight-shared across scales,
    matching a single ViT-H/14 backbone applied at different magnifications)
    plus the multi-scale fusion module."""

    def __init__(self, embed_dim: int = 1024, backbone: nn.Module | None = None):
        super().__init__()
        self.backbone = backbone or PatchEmbedViT(embed_dim=embed_dim)
        self.fusion = MultiScaleFusion(embed_dim=embed_dim)

    def forward(self, patch_40x: torch.Tensor, patch_20x: torch.Tensor, patch_5x: torch.Tensor):
        h_40 = self.backbone(patch_40x)
        h_20 = self.backbone(patch_20x)
        h_5 = self.backbone(patch_5x)
        z_m, alpha = self.fusion(h_40, h_20, h_5)
        return z_m, alpha
