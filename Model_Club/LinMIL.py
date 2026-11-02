import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """Linear self-attention with positive feature map.

    Input:  [B, N, C]
    Output: [B, N, C]
    """

    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def feature_map(x):
        return F.elu(x) + 1.0

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.feature_map(q)
        k = self.feature_map(k)

        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        k_sum = k.sum(dim=2)
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k_sum) + 1e-6)
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return self.dropout(out)


class LinMILBlock(nn.Module):
    def __init__(self, dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LinMIL(nn.Module):
    """Lin-MIL style WSI classifier for this project's bag-feature interface.

    The model uses linear-complexity self-attention over bag instances and returns
    the same keys as the existing Teacher models: probs/logits and score.
    """

    def __init__(
        self,
        in_dim=1024,
        num_classes=2,
        embed_dim=512,
        depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            LinMILBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.score_head = nn.Linear(embed_dim, 1)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        x = data["x"]
        if x.dim() == 3:
            x = x.squeeze(0)
        h = self.embed(x).unsqueeze(0)

        cls_token = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat([cls_token, h], dim=1)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)

        cls_feat = h[:, 0]
        patch_feat = h[:, 1:]
        logits = self.classifier(cls_feat)

        score = self.score_head(patch_feat).squeeze(0).transpose(0, 1)
        score = F.softmax(score, dim=-1).squeeze(0)

        probs = logits
        if not self.training:
            probs = F.softmax(logits, dim=-1)

        z = cls_feat.squeeze(0)
        return {"probs": probs, "score": score, "z": z}
