import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# -----------------------------
# Base layers
# -----------------------------
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, act: str = "relu"):
        super().__init__()
        pad = k // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, padding=pad, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, dilation: int = 1):
        super().__init__()
        self.k = k
        self.dilation = dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=dilation, bias=False)

    def forward(self, x):
        pad = (self.k - 1) * self.dilation
        x = F.pad(x, (pad, 0))
        return self.conv(x)

class TemporalTCNBlock(nn.Module):
    def __init__(self, ch: int, k: int = 3, dilation: int = 1, act: str = "relu"):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, k=k, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)
        self.conv2 = CausalConv1d(ch, ch, k=k, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        r = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + r
        return self.act(x)

# -----------------------------
# Backbone
# -----------------------------
class TactileBackboneTCN(nn.Module):
    """
    Unified backbone network.
    - Finetuning (default): returns features `g`.
    - Pretraining: set `use_decoder=True` to build `dec` and return `g, x_hat`.
    """
    def __init__(self, D: int = 16, H: int = 9, W: int = 7, act: str = "relu", use_decoder: bool = False):
        super().__init__()
        self.H, self.W = H, W
        self.D = D
        self.use_decoder = use_decoder

        # Spatial encoder
        self.s1 = DepthwiseSeparableConv2d(3, D, k=3, act=act)
        self.s2 = DepthwiseSeparableConv2d(D, D, k=3, act=act)
        
        # Temporal encoder
        self.t1 = TemporalTCNBlock(D * H * W, k=3, dilation=1, act=act)
        self.t2 = TemporalTCNBlock(D * H * W, k=3, dilation=2, act=act)
        self.t3 = TemporalTCNBlock(D * H * W, k=3, dilation=4, act=act)

        # Decoder (only for Pretrain masked reconstruction task)
        if self.use_decoder:
            self.dec = nn.Conv2d(D, 3, kernel_size=1, bias=True)

    def forward(self, x):
        B, L, C, H, W = x.shape
        assert (H, W) == (self.H, self.W)
        
        x_ = x.view(B * L, C, H, W)
        f = self.s1(x_)
        f = self.s2(f)
        f = f.view(B, L, self.D, H, W)
        
        g = f.permute(0, 2, 3, 4, 1).contiguous()
        g = g.view(B, self.D * H * W, L)
        g = self.t1(g)
        g = self.t2(g)
        g = self.t3(g)
        g = g.view(B, self.D, H, W, L).permute(0, 4, 1, 2, 3).contiguous()
        
        # If in Pretrain mode, reconstruct and return both
        if self.use_decoder:
            z_ = g.view(B * L, self.D, H, W)
            x_hat = self.dec(z_).view(B, L, 3, H, W)
            return g, x_hat
            
        # If in Finetune mode, return features only
        return g

# -----------------------------
# Classification Heads
# -----------------------------
class TemporalAttentionPooling(nn.Module):
    def __init__(self, D: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(D, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        attn_logits = self.score(x).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)
        return pooled, attn

class ComplianceHead(nn.Module):
    def __init__(self, D: int, hidden: int = 64):
        super().__init__()
        self.temporal_pool = TemporalAttentionPooling(D, hidden=hidden)
        self.fc = nn.Linear(D, 2)

    def forward(self, z: torch.Tensor):
        zt = z.mean(dim=[3, 4])
        pooled, _ = self.temporal_pool(zt)
        return self.fc(pooled)

class SlipOrdinalHead(nn.Module):
    def __init__(self, D: int, hidden: Optional[int] = None, tail_k: int = 5, dropout: float = 0.65):
        super().__init__()
        hidden = D if hidden is None else hidden
        self.tail_k = tail_k

        self.temporal = nn.Sequential(
            CausalConv1d(D, hidden, k=3, dilation=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(hidden, hidden, k=3, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(hidden, 2)

    def forward(self, z: torch.Tensor):
        zt = z.mean(dim=[3, 4])
        h = zt.transpose(1, 2).contiguous()
        h = self.temporal(h)
        k = min(self.tail_k, h.shape[-1])
        h_tail = h[:, :, -k:]
        h_out = h_tail.mean(dim=2)

        return self.fc(h_out)