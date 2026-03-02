# shapelet_encoder/models/LearningShapeletsSeg.py

import math
from dataclasses import dataclass
from typing import Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F


DistMeasure = Literal["euclidean", "cosine", "cross-correlation"]


def _getattr_any(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


@dataclass
class LSSegConfig:
    # core
    num_shapelets: int = 8
    shapelet_len: int = 30
    in_channels: int = 1
    dist_measure: DistMeasure = "euclidean"

    # behavior
    # z-normalize each sliding window & each shapelet (recommended for euclidean)
    znorm: bool = True
    eps: float = 1e-6

    # similarity scaling (only for euclidean/cosine)
    # euclidean: similarity = exp(-dist2 / temperature)
    # cosine: similarity = relu(cos_sim) or (cos_sim+1)/2 depending on cosine_mode
    temperature: float = 1.0

    # cosine mapping
    cosine_mode: Literal["relu", "shift01"] = "relu"

    # output alignment: pad activation map back to length T
    align: Literal["center", "left"] = "center"  # "center" recommended


class LearningShapeletsSeg(nn.Module):
    """
    A lightweight, configurable shapelet-based segmentation module.

    Input:
        x: [bs, T, C]  (ShapeX uses C=1 most of the time)

    Output:
        activations: [bs, T, P]
            - P = num_shapelets
            - values are "similarity-like": larger means stronger match
            - already padded to length T (so downstream segmentation can threshold directly)

    Notes:
    - For euclidean, we compute dist^2 over sliding windows and convert to similarity via exp(-dist2/temperature).
    - For cosine, we compute cosine similarity window-wise and optionally map to [0,1].
    - For cross-correlation, we use conv1d and relu as similarity.
    """

    def __init__(
        self,
        configs=None,
        *,
        num_shapelets: Optional[int] = None,
        shapelet_len: Optional[int] = None,
        in_channels: Optional[int] = None,
        dist_measure: Optional[DistMeasure] = None,
        znorm: Optional[bool] = None,
        temperature: Optional[float] = None,
        device: Optional[str] = None,
    ):
        super().__init__()

        # Read from configs/args if provided (matches your project style)
        cfg_num = _getattr_any(configs, ["num_shapelets", "num_prototypes"], 8)
        cfg_len = _getattr_any(configs, ["shapelet_len", "prototype_len", "prototype_len", "proto_len"], 30)
        cfg_ch = _getattr_any(configs, ["in_channels", "enc_in"], 1)
        cfg_dm = _getattr_any(configs, ["dist_measure", "shapelet_dist_measure"], "euclidean")
        cfg_zn = _getattr_any(configs, ["shapelet_znorm", "znorm"], True)
        cfg_tmp = _getattr_any(configs, ["shapelet_temperature", "temperature"], 1.0)
        cfg_align = _getattr_any(configs, ["shapelet_align", "align"], "center")
        cfg_cos_mode = _getattr_any(configs, ["cosine_mode"], "relu")

        self.cfg = LSSegConfig(
            num_shapelets=int(num_shapelets if num_shapelets is not None else cfg_num),
            shapelet_len=int(shapelet_len if shapelet_len is not None else cfg_len),
            in_channels=int(in_channels if in_channels is not None else cfg_ch),
            dist_measure=str(dist_measure if dist_measure is not None else cfg_dm),
            znorm=bool(znorm if znorm is not None else cfg_zn),
            temperature=float(temperature if temperature is not None else cfg_tmp),
            align=str(cfg_align),
            cosine_mode=str(cfg_cos_mode),
        )

        # keep a "device" attribute for compatibility with your existing code style
        self.device = device or _getattr_any(configs, ["device"], "cpu")

        # shapelets: [P, C, L]
        shp = torch.randn(self.cfg.num_shapelets, self.cfg.in_channels, self.cfg.shapelet_len)
        self.shapelets = nn.Parameter(shp)

        # init
        nn.init.kaiming_normal_(self.shapelets)

        # move module to desired device (optional; pipeline can override)
        try:
            self.to(torch.device(self.device))
        except Exception:
            # if self.device is like "cuda:0" but CUDA not available in env, ignore here
            pass

    # ---------- public helpers ----------
    @torch.no_grad()
    def get_shapelets(self) -> torch.Tensor:
        """Return shapelets as [P, L, C] for easier plotting."""
        return self.shapelets.detach().permute(0, 2, 1).contiguous()

    @torch.no_grad()
    def set_shapelets(self, new_shapelets: torch.Tensor):
        """
        new_shapelets can be [P, C, L] or [P, L, C]
        """
        if new_shapelets.dim() != 3:
            raise ValueError("new_shapelets must be 3D.")
        if new_shapelets.shape[1] == self.cfg.in_channels and new_shapelets.shape[2] == self.cfg.shapelet_len:
            shp = new_shapelets
        elif new_shapelets.shape[2] == self.cfg.in_channels and new_shapelets.shape[1] == self.cfg.shapelet_len:
            shp = new_shapelets.permute(0, 2, 1)
        else:
            raise ValueError(
                f"Shape mismatch. Expect [P,C,L]=[{self.cfg.num_shapelets},{self.cfg.in_channels},{self.cfg.shapelet_len}] "
                f"or [P,L,C]. Got {tuple(new_shapelets.shape)}"
            )
        self.shapelets.copy_(shp.to(self.shapelets.device, dtype=self.shapelets.dtype))

    def actions_sum(self, activations: torch.Tensor, mode: Literal["sum", "max"] = "sum") -> torch.Tensor:
        """
        activations: [bs, T, P]
        return: [bs, T]  used for thresholding/segmentation if you want a single curve
        """
        if mode == "sum":
            return activations.sum(dim=-1)
        elif mode == "max":
            return activations.max(dim=-1).values
        else:
            raise ValueError("mode must be 'sum' or 'max'")

    # ---------- core computations ----------
    def _znorm_windows(self, w: torch.Tensor) -> torch.Tensor:
        # w: [bs, C, T', L]
        mean = w.mean(dim=-1, keepdim=True)
        std = w.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.cfg.eps)
        return (w - mean) / std

    def _znorm_shapelets(self, s: torch.Tensor) -> torch.Tensor:
        # s: [P, C, L]
        mean = s.mean(dim=-1, keepdim=True)
        std = s.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.cfg.eps)
        return (s - mean) / std

    def _pad_to_T(self, act_map: torch.Tensor, T: int) -> torch.Tensor:
        """
        act_map: [bs, T', P]
        return:  [bs, T,  P]
        """
        bs, Tp, P = act_map.shape
        if Tp == T:
            return act_map

        if self.cfg.align == "left":
            # pad at end
            pad_right = max(0, T - Tp)
            if pad_right > 0:
                z = torch.zeros(bs, pad_right, P, device=act_map.device, dtype=act_map.dtype)
                return torch.cat([act_map, z], dim=1)
            return act_map[:, :T, :]

        # center align: pad both sides so the window position is roughly centered
        L = self.cfg.shapelet_len
        left = L // 2
        right = T - (Tp + left)
        right = max(0, right)
        zL = torch.zeros(bs, left, P, device=act_map.device, dtype=act_map.dtype) if left > 0 else None
        zR = torch.zeros(bs, right, P, device=act_map.device, dtype=act_map.dtype) if right > 0 else None

        out = act_map
        if zL is not None:
            out = torch.cat([zL, out], dim=1)
        if zR is not None:
            out = torch.cat([out, zR], dim=1)

        # if still mismatch (odd lengths), crop/pad to exact T
        if out.shape[1] < T:
            z = torch.zeros(bs, T - out.shape[1], P, device=act_map.device, dtype=act_map.dtype)
            out = torch.cat([out, z], dim=1)
        elif out.shape[1] > T:
            out = out[:, :T, :]
        return out

    def _euclidean_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [bs, C, T]
        return: act_map [bs, T', P]
        """
        bs, C, T = x.shape
        L = self.cfg.shapelet_len
        if T < L:
            # no window fits; return zeros
            return torch.zeros(bs, T, self.cfg.num_shapelets, device=x.device, dtype=x.dtype)

        # windows: [bs, C, T', L]
        w = x.unfold(dimension=2, size=L, step=1).contiguous()

        s = self.shapelets
        if self.cfg.znorm:
            w = self._znorm_windows(w)
            s = self._znorm_shapelets(s)

        # broadcast to compute dist2:
        # w: [bs, C, T', L] -> [bs, 1, C, T', L]
        # s: [P, C, L]     -> [1, P, C, 1,  L]
        w2 = w.unsqueeze(1)
        s2 = s.unsqueeze(0).unsqueeze(3)

        dist2 = (w2 - s2).pow(2).sum(dim=(2, 4))  # [bs, P, T']
        dist2 = dist2.permute(0, 2, 1).contiguous()  # [bs, T', P]

        # similarity-like: larger = better match
        # exp(-dist2 / temperature)
        temp = max(self.cfg.temperature, self.cfg.eps)
        act = torch.exp(-dist2 / temp)
        return act

    def _cosine_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [bs, C, T]
        return: act_map [bs, T', P]
        """
        bs, C, T = x.shape
        L = self.cfg.shapelet_len
        if T < L:
            return torch.zeros(bs, T, self.cfg.num_shapelets, device=x.device, dtype=x.dtype)

        w = x.unfold(2, L, 1).contiguous()  # [bs, C, T', L]

        # normalize windows and shapelets by L2
        w_norm = w / (w.norm(p=2, dim=-1, keepdim=True).clamp_min(self.cfg.eps))
        s = self.shapelets
        s_norm = s / (s.norm(p=2, dim=-1, keepdim=True).clamp_min(self.cfg.eps))

        # dot product over L then average over channels
        # w_norm: [bs, C, T', L]
        # s_norm: [P,  C, L]
        # -> want [bs, T', P]
        # compute per-channel dot then mean over C:
        # expand: w -> [bs, 1, C, T', L], s -> [1, P, C, 1, L]
        w2 = w_norm.unsqueeze(1)
        s2 = s_norm.unsqueeze(0).unsqueeze(3)
        dot = (w2 * s2).sum(dim=-1)  # [bs, P, C, T']
        cos = dot.mean(dim=2)        # [bs, P, T']
        cos = cos.permute(0, 2, 1).contiguous()  # [bs, T', P]

        if self.cfg.cosine_mode == "relu":
            act = F.relu(cos)
        else:
            # shift to [0,1]
            act = (cos + 1.0) * 0.5
        return act

    def _xcorr_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [bs, C, T]
        return: act_map [bs, T', P]  where T' = T - L + 1 (valid conv)
        """
        bs, C, T = x.shape
        L = self.cfg.shapelet_len
        if T < L:
            return torch.zeros(bs, T, self.cfg.num_shapelets, device=x.device, dtype=x.dtype)

        # Use conv1d: weight should be [P, C, L]
        w = self.shapelets
        if self.cfg.znorm:
            # normalize kernels to be scale-invariant
            w = self._znorm_shapelets(w)

        # valid conv -> [bs, P, T']
        y = F.conv1d(x, w, bias=None, stride=1, padding=0)
        y = y.permute(0, 2, 1).contiguous()  # [bs, T', P]
        # similarity-like; keep positive part
        act = F.relu(y)
        return act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [bs, T, C] or [bs, T] or [T, C]
        returns activations: [bs, T, P]
        """
        # normalize input shapes
        if x.dim() == 2:
            # could be [bs, T] or [T, C]
            if x.shape[0] > 1 and x.shape[1] > 1:
                # ambiguous; assume [bs, T]
                x = x.unsqueeze(-1)
            else:
                x = x.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        elif x.dim() == 3:
            pass
        else:
            raise ValueError(f"Expected x dim 2 or 3, got {x.dim()}")

        bs, T, C = x.shape
        if C != self.cfg.in_channels:
            raise ValueError(f"in_channels mismatch: x has C={C}, model expects {self.cfg.in_channels}")

        # to [bs, C, T]
        x_ct = x.permute(0, 2, 1).contiguous()

        dm = self.cfg.dist_measure
        if dm == "euclidean":
            act_map = self._euclidean_activation(x_ct)  # [bs, T', P]
        elif dm == "cosine":
            act_map = self._cosine_activation(x_ct)
        elif dm == "cross-correlation":
            act_map = self._xcorr_activation(x_ct)
        else:
            raise ValueError(f"Unknown dist_measure: {dm}")

        # pad back to [bs, T, P]
        act_full = self._pad_to_T(act_map, T)
        return act_full
