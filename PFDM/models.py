from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class AttentionPool1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(channels, max(channels // 2, 1)), nn.Tanh(), nn.Linear(max(channels // 2, 1), 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class PhysioCycleEncoding(nn.Module):
    def __init__(self, channels: int, sample_rate: float = 100.0, min_bpm: float = 60.0, max_bpm: float = 100.0) -> None:
        super().__init__()
        self.channels = channels
        freqs_hz = torch.linspace(min_bpm / 60.0, max_bpm / 60.0, max(channels // 2, 1))
        self.register_buffer("freqs_hz", freqs_hz)
        self.sample_rate = sample_rate
        self.proj = nn.Linear(freqs_hz.numel() * 2, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        t = torch.arange(length, device=x.device, dtype=x.dtype) / float(self.sample_rate)
        phase = 2.0 * torch.pi * t[:, None] * self.freqs_hz.to(device=x.device, dtype=x.dtype)[None, :]
        enc = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return x + self.proj(enc).unsqueeze(0)


class PPGFormerEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 48,
        embedding_size: int = 96,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.2,
        sample_rate: float = 100.0,
        use_cycle_encoding: bool = True,
        use_frequency_branch: bool = True,
        use_stress_gate: bool = True,
    ) -> None:
        super().__init__()
        self.use_frequency_branch = use_frequency_branch
        self.use_stress_gate = use_stress_gate
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, embedding_size, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cycle = PhysioCycleEncoding(embedding_size, sample_rate) if use_cycle_encoding else nn.Identity()
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=transformer_heads,
            dim_feedforward=embedding_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, transformer_layers)
        self.pool = AttentionPool1d(embedding_size)
        self.multi_scale = nn.ModuleList(
            [
                nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1),
                nn.Conv1d(embedding_size, embedding_size, kernel_size=7, padding=3),
                nn.Conv1d(embedding_size, embedding_size, kernel_size=15, padding=7),
            ]
        )
        if use_frequency_branch:
            self.freq_proj = nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.LayerNorm(embedding_size), nn.GELU())
        branches = 3 if use_frequency_branch else 2
        self.gate = nn.Sequential(nn.Linear(embedding_size * branches, embedding_size), nn.GELU(), nn.Linear(embedding_size, branches))
        self.norm = nn.LayerNorm(embedding_size)

    def frequency_summary(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.size(1)
        freq_count = min(length, tokens.size(-1))
        positions = torch.linspace(0.0, 1.0, length, device=tokens.device, dtype=tokens.dtype)
        freqs = torch.linspace(1.0, float(freq_count), freq_count, device=tokens.device, dtype=tokens.dtype)
        basis = torch.cos(torch.pi * positions[:, None] * freqs[None, :])
        spectrum = torch.einsum("btc,tf->bfc", tokens, basis) / max(float(length), 1.0)
        spectrum = spectrum.abs().mean(dim=1)
        if spectrum.size(-1) != tokens.size(-1):
            spectrum = F.pad(spectrum, (0, tokens.size(-1) - spectrum.size(-1)))
        return spectrum

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.stem(x.transpose(1, 2)).transpose(1, 2)
        tokens = self.cycle(tokens)
        tokens = self.transformer(tokens)
        time_feat = self.pool(tokens)
        conv_in = tokens.transpose(1, 2)
        multi_feat = torch.stack([F.gelu(conv(conv_in)).mean(dim=-1) for conv in self.multi_scale], dim=0).mean(dim=0)
        feats = [time_feat, multi_feat]
        if self.use_frequency_branch:
            spectrum = self.frequency_summary(tokens)
            feats.append(self.freq_proj(spectrum))
        if self.use_stress_gate:
            weights = torch.softmax(self.gate(torch.cat(feats, dim=-1)), dim=-1)
            pooled = sum(feat * weights[:, i : i + 1] for i, feat in enumerate(feats))
        else:
            pooled = torch.stack(feats, dim=0).mean(dim=0)
        return {"tokens": tokens, "pooled": self.norm(pooled)}


class PRVEncoder(nn.Module):
    def __init__(self, channels: int = 32, embedding_size: int = 96, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, embedding_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = AttentionPool1d(embedding_size)
        self.norm = nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        tokens = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return {"tokens": tokens, "pooled": self.norm(self.pool(tokens, mask))}


class FusionBlock(nn.Module):
    def __init__(self, embedding_size: int, mode: str, heads: int, dropout: float) -> None:
        super().__init__()
        self.mode = mode
        if mode not in {"concat", "oneway_attention", "cross_attention"}:
            raise ValueError(f"unsupported fusion mode: {mode}")
        if mode != "concat":
            self.ppg_to_prv = nn.MultiheadAttention(embedding_size, heads, dropout=dropout, batch_first=True)
        if mode == "cross_attention":
            self.prv_to_ppg = nn.MultiheadAttention(embedding_size, heads, dropout=dropout, batch_first=True)
        if mode == "cross_attention":
            self.out_size = embedding_size * 4
        elif mode == "oneway_attention":
            self.out_size = embedding_size * 3
        else:
            self.out_size = embedding_size * 2

    def forward(self, ppg: Dict[str, torch.Tensor], prv: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == "concat":
            return torch.cat([ppg["pooled"], prv["pooled"]], dim=-1)
        q = ppg["tokens"]
        attended_ppg, _ = self.ppg_to_prv(q, prv["tokens"], prv["tokens"], need_weights=False)
        ppg_feat = attended_ppg.mean(dim=1)
        # PPG 对 PRV 做注意力：用 PPG 序列作为查询，去关注 PRV 序列中的相关信息，得到一个被 PRV 信息增强后的 ppg_feat。
        # 单向跨模态注意力采用 PPG-to-PRV attention，以 PPG 表征作为 Query，以 PRV 表征作为 Key/Value，引导 PPG 分支从 PRV 心率变异性特征中提取互补信息。
        if self.mode == "oneway_attention":
            return torch.cat([ppg_feat, ppg["pooled"], prv["pooled"]], dim=-1)
        attended_prv, _ = self.prv_to_ppg(prv["tokens"], ppg["tokens"], ppg["tokens"], need_weights=False)
        prv_feat = attended_prv.mean(dim=1)
        return torch.cat([ppg_feat, prv_feat, ppg["pooled"], prv["pooled"]], dim=-1)


class DualStreamPFDM(nn.Module):
    def __init__(
        self,
        modalities: str = "both",
        fusion: str = "cross_attention",
        task_mode: str = "uncertainty",
        ppg_channels: int = 48,
        prv_channels: int = 32,
        embedding_size: int = 96,
        hidden_size: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.2,
        sample_rate: float = 100.0,
        num_emotions: int = 5,
        stats_size: int = 22,
        use_cycle_encoding: bool = True,
        use_frequency_branch: bool = True,
        use_stress_gate: bool = True,
    ) -> None:
        super().__init__()
        if modalities not in {"ppg", "prv", "both"}:
            raise ValueError("modalities must be ppg, prv, or both")
        if task_mode not in {"stress_only", "fixed_multitask", "uncertainty"}:
            raise ValueError("task_mode must be stress_only, fixed_multitask, or uncertainty")
        self.modalities = modalities
        self.task_mode = task_mode
        if modalities in {"ppg", "both"}:
            self.ppg_encoder = PPGFormerEncoder(
                ppg_channels,
                embedding_size,
                transformer_layers,
                transformer_heads,
                dropout,
                sample_rate,
                use_cycle_encoding,
                use_frequency_branch,
                use_stress_gate,
            )
        if modalities in {"prv", "both"}:
            self.prv_encoder = PRVEncoder(prv_channels, embedding_size, dropout)
        if modalities == "both":
            self.fusion = FusionBlock(embedding_size, fusion, transformer_heads, dropout)
            fused_size = self.fusion.out_size
        else:
            fused_size = embedding_size
        self.stats_encoder = nn.Sequential(nn.Linear(stats_size, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout))
        fused_size += 32
        self.shared = nn.Sequential(
            nn.Linear(fused_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stress_head = nn.Linear(hidden_size, 1)
        if task_mode != "stress_only":
            self.emotion_head = nn.Linear(hidden_size, num_emotions)
            self.aux_head = nn.Linear(hidden_size, 2)
        if task_mode == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.modalities == "ppg":
            fused = self.ppg_encoder(batch["ppg"])["pooled"]
        elif self.modalities == "prv":
            fused = self.prv_encoder(batch["prv"], batch.get("prv_mask"))["pooled"]
        else:
            ppg = self.ppg_encoder(batch["ppg"])
            prv = self.prv_encoder(batch["prv"], batch.get("prv_mask"))
            fused = self.fusion(ppg, prv)
        fused = torch.cat([fused, self.stats_encoder(batch["stats"])], dim=-1)
        shared = self.shared(fused)
        out: Dict[str, torch.Tensor] = {"stress": self.stress_head(shared).squeeze(-1)}
        if self.task_mode != "stress_only":
            out["emotion"] = self.emotion_head(shared)
            out["aux"] = self.aux_head(shared)
        if self.task_mode == "uncertainty":
            out["log_vars"] = self.log_vars
        return out
