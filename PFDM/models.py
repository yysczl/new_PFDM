from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class AttentionPool1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 1)
        self.score = nn.Sequential(nn.Linear(channels, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        if mask is not None:
            mask = mask.bool()
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        if mask is not None:
            weights = weights * mask.to(weights.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class AntiAliasDownsample1d(nn.Module):
    def __init__(self, channels: int, stride: int = 4) -> None:
        super().__init__()
        kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel = (kernel / kernel.sum()).view(1, 1, -1).repeat(channels, 1, 1)
        self.register_buffer("kernel", kernel)
        self.channels = channels
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            x,
            self.kernel.to(dtype=x.dtype),
            stride=self.stride,
            padding=self.kernel.size(-1) // 2,
            groups=self.channels,
        )


class PhysioCycleEncoding(nn.Module):
    def __init__(
        self,
        channels: int,
        sample_rate: float = 100.0,
        downsample_factor: int = 16,
        min_bpm: float = 50.0,
        max_bpm: float = 150.0,
    ) -> None:
        super().__init__()
        if min_bpm <= 0 or max_bpm <= min_bpm:
            raise ValueError("expected 0 < min_bpm < max_bpm")
        frequency_count = max(channels // 2, 1)
        self.register_buffer("freqs_hz", torch.linspace(min_bpm / 60.0, max_bpm / 60.0, frequency_count))
        self.sample_rate = float(sample_rate)
        self.downsample_factor = int(downsample_factor)
        self.cycle_proj = nn.Linear(frequency_count * 2, channels)
        self.cycle_logit = nn.Parameter(torch.tensor(-2.0))
        self.channels = channels

    def _standard_encoding(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        indices = torch.arange(0, self.channels, 2, device=device, dtype=dtype)
        scale = torch.exp(indices * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / self.channels))
        encoding = torch.zeros(length, self.channels, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(position * scale)
        if self.channels > 1:
            encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].size(1)])
        return encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        dtype = x.dtype
        device = x.device
        time = torch.arange(length, device=device, dtype=dtype) * (self.downsample_factor / self.sample_rate)
        phase = 2.0 * torch.pi * time[:, None] * self.freqs_hz.to(device=device, dtype=dtype)[None, :]
        cycle = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        cycle = self.cycle_proj(cycle)
        position = self._standard_encoding(length, device, dtype)
        return x + position.unsqueeze(0) + torch.sigmoid(self.cycle_logit) * cycle.unsqueeze(0)


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
        min_bpm: float = 50.0,
        max_bpm: float = 150.0,
        use_anti_alias: bool = True,
    ) -> None:
        super().__init__()
        self.use_frequency_branch = use_frequency_branch
        self.use_stress_gate = use_stress_gate

        first_downsample: nn.Module
        second_downsample: nn.Module
        if use_anti_alias:
            first_conv = nn.Conv1d(1, channels, kernel_size=9, stride=1, padding=4)
            first_downsample = AntiAliasDownsample1d(channels, stride=4)
            second_conv = nn.Conv1d(channels, embedding_size, kernel_size=7, stride=1, padding=3)
            second_downsample = AntiAliasDownsample1d(embedding_size, stride=4)
        else:
            first_conv = nn.Conv1d(1, channels, kernel_size=9, stride=4, padding=4)
            first_downsample = nn.Identity()
            second_conv = nn.Conv1d(channels, embedding_size, kernel_size=7, stride=4, padding=3)
            second_downsample = nn.Identity()
        self.stem = nn.Sequential(
            first_conv,
            first_downsample,
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            second_conv,
            second_downsample,
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cycle = (
            PhysioCycleEncoding(embedding_size, sample_rate, 16, min_bpm, max_bpm)
            if use_cycle_encoding
            else nn.Identity()
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=transformer_heads,
            dim_feedforward=embedding_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, transformer_layers)

        self.multi_scale = nn.ModuleList(
            [
                nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1),
                nn.Conv1d(embedding_size, embedding_size, kernel_size=7, padding=3),
                nn.Conv1d(embedding_size, embedding_size, kernel_size=15, padding=7),
            ]
        )
        self.multi_scale_proj = nn.Conv1d(embedding_size * 3, embedding_size, kernel_size=1)

        if use_frequency_branch:
            frequency_hidden = max(embedding_size // 2, 1)
            self.frequency_gate = nn.Sequential(
                nn.Linear(embedding_size, frequency_hidden),
                nn.GELU(),
                nn.Linear(frequency_hidden, embedding_size),
                nn.Sigmoid(),
            )
            nn.init.constant_(self.frequency_gate[2].bias, 2.0)
            self.frequency_norm = nn.LayerNorm(embedding_size)

        branch_count = 3 if use_frequency_branch else 2
        self.branch_proj = nn.Sequential(
            nn.Linear(embedding_size * branch_count, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
        )
        gate_hidden = max(embedding_size // 4, 1)
        self.channel_gate = nn.Sequential(
            nn.Linear(embedding_size, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, embedding_size),
            nn.Sigmoid(),
        )
        self.temporal_gate = nn.Sequential(nn.Conv1d(2, 1, kernel_size=7, padding=3), nn.Sigmoid())
        self.token_norm = nn.LayerNorm(embedding_size)
        self.pool = AttentionPool1d(embedding_size)
        self.pool_norm = nn.LayerNorm(embedding_size)

    def frequency_enhance(self, embedding: torch.Tensor) -> torch.Tensor:
        length = embedding.size(1)
        channel_first = embedding.transpose(1, 2).contiguous()
        spectrum = torch.fft.fft(channel_first, dim=-1).transpose(1, 2)
        positive_bins = length // 2 + 1
        positive_spectrum = spectrum[:, :positive_bins]
        positive_gain = self.frequency_gate(positive_spectrum.abs())

        # Mirror the real-valued gain so the filtered spectrum remains
        # conjugate-symmetric and the inverse transform stays real-valued.
        if length % 2 == 0:
            negative_gain = positive_gain[:, 1:-1].flip(dims=(1,))
        else:
            negative_gain = positive_gain[:, 1:].flip(dims=(1,))
        full_gain = torch.cat([positive_gain, negative_gain], dim=1)
        filtered = (spectrum * full_gain).transpose(1, 2).contiguous()
        enhanced = torch.fft.ifft(filtered, dim=-1).real.transpose(1, 2)
        return self.frequency_norm(enhanced)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.stem(x.transpose(1, 2)).transpose(1, 2)
        time_features = self.transformer(self.cycle(embedding))

        conv_input = time_features.transpose(1, 2)
        scale_features = [F.gelu(conv(conv_input)) for conv in self.multi_scale]
        multi_features = self.multi_scale_proj(torch.cat(scale_features, dim=1)).transpose(1, 2)

        branches = [time_features, multi_features]
        if self.use_frequency_branch:
            branches.append(self.frequency_enhance(embedding))
        fused = self.branch_proj(torch.cat(branches, dim=-1))

        if self.use_stress_gate:
            channel_weights = self.channel_gate(fused.mean(dim=1)).unsqueeze(1)
            temporal_context = torch.stack([fused.mean(dim=-1), fused.amax(dim=-1)], dim=1)
            temporal_weights = self.temporal_gate(temporal_context).transpose(1, 2)
            fused = fused + fused * channel_weights * temporal_weights

        tokens = self.token_norm(fused)
        return {"tokens": tokens, "pooled": self.pool_norm(self.pool(tokens)), "embedding": embedding}


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
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            mask = mask.to(device=tokens.device, dtype=torch.bool)
        return {"tokens": tokens, "pooled": self.norm(self.pool(tokens, mask)), "mask": mask}


class FusionBlock(nn.Module):
    def __init__(self, embedding_size: int, output_size: int, mode: str, heads: int, dropout: float) -> None:
        super().__init__()
        if mode not in {"concat", "oneway_attention", "cross_attention"}:
            raise ValueError(f"unsupported fusion mode: {mode}")
        self.mode = mode
        self.out_size = output_size
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_size * 2, output_size),
            nn.LayerNorm(output_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if mode != "concat":
            self.ppg_to_prv = nn.MultiheadAttention(embedding_size, heads, dropout=dropout, batch_first=True)
            self.ppg_gate = nn.Sequential(nn.Linear(embedding_size * 2, embedding_size), nn.Sigmoid())
            self.ppg_norm = nn.LayerNorm(embedding_size)
            self.ppg_pool = AttentionPool1d(embedding_size)
        if mode == "cross_attention":
            self.prv_to_ppg = nn.MultiheadAttention(embedding_size, heads, dropout=dropout, batch_first=True)
            self.prv_gate = nn.Sequential(nn.Linear(embedding_size * 2, embedding_size), nn.Sigmoid())
            self.prv_norm = nn.LayerNorm(embedding_size)
            self.prv_pool = AttentionPool1d(embedding_size)
            self.weight_net = nn.Sequential(
                nn.Linear(embedding_size * 2, embedding_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_size, 2),
            )

    @staticmethod
    def _safe_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        mask = mask.bool()
        empty = ~mask.any(dim=1)
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True
        return mask

    def forward(
        self, ppg: Dict[str, torch.Tensor], prv: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ppg_features = ppg["pooled"]
        prv_features = prv["pooled"]
        if self.mode == "concat":
            return self.output_proj(torch.cat([ppg_features, prv_features], dim=-1)), None

        prv_mask = self._safe_mask(prv.get("mask"))
        attended_ppg, _ = self.ppg_to_prv(
            ppg["tokens"],
            prv["tokens"],
            prv["tokens"],
            key_padding_mask=None if prv_mask is None else ~prv_mask,
            need_weights=False,
        )
        ppg_gate = self.ppg_gate(torch.cat([ppg["tokens"], attended_ppg], dim=-1))
        enhanced_ppg = self.ppg_norm(ppg["tokens"] + ppg_gate * attended_ppg)
        ppg_features = self.ppg_pool(enhanced_ppg)
        if self.mode == "oneway_attention":
            return self.output_proj(torch.cat([ppg_features, prv_features], dim=-1)), None

        attended_prv, _ = self.prv_to_ppg(prv["tokens"], ppg["tokens"], ppg["tokens"], need_weights=False)
        prv_gate = self.prv_gate(torch.cat([prv["tokens"], attended_prv], dim=-1))
        enhanced_prv = self.prv_norm(prv["tokens"] + prv_gate * attended_prv)
        prv_features = self.prv_pool(enhanced_prv, prv_mask)

        weights = torch.softmax(self.weight_net(torch.cat([ppg_features, prv_features], dim=-1)), dim=-1)
        weighted = torch.cat(
            [ppg_features * weights[:, :1], prv_features * weights[:, 1:2]],
            dim=-1,
        )
        return self.output_proj(weighted), weights


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
        use_stats: bool = False,
        use_cycle_encoding: bool = True,
        use_frequency_branch: bool = True,
        use_stress_gate: bool = True,
        min_bpm: float = 50.0,
        max_bpm: float = 150.0,
        use_anti_alias: bool = True,
    ) -> None:
        super().__init__()
        if modalities not in {"ppg", "prv", "both"}:
            raise ValueError("modalities must be ppg, prv, or both")
        if task_mode not in {"stress_only", "fixed_multitask", "uncertainty"}:
            raise ValueError("task_mode must be stress_only, fixed_multitask, or uncertainty")
        self.modalities = modalities
        self.task_mode = task_mode
        self.use_stats = use_stats and stats_size > 0

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
                min_bpm,
                max_bpm,
                use_anti_alias,
            )
        if modalities in {"prv", "both"}:
            self.prv_encoder = PRVEncoder(prv_channels, embedding_size, dropout)

        if modalities == "both":
            self.fusion = FusionBlock(embedding_size, hidden_size, fusion, transformer_heads, dropout)
            fused_size = self.fusion.out_size
        else:
            fused_size = embedding_size
        if self.use_stats:
            self.stats_encoder = nn.Sequential(
                nn.Linear(stats_size, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(dropout),
            )
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
        head_size = max(hidden_size // 2, 1)
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, 1),
        )
        if task_mode != "stress_only":
            self.emotion_head = nn.Sequential(
                nn.Linear(hidden_size, head_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_size, num_emotions),
            )
        if task_mode == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        fusion_weights = None
        if self.modalities == "ppg":
            fused = self.ppg_encoder(batch["ppg"])["pooled"]
        elif self.modalities == "prv":
            fused = self.prv_encoder(batch["prv"], batch.get("prv_mask"))["pooled"]
        else:
            ppg = self.ppg_encoder(batch["ppg"])
            prv = self.prv_encoder(batch["prv"], batch.get("prv_mask"))
            fused, fusion_weights = self.fusion(ppg, prv)
        if self.use_stats:
            fused = torch.cat([fused, self.stats_encoder(batch["stats"])], dim=-1)

        shared = self.shared(fused)
        out: Dict[str, torch.Tensor] = {"stress": self.stress_head(shared).squeeze(-1)}
        if self.task_mode != "stress_only":
            out["emotion"] = self.emotion_head(shared)
        if self.task_mode == "uncertainty":
            out["log_vars"] = self.log_vars
        if fusion_weights is not None:
            out["fusion_weights"] = fusion_weights
        return out
