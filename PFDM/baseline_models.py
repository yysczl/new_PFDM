from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


DEEP_BASELINES = ("mlp", "fcn", "cnn1d", "resnet1d", "inceptiontime", "lstm", "gru", "cnn_gru", "tcn", "transformer")


def downsample_sequence(x: torch.Tensor, max_steps: int) -> torch.Tensor:
    if x.size(1) <= max_steps:
        return x
    positions = torch.linspace(0, x.size(1) - 1, max_steps, device=x.device)
    indices = positions.round().long()
    return x.index_select(1, indices)


def resize_sequence(x: torch.Tensor, steps: int) -> torch.Tensor:
    x = downsample_sequence(x, steps)
    if x.size(1) < steps:
        pad = x.new_zeros(x.size(0), steps - x.size(1), x.size(2))
        x = torch.cat([x, pad], dim=1)
    return x


def normalize_baseline_modality(modalities: str) -> str:
    aliases = {
        "ppg": "rppg",
        "rppg": "rppg",
        "prv": "hr",
        "hr": "hr",
        "both": "both",
    }
    if modalities not in aliases:
        raise ValueError("modalities must be one of rppg, hr, both, ppg, or prv")
    return aliases[modalities]


class AttentionPool1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 1)
        self.score = nn.Sequential(nn.Linear(channels, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class MLPSequenceEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, steps: int = 256) -> None:
        super().__init__()
        self.steps = steps
        self.net = nn.Sequential(
            nn.Linear(steps, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = resize_sequence(x, self.steps)
        return self.net(x.squeeze(-1))


class CNN1DEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        mid = max(hidden_size // 2, 16)
        self.net = nn.Sequential(
            nn.Conv1d(1, mid, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(mid, hidden_size, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = AttentionPool1d(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(self.pool(tokens))


class FCNEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        mid = max(hidden_size // 2, 16)
        self.net = nn.Sequential(
            nn.Conv1d(1, mid, kernel_size=7, padding=3),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(mid, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.net(x.transpose(1, 2))
        return self.norm(tokens.mean(dim=-1))


class ResNet1DBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class ResNet1DEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        stem_channels = max(hidden_size // 2, 16)
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
            nn.Conv1d(stem_channels, hidden_size, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResNet1DBlock(hidden_size, dropout),
            ResNet1DBlock(hidden_size, dropout),
            ResNet1DBlock(hidden_size, dropout),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.blocks(self.stem(x.transpose(1, 2)))
        return self.norm(tokens.mean(dim=-1))


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        branch_channels = max(out_channels // 4, 8)
        bottleneck_channels = max(in_channels // 2, 8)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1)
        self.conv9 = nn.Conv1d(bottleneck_channels, branch_channels, kernel_size=9, padding=4)
        self.conv19 = nn.Conv1d(bottleneck_channels, branch_channels, kernel_size=19, padding=9)
        self.conv39 = nn.Conv1d(bottleneck_channels, branch_channels, kernel_size=39, padding=19)
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
        )
        merged_channels = branch_channels * 4
        self.project = nn.Conv1d(merged_channels, out_channels, kernel_size=1)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(x)
        merged = torch.cat(
            [
                self.conv9(bottleneck),
                self.conv19(bottleneck),
                self.conv39(bottleneck),
                self.pool_branch(x),
            ],
            dim=1,
        )
        out = self.project(merged)
        out = self.dropout(self.bn(out))
        return F.gelu(out + self.residual(x))


class InceptionTimeEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        stem_channels = max(hidden_size // 2, 16)
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            InceptionBlock1D(stem_channels, hidden_size, dropout),
            InceptionBlock1D(hidden_size, hidden_size, dropout),
            InceptionBlock1D(hidden_size, hidden_size, dropout),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.blocks(self.stem(x.transpose(1, 2)))
        return self.norm(tokens.mean(dim=-1))


class GRUEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, max_steps: int = 128) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.input_proj = nn.Linear(1, hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.pool = AttentionPool1d(hidden_size * 2)
        self.out = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = downsample_sequence(x, self.max_steps)
        tokens = self.input_proj(x)
        tokens, _ = self.gru(tokens)
        return self.out(self.pool(self.dropout(tokens)))


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, max_steps: int = 128) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.input_proj = nn.Linear(1, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.pool = AttentionPool1d(hidden_size * 2)
        self.out = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = downsample_sequence(x, self.max_steps)
        tokens = self.input_proj(x)
        tokens, _ = self.lstm(tokens)
        return self.out(self.pool(self.dropout(tokens)))


class CNNGRUEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        mid = max(hidden_size // 2, 16)
        self.conv = nn.Sequential(
            nn.Conv1d(1, mid, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(mid, hidden_size, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.pool = AttentionPool1d(hidden_size * 2)
        self.out = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.conv(x.transpose(1, 2)).transpose(1, 2)
        tokens, _ = self.gru(tokens)
        return self.out(self.pool(tokens))


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            TCNBlock(hidden_size, dilation=1, dropout=dropout),
            TCNBlock(hidden_size, dilation=2, dropout=dropout),
            TCNBlock(hidden_size, dilation=4, dropout=dropout),
            TCNBlock(hidden_size, dilation=8, dropout=dropout),
        )
        self.pool = AttentionPool1d(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.blocks(self.in_proj(x.transpose(1, 2))).transpose(1, 2)
        return self.norm(self.pool(tokens))


def sinusoidal_encoding(length: int, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, channels, 2, device=device, dtype=dtype) * (-math.log(10000.0) / channels))
    encoding = torch.zeros(length, channels, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    if channels > 1:
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
    return encoding


class TransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        layers: int,
        heads: int,
        max_tokens: int = 256,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.input_proj = nn.Linear(1, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.pool = AttentionPool1d(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = downsample_sequence(x, self.max_tokens)
        tokens = self.input_proj(x)
        tokens = tokens + sinusoidal_encoding(tokens.size(1), tokens.size(2), tokens.device, tokens.dtype).unsqueeze(0)
        tokens = self.encoder(tokens)
        return self.norm(self.pool(tokens))


def make_encoder(
    model_name: str,
    hidden_size: int,
    dropout: float,
    transformer_layers: int,
    transformer_heads: int,
    gru_max_steps: int = 128,
    mlp_steps: int = 256,
) -> nn.Module:
    if model_name == "mlp":
        return MLPSequenceEncoder(hidden_size, dropout, steps=mlp_steps)
    if model_name == "fcn":
        return FCNEncoder(hidden_size, dropout)
    if model_name == "cnn1d":
        return CNN1DEncoder(hidden_size, dropout)
    if model_name == "resnet1d":
        return ResNet1DEncoder(hidden_size, dropout)
    if model_name == "inceptiontime":
        return InceptionTimeEncoder(hidden_size, dropout)
    if model_name == "lstm":
        return LSTMEncoder(hidden_size, dropout, max_steps=gru_max_steps)
    if model_name == "gru":
        return GRUEncoder(hidden_size, dropout, max_steps=gru_max_steps)
    if model_name == "cnn_gru":
        return CNNGRUEncoder(hidden_size, dropout)
    if model_name == "tcn":
        return TCNEncoder(hidden_size, dropout)
    if model_name == "transformer":
        return TransformerSequenceEncoder(hidden_size, dropout, transformer_layers, transformer_heads)
    raise ValueError(f"unknown deep baseline: {model_name}")


class BaselineRegressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        modalities: str,
        stats_size: int,
        hidden_size: int = 64,
        dropout: float = 0.2,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        gru_max_steps: int = 128,
        mlp_steps: int = 256,
        use_stats: bool = True,
    ) -> None:
        super().__init__()
        self.modalities = normalize_baseline_modality(modalities)
        self.use_stats = use_stats and stats_size > 0
        if self.modalities in {"rppg", "both"}:
            self.rppg_encoder = make_encoder(
                model_name, hidden_size, dropout, transformer_layers, transformer_heads, gru_max_steps, mlp_steps
            )
        if self.modalities in {"hr", "both"}:
            self.hr_encoder = make_encoder(
                model_name, hidden_size, dropout, transformer_layers, transformer_heads, gru_max_steps, mlp_steps
            )

        feature_size = hidden_size * (2 if self.modalities == "both" else 1)
        if self.use_stats:
            stats_hidden = min(max(stats_size * 2, 16), 48)
            self.stats_encoder = nn.Sequential(
                nn.Linear(stats_size, stats_hidden),
                nn.LayerNorm(stats_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            feature_size += stats_hidden

        self.head = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = []
        if self.modalities in {"rppg", "both"}:
            features.append(self.rppg_encoder(batch["ppg"]))
        if self.modalities in {"hr", "both"}:
            features.append(self.hr_encoder(batch["prv"]))
        if self.use_stats:
            features.append(self.stats_encoder(batch["stats"]))
        fused = torch.cat(features, dim=-1)
        return {"stress": self.head(fused).squeeze(-1)}
