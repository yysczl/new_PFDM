from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LABEL_COLUMNS = ("stress_score", "anxiety_score", "depression_score")
EMOTION_NAMES = ("calm", "fearness", "happiness", "sadness", "tension")


@dataclass
class PFDMData:
    ppg: np.ndarray
    prv: np.ndarray
    prv_mask: np.ndarray
    stats: np.ndarray
    stress: np.ndarray
    aux: np.ndarray
    emotion: np.ndarray
    row_id: np.ndarray
    sample_id: np.ndarray
    emotion_name: np.ndarray


def _read_numeric(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError(f"non-numeric or missing values found in {path}")
    return numeric


def _load_prv_lengths(path: Path | None) -> Dict[Tuple[str, int], int]:
    if path is None or not path.exists():
        return {}
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"emotion", "row_index", "corrected_interval_count"}
    if not required.issubset(frame.columns):
        return {}
    return {
        (str(row.emotion), int(row.row_index)): int(row.corrected_interval_count)
        for row in frame.itertuples(index=False)
    }


def _stats(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    rows = []
    for i, row in enumerate(x.astype(np.float32)):
        valid = row if mask is None else row[mask[i]]
        if len(valid) == 0:
            valid = row
        diff = np.diff(valid)
        rows.append(
            [
                float(valid.mean()),
                float(valid.std()),
                float(valid.min()),
                float(valid.max()),
                float(np.quantile(valid, 0.25)),
                float(np.quantile(valid, 0.50)),
                float(np.quantile(valid, 0.75)),
                float(diff.std()) if len(diff) else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def resolve_emotions(emotion: str, all_emotions: Iterable[str] = EMOTION_NAMES) -> List[str]:
    if emotion == "all":
        return list(all_emotions)
    if emotion not in all_emotions:
        raise ValueError(f"unknown emotion {emotion!r}; expected all or one of {list(all_emotions)}")
    return [emotion]


def load_pfdm_data(ppg_dir: Path, prv_dir: Path, emotion: str, prv_report: Path | None = None) -> PFDMData:
    emotions = resolve_emotions(emotion)
    lengths = _load_prv_lengths(prv_report)
    ppg_parts: List[np.ndarray] = []
    prv_parts: List[np.ndarray] = []
    mask_parts: List[np.ndarray] = []
    stats_parts: List[np.ndarray] = []
    stress_parts: List[np.ndarray] = []
    aux_parts: List[np.ndarray] = []
    emotion_parts: List[np.ndarray] = []
    row_parts: List[np.ndarray] = []
    sample_parts: List[np.ndarray] = []
    name_parts: List[np.ndarray] = []

    for emotion_name in emotions:
        ppg_path = ppg_dir / f"{emotion_name}.csv"
        prv_path = prv_dir / f"{emotion_name}.csv"
        if not ppg_path.exists() or not prv_path.exists():
            raise FileNotFoundError(f"missing paired files: {ppg_path} / {prv_path}")
        ppg_df = _read_numeric(ppg_path)
        prv_df = _read_numeric(prv_path)
        if len(ppg_df) != len(prv_df):
            raise ValueError(f"row count mismatch for {emotion_name}: PPG={len(ppg_df)}, PRV={len(prv_df)}")
        if not all(col in ppg_df.columns for col in LABEL_COLUMNS):
            raise ValueError(f"missing label columns in {ppg_path}")
        if not np.allclose(ppg_df[list(LABEL_COLUMNS)].to_numpy(), prv_df[list(LABEL_COLUMNS)].to_numpy()):
            raise ValueError(f"PPG/PRV labels are not aligned for {emotion_name}")

        ppg = ppg_df.drop(columns=list(LABEL_COLUMNS)).to_numpy(dtype=np.float32)
        prv = prv_df.drop(columns=list(LABEL_COLUMNS)).to_numpy(dtype=np.float32)
        mask = np.ones_like(prv, dtype=bool)
        for row_index in range(1, len(prv) + 1):
            valid = lengths.get((emotion_name, row_index), prv.shape[1])
            mask[row_index - 1, max(0, min(valid, prv.shape[1])) :] = False

        row_id = np.arange(len(ppg), dtype=np.float32)
        row_norm = row_id / max(float(len(ppg) - 1), 1.0)
        emotion_id = EMOTION_NAMES.index(emotion_name)
        emotion_one_hot = np.zeros((len(ppg), len(EMOTION_NAMES)), dtype=np.float32)
        emotion_one_hot[:, emotion_id] = 1.0
        stats = np.concatenate([_stats(ppg), _stats(prv, mask), row_norm[:, None], emotion_one_hot], axis=1)

        ppg_parts.append(ppg)
        prv_parts.append(prv)
        mask_parts.append(mask)
        stats_parts.append(stats)
        stress_parts.append(ppg_df["stress_score"].to_numpy(dtype=np.float32))
        aux_parts.append(ppg_df[["anxiety_score", "depression_score"]].to_numpy(dtype=np.float32))
        emotion_parts.append(np.full(len(ppg), emotion_id, dtype=np.int64))
        row_parts.append(row_norm.astype(np.float32))
        sample_parts.append(np.arange(len(ppg), dtype=np.int64))
        name_parts.append(np.asarray([emotion_name] * len(ppg), dtype=object))

    return PFDMData(
        ppg=np.concatenate(ppg_parts),
        prv=np.concatenate(prv_parts),
        prv_mask=np.concatenate(mask_parts),
        stats=np.concatenate(stats_parts),
        stress=np.concatenate(stress_parts),
        aux=np.concatenate(aux_parts),
        emotion=np.concatenate(emotion_parts),
        row_id=np.concatenate(row_parts),
        sample_id=np.concatenate(sample_parts),
        emotion_name=np.concatenate(name_parts),
    )


class PFDMDataset(Dataset):
    def __init__(self, data: PFDMData, indices: np.ndarray, ppg: np.ndarray, prv: np.ndarray, stats: np.ndarray) -> None:
        self.data = data
        self.indices = indices.astype(np.int64)
        self.ppg = ppg.astype(np.float32)
        self.prv = prv.astype(np.float32)
        self.stats = stats.astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        src = self.indices[item]
        return {
            "ppg": torch.from_numpy(self.ppg[item, :, None]),
            "prv": torch.from_numpy(self.prv[item, :, None]),
            "prv_mask": torch.from_numpy(self.data.prv_mask[src].astype(bool)),
            "stats": torch.from_numpy(self.stats[item]),
            "stress": torch.tensor(self.data.stress[src], dtype=torch.float32),
            "aux": torch.from_numpy(self.data.aux[src].astype(np.float32)),
            "emotion": torch.tensor(int(self.data.emotion[src]), dtype=torch.long),
            "index": torch.tensor(int(src), dtype=torch.long),
        }
