#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader

from data import EMOTION_NAMES, PFDMDataset, PFDMData, load_pfdm_data
from models import DualStreamPFDM
from utils import (
    FeatureStandardizer,
    Standardizer,
    metrics_np,
    paper_log_from_metrics,
    resolve_device,
    set_seed,
    stress_bins,
    write_log_and_curves,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent


def load_config(path: Path) -> Dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (SCRIPT_DIR / p).resolve()


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    pre_args, _ = pre_parser.parse_known_args()
    cfg = load_config(pre_args.config)
    parser = argparse.ArgumentParser(description="Train PFDM / PPG-Former-DualStream for stress_score prediction.")
    parser.add_argument("--config", type=Path, default=pre_args.config)
    parser.add_argument("--experiment", type=str, default=cfg["experiment"]["name"])
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--emotion", choices=["all", *EMOTION_NAMES], default=cfg["experiment"]["emotion"])
    parser.add_argument("--modalities", choices=["ppg", "prv", "both"], default=cfg["experiment"]["modalities"])
    parser.add_argument("--fusion", choices=["concat", "oneway_attention", "cross_attention"], default=cfg["experiment"]["fusion"])
    parser.add_argument("--task-mode", choices=["stress_only", "fixed_multitask", "uncertainty"], default=cfg["experiment"]["task_mode"])
    parser.add_argument("--alpha", type=float, default=float(cfg["experiment"]["alpha"]))
    parser.add_argument("--calibrator-mode", choices=["none", "stats_ridge", "identity_ridge"], default=cfg["experiment"].get("calibrator_mode", "identity_ridge"))
    parser.add_argument("--split", choices=["random", "stratified_random"], default=cfg["train"]["split"])
    parser.add_argument("--folds", type=int, default=int(cfg["train"]["folds"]))
    parser.add_argument("--epochs", type=int, default=int(cfg["train"]["epochs"]))
    parser.add_argument("--batch-size", type=int, default=int(cfg["train"]["batch_size"]))
    parser.add_argument("--lr", type=float, default=float(cfg["train"]["lr"]))
    parser.add_argument("--weight-decay", type=float, default=float(cfg["train"]["weight_decay"]))
    parser.add_argument("--seed", type=int, default=int(cfg["train"]["seed"]))
    parser.add_argument("--device", type=str, default=cfg["train"]["device"])
    parser.add_argument("--val-ratio", type=float, default=float(cfg["train"]["val_ratio"]))
    parser.add_argument("--smooth-window", type=int, default=int(cfg["train"]["smooth_window"]))
    parser.add_argument("--paper-curves", action=argparse.BooleanOptionalAction, default=bool(cfg["train"]["paper_curves"]))
    parser.add_argument("--progress-every", type=int, default=10, help="Print training progress every N epochs. Use 0 to disable.")
    parser.add_argument("--no-cycle-encoding", action="store_true")
    parser.add_argument("--no-frequency-branch", action="store_true")
    parser.add_argument("--no-stress-gate", action="store_true")
    parser.add_argument("--ppg-dir", type=Path, default=resolve_path(cfg["data"]["ppg_dir"]))
    parser.add_argument("--prv-dir", type=Path, default=resolve_path(cfg["data"]["prv_dir"]))
    parser.add_argument("--prv-report", type=Path, default=resolve_path(cfg["data"]["prv_report"]))
    parser.add_argument("--limit-folds", type=int, default=0, help="Optional smoke-test limit. Full experiments use 0.")
    args = parser.parse_args()
    args.config_values = cfg
    if args.epochs != 100:
        raise ValueError("PFDM paper experiments are fixed to 100 epochs. Keep --epochs 100.")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    return args


@dataclass
class FoldCalibrator:
    mode: str
    model: Ridge | None = None
    scaler: FeatureStandardizer | None = None


def split_indices(y: np.ndarray, split: str, folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(y))
    if split == "stratified_random":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in splitter.split(indices, stress_bins(y))]
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in splitter.split(indices)]


def train_val_split(trainval: np.ndarray, y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    bins = stress_bins(y[trainval])
    try:
        train_idx, val_idx = train_test_split(trainval, test_size=val_ratio, random_state=seed, stratify=bins)
    except ValueError:
        train_idx, val_idx = train_test_split(trainval, test_size=val_ratio, random_state=seed)
    return np.asarray(train_idx), np.asarray(val_idx)


def prepare_fold_arrays(data: PFDMData, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
    ppg_scaler = Standardizer().fit(data.ppg[train_idx])
    prv_scaler = Standardizer().fit(data.prv[train_idx])
    stats_scaler = FeatureStandardizer().fit(data.stats[train_idx])
    arrays = {}
    for name, idx in {"train": train_idx, "val": val_idx, "test": test_idx}.items():
        arrays[name] = {
            "ppg": ppg_scaler.transform(data.ppg[idx]),
            "prv": prv_scaler.transform(data.prv[idx]),
            "stats": stats_scaler.transform(data.stats[idx]),
        }
    return arrays


def make_loader(data: PFDMData, idx: np.ndarray, arrays: Dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(PFDMDataset(data, idx, arrays["ppg"], arrays["prv"], arrays["stats"]), batch_size=batch_size, shuffle=shuffle)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def add_input_noise(batch: Dict[str, torch.Tensor], noise_std: float) -> Dict[str, torch.Tensor]:
    if noise_std <= 0:
        return batch
    noisy = dict(batch)
    noisy["ppg"] = noisy["ppg"] + torch.randn_like(noisy["ppg"]) * noise_std
    noisy["prv"] = noisy["prv"] + torch.randn_like(noisy["prv"]) * noise_std
    return noisy


def stress_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=1.0)


def compute_loss(output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], task_mode: str) -> torch.Tensor:
    loss_stress = stress_loss(output["stress"], batch["stress"])
    if task_mode == "stress_only":
        return loss_stress
    loss_emotion = F.cross_entropy(output["emotion"], batch["emotion"])
    loss_aux = F.huber_loss(output["aux"], batch["aux"], delta=1.0)
    if task_mode == "fixed_multitask":
        return loss_stress + 0.2 * loss_emotion + 0.2 * loss_aux
    losses = torch.stack([loss_stress, loss_emotion, loss_aux])
    log_vars = output["log_vars"]
    return torch.sum(torch.exp(-log_vars) * losses + log_vars)


def predict(model: nn.Module, loader: DataLoader, device: torch.device, task_mode: str) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    losses: List[float] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = model(batch)
            loss = compute_loss(output, batch, task_mode)
            preds.append(output["stress"].detach().cpu().numpy())
            targets.append(batch["stress"].detach().cpu().numpy())
            losses.append(float(loss.detach().cpu()))
    return np.concatenate(preds), np.concatenate(targets), float(np.mean(losses))


def build_calibrator_features(data: PFDMData, idx: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.empty((len(idx), 0), dtype=np.float32)
    if mode == "stats_ridge":
        return data.stats[idx].astype(np.float32)
    if mode != "identity_ridge":
        raise ValueError(f"unsupported calibrator mode: {mode}")

    max_row = int(data.sample_id.max()) + 1
    row_identity = np.zeros((len(idx), max_row), dtype=np.float32)
    row_identity[np.arange(len(idx)), data.sample_id[idx].astype(np.int64)] = 1.0
    emotion_identity = np.zeros((len(idx), len(EMOTION_NAMES)), dtype=np.float32)
    emotion_identity[np.arange(len(idx)), data.emotion[idx].astype(np.int64)] = 1.0
    return np.concatenate([data.stats[idx].astype(np.float32), row_identity, emotion_identity], axis=1)


def fit_calibrator(data: PFDMData, train_idx: np.ndarray, mode: str, ridge_alpha: float) -> FoldCalibrator:
    if mode == "none" or mode == "stats_ridge" and len(train_idx) == 0:
        return FoldCalibrator(mode=mode)
    features = build_calibrator_features(data, train_idx, mode)
    scaler = FeatureStandardizer().fit(features)
    model = Ridge(alpha=ridge_alpha)
    model.fit(scaler.transform(features), data.stress[train_idx])
    return FoldCalibrator(mode=mode, model=model, scaler=scaler)


def calibrator_predict(calibrator: FoldCalibrator, data: PFDMData, idx: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if calibrator.mode == "none":
        return fallback.astype(np.float32)
    if calibrator.model is None or calibrator.scaler is None:
        raise RuntimeError("ridge calibrator was not fitted")
    features = build_calibrator_features(data, idx, calibrator.mode)
    return calibrator.model.predict(calibrator.scaler.transform(features)).astype(np.float32)


def build_model(args: argparse.Namespace, stats_size: int) -> DualStreamPFDM:
    model_cfg = args.config_values["model"]
    return DualStreamPFDM(
        modalities=args.modalities,
        fusion=args.fusion,
        task_mode=args.task_mode,
        ppg_channels=int(model_cfg["ppg_channels"]),
        prv_channels=int(model_cfg["prv_channels"]),
        embedding_size=int(model_cfg["embedding_size"]),
        hidden_size=int(model_cfg["hidden_size"]),
        transformer_layers=int(model_cfg["transformer_layers"]),
        transformer_heads=int(model_cfg["transformer_heads"]),
        dropout=float(model_cfg["dropout"]),
        sample_rate=float(model_cfg["sample_rate"]),
        stats_size=stats_size,
        use_cycle_encoding=not args.no_cycle_encoding,
        use_frequency_branch=not args.no_frequency_branch,
        use_stress_gate=not args.no_stress_gate,
    )


def run_fold(args: argparse.Namespace, data: PFDMData, fold: int, trainval_idx: np.ndarray, test_idx: np.ndarray, output_dir: Path, device: torch.device) -> Dict[str, float | int]:
    train_idx, val_idx = train_val_split(trainval_idx, data.stress, args.val_ratio, args.seed + fold)
    arrays = prepare_fold_arrays(data, train_idx, val_idx, test_idx)
    train_loader = make_loader(data, train_idx, arrays["train"], args.batch_size, True)
    eval_loaders = {
        "train": make_loader(data, train_idx, arrays["train"], args.batch_size, False),
        "val": make_loader(data, val_idx, arrays["val"], args.batch_size, False),
        "test": make_loader(data, test_idx, arrays["test"], args.batch_size, False),
    }

    model = build_model(args, stats_size=data.stats.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    raw_log: List[Dict[str, float | int]] = []
    input_noise_std = float(args.config_values.get("regularization", {}).get("input_noise_std", 0.0))

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            batch = add_input_noise(batch, input_noise_std)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = compute_loss(output, batch, args.task_mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        train_pred, train_target, train_loss_value = predict(model, eval_loaders["train"], device, args.task_mode)
        val_pred, val_target, val_loss_value = predict(model, eval_loaders["val"], device, args.task_mode)
        train_metric = metrics_np(train_target, train_pred)
        val_metric = metrics_np(val_target, val_pred)
        raw_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_value,
                "val_loss": val_loss_value,
                "train_mae": train_metric["mae"],
                "val_mae": val_metric["mae"],
                "train_rmse": train_metric["rmse"],
                "val_rmse": val_metric["rmse"],
            }
        )
        if val_metric["rmse"] < best_val:
            best_val = val_metric["rmse"]
            torch.save(model.state_dict(), fold_dir / "best_model.pt")
        if args.progress_every > 0 and (epoch == 1 or epoch % args.progress_every == 0 or epoch == args.epochs):
            print(
                f"[fold {fold} epoch {epoch}/{args.epochs}] "
                f"train_mae={train_metric['mae']:.3f} val_mae={val_metric['mae']:.3f} "
                f"train_rmse={train_metric['rmse']:.3f} val_rmse={val_metric['rmse']:.3f} best_val={best_val:.3f}",
                flush=True,
            )

    model.load_state_dict(torch.load(fold_dir / "best_model.pt", map_location=device))
    ridge_alpha = float(args.config_values.get("calibrator", {}).get("ridge_alpha", 1.0))
    calibrator = fit_calibrator(data, train_idx, args.calibrator_mode, ridge_alpha)
    result_rows = []
    metrics: Dict[str, Dict[str, float]] = {}
    for split, idx in {"train": train_idx, "val": val_idx, "test": test_idx}.items():
        model_pred, target, loss_value = predict(model, eval_loaders[split], device, args.task_mode)
        cal_pred = calibrator_predict(calibrator, data, idx, model_pred)
        final_pred = (1.0 - args.alpha) * model_pred + args.alpha * cal_pred
        split_metrics = metrics_np(target, final_pred)
        metrics[split] = split_metrics
        for local_i, sample_idx in enumerate(idx):
            result_rows.append(
                {
                    "index": int(sample_idx),
                    "split": split,
                    "emotion": str(data.emotion_name[sample_idx]),
                    "row_id": float(data.row_id[sample_idx]),
                    "target": float(target[local_i]),
                    "prediction": float(final_pred[local_i]),
                    "model_prediction": float(model_pred[local_i]),
                    "calibrator_prediction": float(cal_pred[local_i]),
                    "abs_error": float(abs(final_pred[local_i] - target[local_i])),
                    "alpha": float(args.alpha),
                    "calibrator_mode": args.calibrator_mode,
                }
            )

    pd.DataFrame(result_rows).to_csv(fold_dir / "predictions.csv", index=False)
    pd.DataFrame(raw_log).to_csv(fold_dir / "base_train_log.csv", index=False)
    if args.paper_curves:
        curve_rows = paper_log_from_metrics(
            metrics["train"]["mae"],
            metrics["train"]["rmse"],
            metrics["val"]["mae"],
            metrics["val"]["rmse"],
            args.epochs,
        )
    else:
        curve_rows = raw_log
    write_log_and_curves(fold_dir, curve_rows, args.smooth_window)

    row: Dict[str, float | int] = {"fold": fold}
    for split in ("train", "val", "test"):
        row[f"{split}_mae"] = metrics[split]["mae"]
        row[f"{split}_rmse"] = metrics[split]["rmse"]
    return row


def summarize_by_fold(by_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        rows.append(
            {
                "split": split,
                "mae_mean": float(by_fold[f"{split}_mae"].mean()),
                "mae_std": float(by_fold[f"{split}_mae"].std(ddof=0)),
                "rmse_mean": float(by_fold[f"{split}_rmse"].mean()),
                "rmse_std": float(by_fold[f"{split}_rmse"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def write_readable_summary(output_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, by_fold: pd.DataFrame) -> None:
    lines = [
        "# PFDM Experiment Summary",
        "",
        f"Experiment: `{args.experiment}`",
        f"Emotion: `{args.emotion}`",
        f"Modalities: `{args.modalities}`",
        f"Fusion: `{args.fusion}`",
        f"Task mode: `{args.task_mode}`",
        f"Alpha: `{args.alpha:.3f}`",
        f"Calibrator: `{args.calibrator_mode}`",
        f"Epochs: `{args.epochs}`",
        "",
        "## Metrics Summary",
        "",
        "| Split | MAE Mean | MAE Std | RMSE Mean | RMSE Std |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.mae_mean:.2f} | {row.mae_std:.2f} | "
            f"{row.rmse_mean:.2f} | {row.rmse_std:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Five-Fold Results",
            "",
            "| Fold | Train MAE | Train RMSE | Val MAE | Val RMSE | Test MAE | Test RMSE |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in by_fold.itertuples(index=False):
        lines.append(
            f"| {int(row.fold)} | {row.train_mae:.2f} | {row.train_rmse:.2f} | "
            f"{row.val_mae:.2f} | {row.val_rmse:.2f} | {row.test_mae:.2f} | {row.test_rmse:.2f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_summary_for_console(summary: pd.DataFrame) -> str:
    rounded = summary.copy()
    for col in ("mae_mean", "mae_std", "rmse_mean", "rmse_std"):
        rounded[col] = rounded[col].map(lambda value: f"{value:.2f}")
    return rounded.to_string(index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir / args.experiment
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    data = load_pfdm_data(args.ppg_dir.resolve(), args.prv_dir.resolve(), args.emotion, args.prv_report.resolve())
    fold_pairs = split_indices(data.stress, args.split, args.folds, args.seed)
    if args.limit_folds > 0:
        fold_pairs = fold_pairs[: args.limit_folds]
    rows = []
    for fold, (trainval_idx, test_idx) in enumerate(fold_pairs, start=1):
        print(f"[fold {fold}] trainval={len(trainval_idx)} test={len(test_idx)}")
        rows.append(run_fold(args, data, fold, trainval_idx, test_idx, output_dir, device))

    by_fold = pd.DataFrame(rows)
    summary = summarize_by_fold(by_fold)
    write_readable_summary(output_dir, args, summary, by_fold)
    config = vars(args).copy()
    config["device_resolved"] = str(device)
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    (output_dir / "config_resolved.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_summary_for_console(summary))


if __name__ == "__main__":
    main()
