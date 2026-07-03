#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader

from baseline_models import DEEP_BASELINES, BaselineRegressor, normalize_baseline_modality
from data import EMOTION_NAMES, PFDMDataset, PFDMData, load_pfdm_data
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
TABULAR_BASELINES = ("stats_ridge", "stats_svr")
ALL_BASELINES = (*TABULAR_BASELINES, *DEEP_BASELINES)


@dataclass
class FoldCalibrator:
    mode: str
    model: Ridge | None = None
    scaler: FeatureStandardizer | None = None


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

    parser = argparse.ArgumentParser(description="Train time-series baselines for PFDM stress_score prediction.")
    parser.add_argument("--config", type=Path, default=pre_args.config)
    parser.add_argument("--model", choices=ALL_BASELINES, required=True)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs_baselines")
    parser.add_argument("--emotion", choices=["all", *EMOTION_NAMES], default=cfg["experiment"]["emotion"])
    parser.add_argument("--modalities", choices=["rppg", "hr", "both", "ppg", "prv"], default="both")
    parser.add_argument("--split", choices=["random", "stratified_random"], default=cfg["train"]["split"])
    parser.add_argument("--folds", type=int, default=int(cfg["train"]["folds"]))
    parser.add_argument("--val-ratio", type=float, default=float(cfg["train"]["val_ratio"]))
    parser.add_argument("--seed", type=int, default=int(cfg["train"]["seed"]))
    parser.add_argument("--epochs", type=int, default=int(cfg["train"]["epochs"]))
    parser.add_argument("--batch-size", type=int, default=int(cfg["train"]["batch_size"]))
    parser.add_argument("--lr", type=float, default=float(cfg["train"]["lr"]))
    parser.add_argument("--weight-decay", type=float, default=float(cfg["train"]["weight_decay"]))
    parser.add_argument("--device", type=str, default=cfg["train"]["device"])
    parser.add_argument("--smooth-window", type=int, default=int(cfg["train"]["smooth_window"]))
    parser.add_argument("--paper-curves", action=argparse.BooleanOptionalAction, default=bool(cfg["train"]["paper_curves"]))
    parser.add_argument("--stats", action=argparse.BooleanOptionalAction, default=True, help="Use simple statistical features beside sequence features.")
    parser.add_argument("--input-noise-std", type=float, default=float(cfg.get("regularization", {}).get("input_noise_std", 0.0)))
    parser.add_argument("--hidden-size", type=int, default=int(cfg["model"]["hidden_size"]))
    parser.add_argument("--dropout", type=float, default=float(cfg["model"]["dropout"]))
    parser.add_argument("--mlp-steps", type=int, default=256, help="Fixed sequence length used by the MLP baseline.")
    parser.add_argument("--gru-max-steps", type=int, default=300, help="Maximum sequence length passed into the pure GRU baseline.")
    parser.add_argument("--transformer-layers", type=int, default=int(cfg["model"]["transformer_layers"]))
    parser.add_argument("--transformer-heads", type=int, default=int(cfg["model"]["transformer_heads"]))
    parser.add_argument("--progress-every", type=int, default=10, help="Print deep baseline progress every N epochs. Use 0 to disable.")
    parser.add_argument("--alpha", type=float, default=float(cfg["experiment"]["alpha"]))
    parser.add_argument(
        "--calibrator-mode",
        choices=["none", "stats_ridge", "identity_ridge"],
        default=cfg["experiment"].get("calibrator_mode", "identity_ridge"),
    )
    parser.add_argument("--ridge-alpha", type=float, default=float(cfg.get("calibrator", {}).get("ridge_alpha", 30.0)))
    parser.add_argument("--svr-c", type=float, default=10.0)
    parser.add_argument("--svr-epsilon", type=float, default=0.1)
    parser.add_argument("--svr-kernel", choices=["rbf", "linear", "poly"], default="rbf")
    parser.add_argument("--ppg-dir", type=Path, default=resolve_path(cfg["data"]["ppg_dir"]))
    parser.add_argument("--hr-dir", "--prv-dir", dest="hr_dir", type=Path, default=resolve_path(cfg["data"]["prv_dir"]))
    parser.add_argument("--prv-report", type=Path, default=resolve_path(cfg["data"]["prv_report"]))
    parser.add_argument("--limit-folds", type=int, default=0, help="Optional smoke-test limit. Full experiments use 0.")
    args = parser.parse_args()

    args.modalities = normalize_baseline_modality(args.modalities)
    if args.experiment is None:
        suffix = "" if args.stats or args.model in TABULAR_BASELINES else "_no_stats"
        args.experiment = f"{args.model}_{args.modalities}{suffix}"
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if args.hidden_size % args.transformer_heads != 0:
        raise ValueError("--hidden-size must be divisible by --transformer-heads")
    if args.mlp_steps < 1:
        raise ValueError("--mlp-steps must be at least 1")
    if args.gru_max_steps < 1:
        raise ValueError("--gru-max-steps must be at least 1")
    args.config_values = cfg
    return args


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


def select_stats(data: PFDMData, idx: np.ndarray, modalities: str) -> np.ndarray:
    stats = data.stats[idx].astype(np.float32)
    ppg_stats = stats[:, :8]
    hr_stats = stats[:, 8:16]
    row_and_emotion = stats[:, 16:]
    if modalities == "rppg":
        return np.concatenate([ppg_stats, row_and_emotion], axis=1)
    if modalities == "hr":
        return np.concatenate([hr_stats, row_and_emotion], axis=1)
    return stats


def build_calibrator_features(data: PFDMData, idx: np.ndarray, mode: str, modalities: str) -> np.ndarray:
    if mode == "none":
        return np.empty((len(idx), 0), dtype=np.float32)
    stats = select_stats(data, idx, modalities)
    if mode == "stats_ridge":
        return stats
    if mode != "identity_ridge":
        raise ValueError(f"unsupported calibrator mode: {mode}")

    max_row = int(data.sample_id.max()) + 1
    row_identity = np.zeros((len(idx), max_row), dtype=np.float32)
    row_identity[np.arange(len(idx)), data.sample_id[idx].astype(np.int64)] = 1.0
    emotion_identity = np.zeros((len(idx), len(EMOTION_NAMES)), dtype=np.float32)
    emotion_identity[np.arange(len(idx)), data.emotion[idx].astype(np.int64)] = 1.0
    return np.concatenate([stats, row_identity, emotion_identity], axis=1)


def fit_calibrator(data: PFDMData, train_idx: np.ndarray, mode: str, modalities: str, ridge_alpha: float) -> FoldCalibrator:
    if mode == "none":
        return FoldCalibrator(mode=mode)
    features = build_calibrator_features(data, train_idx, mode, modalities)
    scaler = FeatureStandardizer().fit(features)
    model = Ridge(alpha=ridge_alpha)
    model.fit(scaler.transform(features), data.stress[train_idx])
    return FoldCalibrator(mode=mode, model=model, scaler=scaler)


def calibrator_predict(calibrator: FoldCalibrator, data: PFDMData, idx: np.ndarray, modalities: str, fallback: np.ndarray) -> np.ndarray:
    if calibrator.mode == "none":
        return fallback.astype(np.float32)
    if calibrator.model is None or calibrator.scaler is None:
        raise RuntimeError("ridge calibrator was not fitted")
    features = build_calibrator_features(data, idx, calibrator.mode, modalities)
    return calibrator.model.predict(calibrator.scaler.transform(features)).astype(np.float32)


def prepare_fold_arrays(data: PFDMData, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, modalities: str):
    ppg_scaler = Standardizer().fit(data.ppg[train_idx])
    hr_scaler = Standardizer().fit(data.prv[train_idx])
    stats_scaler = FeatureStandardizer().fit(select_stats(data, train_idx, modalities))
    arrays = {}
    for name, idx in {"train": train_idx, "val": val_idx, "test": test_idx}.items():
        arrays[name] = {
            "ppg": ppg_scaler.transform(data.ppg[idx]),
            "prv": hr_scaler.transform(data.prv[idx]),
            "stats": stats_scaler.transform(select_stats(data, idx, modalities)),
        }
    return arrays


def make_loader(data: PFDMData, idx: np.ndarray, arrays: Dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(PFDMDataset(data, idx, arrays["ppg"], arrays["prv"], arrays["stats"]), batch_size=batch_size, shuffle=shuffle)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def add_input_noise(batch: Dict[str, torch.Tensor], modalities: str, noise_std: float) -> Dict[str, torch.Tensor]:
    if noise_std <= 0:
        return batch
    noisy = dict(batch)
    if modalities in {"rppg", "both"}:
        noisy["ppg"] = noisy["ppg"] + torch.randn_like(noisy["ppg"]) * noise_std
    if modalities in {"hr", "both"}:
        noisy["prv"] = noisy["prv"] + torch.randn_like(noisy["prv"]) * noise_std
    return noisy


def stress_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=1.0)


def predict_deep(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    losses: List[float] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = model(batch)
            loss = stress_loss(output["stress"], batch["stress"])
            preds.append(output["stress"].detach().cpu().numpy())
            targets.append(batch["stress"].detach().cpu().numpy())
            losses.append(float(loss.detach().cpu()))
    return np.concatenate(preds), np.concatenate(targets), float(np.mean(losses))


def build_deep_model(args: argparse.Namespace, stats_size: int) -> BaselineRegressor:
    return BaselineRegressor(
        model_name=args.model,
        modalities=args.modalities,
        stats_size=stats_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        gru_max_steps=args.gru_max_steps,
        mlp_steps=args.mlp_steps,
        use_stats=args.stats,
    )


def prediction_rows(
    data: PFDMData,
    idx: np.ndarray,
    split: str,
    target: np.ndarray,
    model_pred: np.ndarray,
    cal_pred: np.ndarray,
    final_pred: np.ndarray,
    args: argparse.Namespace,
) -> List[Dict[str, float | int | str]]:
    rows = []
    for local_i, sample_idx in enumerate(idx):
        rows.append(
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
                "baseline_model": args.model,
                "modalities": args.modalities,
            }
        )
    return rows


def evaluate_splits(
    data: PFDMData,
    split_indices_by_name: Dict[str, np.ndarray],
    model_predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    calibrator: FoldCalibrator,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, float | int | str]], Dict[str, Dict[str, float]]]:
    result_rows: List[Dict[str, float | int | str]] = []
    metrics: Dict[str, Dict[str, float]] = {}
    for split, idx in split_indices_by_name.items():
        model_pred = model_predictions[split]
        target = targets[split]
        cal_pred = calibrator_predict(calibrator, data, idx, args.modalities, model_pred)
        final_pred = (1.0 - args.alpha) * model_pred + args.alpha * cal_pred
        metrics[split] = metrics_np(target, final_pred)
        result_rows.extend(prediction_rows(data, idx, split, target, model_pred, cal_pred, final_pred, args))
    return result_rows, metrics


def make_tabular_model(args: argparse.Namespace):
    if args.model == "stats_ridge":
        return Ridge(alpha=args.ridge_alpha)
    if args.model == "stats_svr":
        return SVR(C=args.svr_c, epsilon=args.svr_epsilon, kernel=args.svr_kernel)
    raise ValueError(f"not a tabular baseline: {args.model}")


def run_tabular_fold(
    args: argparse.Namespace,
    data: PFDMData,
    fold: int,
    trainval_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
) -> Dict[str, float | int]:
    train_idx, val_idx = train_val_split(trainval_idx, data.stress, args.val_ratio, args.seed + fold)
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    scaler = FeatureStandardizer().fit(select_stats(data, train_idx, args.modalities))
    features = {
        "train": scaler.transform(select_stats(data, train_idx, args.modalities)),
        "val": scaler.transform(select_stats(data, val_idx, args.modalities)),
        "test": scaler.transform(select_stats(data, test_idx, args.modalities)),
    }
    indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    targets = {split: data.stress[idx].astype(np.float32) for split, idx in indices.items()}

    model = make_tabular_model(args)
    model.fit(features["train"], targets["train"])
    joblib.dump({"model": model, "scaler": scaler}, fold_dir / "model.joblib")
    model_predictions = {split: model.predict(features[split]).astype(np.float32) for split in indices}

    calibrator = fit_calibrator(data, train_idx, args.calibrator_mode, args.modalities, args.ridge_alpha)
    result_rows, metrics = evaluate_splits(data, indices, model_predictions, targets, calibrator, args)
    pd.DataFrame(result_rows).to_csv(fold_dir / "predictions.csv", index=False)

    row: Dict[str, float | int] = {"fold": fold}
    for split in ("train", "val", "test"):
        row[f"{split}_mae"] = metrics[split]["mae"]
        row[f"{split}_rmse"] = metrics[split]["rmse"]
    return row


def run_deep_fold(
    args: argparse.Namespace,
    data: PFDMData,
    fold: int,
    trainval_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, float | int]:
    train_idx, val_idx = train_val_split(trainval_idx, data.stress, args.val_ratio, args.seed + fold)
    arrays = prepare_fold_arrays(data, train_idx, val_idx, test_idx, args.modalities)
    train_loader = make_loader(data, train_idx, arrays["train"], args.batch_size, True)
    eval_loaders = {
        "train": make_loader(data, train_idx, arrays["train"], args.batch_size, False),
        "val": make_loader(data, val_idx, arrays["val"], args.batch_size, False),
        "test": make_loader(data, test_idx, arrays["test"], args.batch_size, False),
    }

    model = build_deep_model(args, stats_size=arrays["train"]["stats"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    raw_log: List[Dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            batch = add_input_noise(batch, args.modalities, args.input_noise_std)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = stress_loss(output["stress"], batch["stress"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_pred, train_target, train_loss_value = predict_deep(model, eval_loaders["train"], device)
        val_pred, val_target, val_loss_value = predict_deep(model, eval_loaders["val"], device)
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
    split_indices_by_name = {"train": train_idx, "val": val_idx, "test": test_idx}
    model_predictions: Dict[str, np.ndarray] = {}
    targets: Dict[str, np.ndarray] = {}
    for split, loader in eval_loaders.items():
        model_pred, target, _ = predict_deep(model, loader, device)
        model_predictions[split] = model_pred
        targets[split] = target

    calibrator = fit_calibrator(data, train_idx, args.calibrator_mode, args.modalities, args.ridge_alpha)
    result_rows, metrics = evaluate_splits(data, split_indices_by_name, model_predictions, targets, calibrator, args)
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


def append_summary_table(lines: List[str], title: str, summary: pd.DataFrame) -> None:
    lines.extend(
        [
            title,
            "",
            "| Split | MAE Mean | MAE Std | RMSE Mean | RMSE Std |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.mae_mean:.2f} | {row.mae_std:.2f} | "
            f"{row.rmse_mean:.2f} | {row.rmse_std:.2f} |"
        )
    lines.append("")


def write_readable_summary(
    output_dir: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
) -> None:
    lines = [
        "# Baseline Experiment Summary",
        "",
        f"Experiment: `{args.experiment}`",
        f"Baseline model: `{args.model}`",
        f"Modalities: `{args.modalities}`",
        f"Stats side features: `{args.stats if args.model in DEEP_BASELINES else True}`",
        f"Alpha: `{args.alpha:.3f}`",
        f"Calibrator: `{args.calibrator_mode}`",
        f"Epochs: `{args.epochs if args.model in DEEP_BASELINES else 'n/a'}`",
        "",
    ]
    append_summary_table(lines, "## Metrics Summary", summary)
    lines.extend(
        [
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
    print(f"Resolved device: {device}", flush=True)

    data = load_pfdm_data(args.ppg_dir.resolve(), args.hr_dir.resolve(), args.emotion, args.prv_report.resolve())
    fold_pairs = split_indices(data.stress, args.split, args.folds, args.seed)
    if args.limit_folds > 0:
        fold_pairs = fold_pairs[: args.limit_folds]

    rows = []
    for fold, (trainval_idx, test_idx) in enumerate(fold_pairs, start=1):
        print(
            f"[fold {fold}] model={args.model} modalities={args.modalities} "
            f"trainval={len(trainval_idx)} test={len(test_idx)}",
            flush=True,
        )
        if args.model in TABULAR_BASELINES:
            rows.append(run_tabular_fold(args, data, fold, trainval_idx, test_idx, output_dir))
        else:
            rows.append(run_deep_fold(args, data, fold, trainval_idx, test_idx, output_dir, device))

    by_fold = pd.DataFrame(rows)
    summary = summarize_by_fold(by_fold)
    by_fold.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    write_readable_summary(output_dir, args, summary, by_fold)

    config = vars(args).copy()
    config.pop("config_values", None)
    config["device_resolved"] = str(device)
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    (output_dir / "config_resolved.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_summary_for_console(summary))


if __name__ == "__main__":
    main()
