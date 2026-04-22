"""Training and evaluation helpers for M1c calibration models."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import M1CDataset
from .features import FeatureBuilder
from .metrics import classification_metrics
from .model import CalibrationNet, CalibrationNetConfig
from .schema import M1CRow, ORDER_LABELS


@dataclass(frozen=True)
class CalibrationTrainConfig:
    hidden_dims: tuple[int, ...] = (64, 32)
    latent_dim: int = 16
    dropout: float = 0.10
    epochs: int = 120
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    min_delta: float = 1e-4
    seed: int = 0
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden_dims"] = list(self.hidden_dims)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationTrainConfig":
        return cls(
            hidden_dims=tuple(int(value) for value in payload.get("hidden_dims", (64, 32))),
            latent_dim=int(payload.get("latent_dim", 16)),
            dropout=float(payload.get("dropout", 0.10)),
            epochs=int(payload.get("epochs", 120)),
            batch_size=int(payload.get("batch_size", 32)),
            lr=float(payload.get("lr", 1e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            patience=int(payload.get("patience", 20)),
            min_delta=float(payload.get("min_delta", 1e-4)),
            seed=int(payload.get("seed", 0)),
            device=str(payload.get("device", "cpu")),
        )


@dataclass
class CalibrationBundle:
    model: CalibrationNet
    feature_builder: FeatureBuilder
    model_config: CalibrationNetConfig
    train_config: CalibrationTrainConfig
    history: list[dict[str, Any]]
    best_epoch: int
    label_names: tuple[str, ...] = ORDER_LABELS


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device_from_model(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _run_epoch(
    model: CalibrationNet,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_count = 0
    y_true_idx: list[int] = []
    y_pred_idx: list[int] = []

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits, _latent = model(features)
        loss = criterion(logits, targets)

        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        batch_size = int(features.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        y_true_idx.extend(int(idx) for idx in targets.detach().cpu().tolist())
        y_pred_idx.extend(int(idx) for idx in logits.detach().cpu().argmax(dim=-1).tolist())

    y_true = [ORDER_LABELS[idx] for idx in y_true_idx]
    y_pred = [ORDER_LABELS[idx] for idx in y_pred_idx]
    metrics = classification_metrics(y_true, y_pred, labels=ORDER_LABELS)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def train_calibration_model(
    train_rows: Sequence[M1CRow],
    val_rows: Sequence[M1CRow] | None = None,
    *,
    config: CalibrationTrainConfig | None = None,
) -> CalibrationBundle:
    config = config or CalibrationTrainConfig()
    if not train_rows:
        raise ValueError("train_calibration_model requires at least one training row")

    _set_seed(config.seed)
    device = torch.device(config.device)

    feature_builder = FeatureBuilder.fit(train_rows)
    train_dataset = M1CDataset(train_rows, feature_builder)
    validation_rows = list(val_rows) if val_rows else list(train_rows)
    validation_dataset = M1CDataset(validation_rows, feature_builder)

    model_config = CalibrationNetConfig(
        input_dim=feature_builder.output_dim,
        hidden_dims=tuple(config.hidden_dims),
        latent_dim=config.latent_dim,
        output_dim=len(ORDER_LABELS),
        dropout=config.dropout,
    )
    model = CalibrationNet(model_config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)

    history: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    stalled = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        validation_metrics = _run_epoch(model, validation_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "val_loss": float(validation_metrics["loss"]),
                "val_accuracy": float(validation_metrics["accuracy"]),
                "val_macro_f1": float(validation_metrics["macro_f1"]),
            }
        )

        val_loss = float(validation_metrics["loss"])
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stalled = 0
        else:
            stalled += 1
            if config.patience > 0 and stalled >= config.patience:
                break

    model.load_state_dict(best_state)
    return CalibrationBundle(
        model=model,
        feature_builder=feature_builder,
        model_config=model_config,
        train_config=config,
        history=history,
        best_epoch=best_epoch or len(history),
    )


def predict_labels(
    bundle: CalibrationBundle,
    rows: Sequence[M1CRow],
    *,
    batch_size: int | None = None,
) -> list[str]:
    if not rows:
        return []

    dataset = M1CDataset(rows, bundle.feature_builder)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or bundle.train_config.batch_size,
        shuffle=False,
    )
    device = _device_from_model(bundle.model)
    bundle.model.eval()

    predictions: list[str] = []
    with torch.no_grad():
        for features, _targets in loader:
            features = features.to(device)
            logits, _latent = bundle.model(features)
            indices = logits.argmax(dim=-1).detach().cpu().tolist()
            predictions.extend(ORDER_LABELS[int(index)] for index in indices)
    return predictions


def evaluate_bundle(
    bundle: CalibrationBundle,
    rows: Sequence[M1CRow],
    *,
    batch_size: int | None = None,
) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "correct": 0,
            "confusion_matrix": [[0 for _ in ORDER_LABELS] for _ in ORDER_LABELS],
            "per_label": {
                label: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
                for label in ORDER_LABELS
            },
            "loss": 0.0,
        }

    dataset = M1CDataset(rows, bundle.feature_builder)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or bundle.train_config.batch_size,
        shuffle=False,
    )
    device = _device_from_model(bundle.model)
    criterion = nn.CrossEntropyLoss()
    metrics = _run_epoch(bundle.model, loader, criterion, device)
    metrics["predictions"] = predict_labels(bundle, rows, batch_size=batch_size)
    return metrics


def save_checkpoint(bundle: CalibrationBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": bundle.model.state_dict(),
        "feature_builder": bundle.feature_builder.to_dict(),
        "model_config": bundle.model_config.to_dict(),
        "train_config": bundle.train_config.to_dict(),
        "history": bundle.history,
        "best_epoch": bundle.best_epoch,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> CalibrationBundle:
    payload = torch.load(Path(path), map_location=map_location)
    feature_builder = FeatureBuilder.from_dict(payload["feature_builder"])
    model_config = CalibrationNetConfig.from_dict(payload["model_config"])
    train_config = CalibrationTrainConfig.from_dict(payload["train_config"])
    model = CalibrationNet(model_config)
    model.load_state_dict(payload["model_state"])
    model.to(map_location)
    return CalibrationBundle(
        model=model,
        feature_builder=feature_builder,
        model_config=model_config,
        train_config=train_config,
        history=list(payload.get("history", [])),
        best_epoch=int(payload.get("best_epoch", 0)),
    )
