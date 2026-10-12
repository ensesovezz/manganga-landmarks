"""Entrenamiento de la U-Net.

    manganga train --data-dir . --epochs 40 --size 400 --base 32

La métrica de éxito es la del paper: porcentaje de alas con los 19 puntos
detectados *y* ordenados correctamente (disparidad Procrustes dentro del umbral).
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from manganga import IMAGE_SIZE, N_LANDMARKS
from manganga.data.dataset import WingMaskDataset, list_pairs
from manganga.detect import predict_map
from manganga.landmarks import detect_dots, load_reference, order_shape
from manganga.landmarks.procrustes import DEFAULT_MAX_DISPARITY
from manganga.models import build_model, count_parameters
from manganga.preprocess import apply_clahe, letterbox


@dataclass
class TrainConfig:
    data_dir: Path = Path(".")
    train_images: str = "train"
    train_masks: str = "masks"
    val_images: str = "test"
    val_masks: str = "test_masks"
    out_dir: Path = Path("out/checkpoints")
    size: int = IMAGE_SIZE
    base: int = 32
    depth: int = 4
    epochs: int = 40
    batch_size: int = 8
    lr: float = 1e-3
    pos_weight: float = 50.0
    val_split: float = 0.05
    max_train_samples: int | None = None
    metric_every: int = 5
    metric_samples: int = 200
    workers: int = 0
    seed: int = 0
    max_disparity: float = DEFAULT_MAX_DISPARITY
    reference_path: str | None = None
    device: str = "cpu"


@dataclass
class TrainHistory:
    rows: list[dict] = field(default_factory=list)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_pairs(pairs, val_split: float, seed: int):
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    return shuffled[n_val:], shuffled[:n_val]


@torch.no_grad()
def evaluate_detection(
    model: nn.Module,
    pairs,
    reference: np.ndarray,
    size: int,
    device: str,
    max_samples: int = 200,
    max_disparity: float = DEFAULT_MAX_DISPARITY,
    use_clahe: bool = True,
) -> dict:
    """Métrica real: % de alas con 19 puntos detectados y ordenados bien."""
    import cv2

    model.eval()
    sample = pairs[:max_samples]
    ok_count, detected, disparities = 0, 0, []
    for img_path, _ in sample:
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = letterbox(image, size=size)
        if use_clahe:
            image = apply_clahe(image)
        prob = predict_map(model, image.astype(np.float32) / 255.0, device)
        dots = detect_dots(prob, n_expected=N_LANDMARKS)
        if not dots.ok:
            continue
        detected += 1
        shape = order_shape(dots.points, reference, max_disparity=max_disparity)
        if shape.ok:
            ok_count += 1
            disparities.append(shape.disparity)

    total = max(1, len(sample))
    return {
        "n_samples": len(sample),
        "pct_19_detectados": round(100 * detected / total, 2),
        "pct_ok_ordenados": round(100 * ok_count / total, 2),
        "disparidad_media": round(float(np.mean(disparities)), 4) if disparities else None,
    }


def train(config: TrainConfig) -> dict:
    set_seed(config.seed)
    device = torch.device(config.device)
    data_dir = Path(config.data_dir)

    all_pairs = list_pairs(data_dir / config.train_images, data_dir / config.train_masks)
    if not all_pairs:
        raise FileNotFoundError(
            f"No se encontraron pares en {data_dir / config.train_images} + {data_dir / config.train_masks}"
        )
    if config.max_train_samples:
        all_pairs = all_pairs[: config.max_train_samples]

    train_pairs, val_pairs = split_pairs(all_pairs, config.val_split, config.seed)

    external_val = list_pairs(data_dir / config.val_images, data_dir / config.val_masks)
    metric_pairs = external_val if external_val else val_pairs

    train_ds = WingMaskDataset(
        data_dir / config.train_images,
        data_dir / config.train_masks,
        size=config.size,
        augment=True,
        pairs=train_pairs,
        seed=config.seed,
    )
    val_ds = WingMaskDataset(
        data_dir / config.train_images,
        data_dir / config.train_masks,
        size=config.size,
        augment=False,
        pairs=val_pairs,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.workers, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.workers)

    model = build_model(base=config.base, depth=config.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.pos_weight, device=device))

    reference = load_reference(config.reference_path)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.out_dir / "training_log.csv"
    best_path = config.out_dir / "best.pt"

    print(
        f"U-Net base={config.base} depth={config.depth} ({count_parameters(model):,} parámetros) | "
        f"train={len(train_ds)} val={len(val_ds)} métrica sobre {len(metric_pairs)} | device={device}"
    )

    history = TrainHistory()
    best_score = -1.0
    best_metrics: dict = {}

    for epoch in range(1, config.epochs + 1):
        started = time.time()
        model.train()
        running = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(images)
        train_loss = running / max(1, len(train_ds))

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                val_running += criterion(model(images), masks).item() * len(images)
        val_loss = val_running / max(1, len(val_ds))
        scheduler.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "seconds": round(time.time() - started, 1),
        }

        run_metric = (epoch % config.metric_every == 0) or (epoch == config.epochs)
        if run_metric:
            metrics = evaluate_detection(
                model,
                metric_pairs,
                reference,
                size=config.size,
                device=config.device,
                max_samples=config.metric_samples,
                max_disparity=config.max_disparity,
            )
            row.update(metrics)
            score = float(metrics["pct_ok_ordenados"])
            if score >= best_score:
                best_score = score
                best_metrics = metrics
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": {
                            "base": config.base,
                            "depth": config.depth,
                            "size": config.size,
                            "clahe": True,
                        },
                        "epoch": epoch,
                        "metrics": metrics,
                    },
                    best_path,
                )
            print(
                f"epoch {epoch:3d} | train {train_loss:.5f} | val {val_loss:.5f} | "
                f"19 pts {metrics['pct_19_detectados']:6.2f}% | ok {metrics['pct_ok_ordenados']:6.2f}% | "
                f"{row['seconds']}s"
            )
        else:
            print(f"epoch {epoch:3d} | train {train_loss:.5f} | val {val_loss:.5f} | {row['seconds']}s")

        history.rows.append(row)
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history.rows[0].keys()))
            writer.writeheader()
            writer.writerows(history.rows)

    return {
        "best_checkpoint": str(best_path),
        "best_score_pct_ok_ordenados": best_score,
        "best_metrics": best_metrics,
        "epochs": config.epochs,
        "log": str(log_path),
    }
