"""Inferencia: foto de ala -> 19 landmarks ordenados.

Uso programático:

    from manganga.detect import load_checkpoint, DetectionRunner

    model, config = load_checkpoint("out/checkpoints/best.pt")
    runner = DetectionRunner(model, config)
    result = runner.detect("ala.jpg")
    print(result.points.shape)  # (19, 2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from manganga import IMAGE_SIZE, N_LANDMARKS
from manganga.crop import crop_wing
from manganga.landmarks import detect_dots, load_reference, order_shape
from manganga.landmarks.blobs import CANDIDATE_MARGIN, expected_radius_for
from manganga.landmarks.procrustes import DEFAULT_MAX_DISPARITY
from manganga.models import build_model
from manganga.preprocess import apply_clahe, letterbox, load_gray

DEFAULT_CHECKPOINT = Path("out/checkpoints/best.pt")


@dataclass
class Detection:
    """Resultado de detectar los landmarks en una imagen.

    ``points`` trae los 19 puntos ordenados siempre que se hayan podido ordenar,
    aunque la forma haya quedado marcada como dudosa (``status="flagged"``).
    ``ok`` es False en ese caso: los resultados dudosos se escriben en
    ``out/flagged/`` para poder inspeccionarlos.
    """

    image: str
    points: np.ndarray | None  # (19, 2) ordenados, o None si no se pudo
    status: str
    disparity: float = float("inf")
    mirrored: bool = False
    n_detected: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.points is not None
            and len(self.points) == N_LANDMARKS
            and self.status == "ok"
        )

    def as_row(self) -> dict:
        return {
            "image": self.image,
            "ok": self.ok,
            "n_detected": self.n_detected,
            "status": self.status,
            "disparity": round(self.disparity, 5) if np.isfinite(self.disparity) else "",
            "mirrored": self.mirrored,
            "notes": " | ".join(self.notes),
        }


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Carga un checkpoint de la U-Net guardado por ``manganga train``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint {path}. Entrená con `manganga train` "
            "(o descargá uno desde Colab/Kaggle, ver notebooks/train_unet.ipynb)."
        )
    checkpoint = torch.load(path, map_location=device)
    config = dict(checkpoint.get("config", {}))
    model = build_model(
        base=int(config.get("base", 32)),
        depth=int(config.get("depth", 4)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(device)
    return model, config


@torch.no_grad()
def predict_map(model: torch.nn.Module, image_input: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Mapa de probabilidad (H, W) en [0, 1] para una imagen ya preprocesada."""
    tensor = torch.from_numpy(image_input[None, None].astype(np.float32)).to(device)
    logits = model(tensor)
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


class DetectionRunner:
    """Encapsula modelo + referencia para detectar en una o varias imágenes."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict | None = None,
        reference_path: str | None = None,
        device: str = "cpu",
        max_disparity: float = DEFAULT_MAX_DISPARITY,
        use_clahe: bool | None = None,
    ):
        self.model = model
        self.config = config or {}
        self.device = device
        self.reference = load_reference(reference_path)
        self.max_disparity = max_disparity
        self.size = int(self.config.get("size", IMAGE_SIZE))
        self.use_clahe = bool(self.config.get("clahe", True)) if use_clahe is None else use_clahe

    def _preprocess(self, path: str | Path, crop: bool) -> tuple[np.ndarray | None, list[str]]:
        notes: list[str] = []
        image = load_gray(path)
        if crop:
            result = crop_wing(image, size=self.size)
            if not result.ok or result.image is None:
                return None, [f"recorte fallido: {result.note}"]
            image = result.image
            notes.append(f"recorte automático (área {result.area_ratio:.3f})")
        else:
            image = letterbox(image, size=self.size)
        if self.use_clahe:
            image = apply_clahe(image)
        return image.astype(np.float32) / 255.0, notes

    def detect(self, path: str | Path, crop: bool = False) -> Detection:
        path = Path(path)
        prepared, notes = self._preprocess(path, crop)
        if prepared is None:
            return Detection(image=path.name, points=None, status="crop_failed", notes=notes)

        prob = predict_map(self.model, prepared, device=self.device)
        dots = detect_dots(
            prob,
            n_expected=N_LANDMARKS,
            expected_radius=expected_radius_for(self.size),
            max_candidates=N_LANDMARKS + CANDIDATE_MARGIN,
        )

        if len(dots.points) < N_LANDMARKS:
            return Detection(
                image=path.name,
                points=None,
                status=dots.status,
                n_detected=len(dots.points),
                notes=notes + dots.notes,
            )

        shape = order_shape(dots.points, self.reference, max_disparity=self.max_disparity)
        result = Detection(
            image=path.name,
            points=shape.points if len(shape.points) == N_LANDMARKS else None,
            status="ok" if shape.ok else "flagged",
            disparity=shape.disparity,
            mirrored=shape.mirrored,
            n_detected=len(dots.points),
            notes=notes,
        )
        if shape.note:
            result.notes.append(shape.note)
        return result

    def detect_batch(self, paths: list[Path], crop: bool = False, progress: bool = True) -> list[Detection]:
        iterator = paths
        if progress:
            from tqdm import tqdm

            iterator = tqdm(paths, desc="detectando")
        return [self.detect(path, crop=crop) for path in iterator]


def collect_images(input_dir: str | Path) -> list[Path]:
    """Lista imágenes de un directorio (recursivo)."""
    input_dir = Path(input_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in extensions)
