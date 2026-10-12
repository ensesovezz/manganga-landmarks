"""Dataset de alas + máscaras para entrenar la U-Net.

``train/`` (o ``test/``) + ``masks/`` (o ``test_masks/``) tienen los mismos
nombres de archivo, así que el emparejamiento es directo.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from manganga import IMAGE_SIZE
from manganga.preprocess import apply_clahe, letterbox

IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def list_pairs(image_dir: str | Path, mask_dir: str | Path) -> list[tuple[Path, Path]]:
    """Devuelve los pares (imagen, máscara) con nombre coincidente."""
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    pairs = []
    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            # El upstream mezcla .bmp y .jpg: probar otras extensiones.
            candidates = [mask_dir / f"{img_path.stem}{ext}" for ext in IMAGE_EXTENSIONS]
            found = next((c for c in candidates if c.exists()), None)
            if found is None:
                continue
            mask_path = found
        pairs.append((img_path, mask_path))
    return pairs


def _augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Aumento geométrico y fotométrico pareado (la salida son discos: es invariante al orden)."""
    size = image.shape[0]
    angle = float(rng.uniform(0, 360))
    scale = float(rng.uniform(0.9, 1.1))
    tx, ty = rng.uniform(-15, 15, size=2)

    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    image = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(mask, matrix, (size, size), flags=cv2.INTER_NEAREST, borderValue=0)

    if rng.random() < 0.5:
        image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
    if rng.random() < 0.5:
        image, mask = image[::-1, :].copy(), mask[::-1, :].copy()

    if rng.random() < 0.5:
        alpha = float(rng.uniform(0.85, 1.15))
        beta = float(rng.uniform(-15, 15))
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        noise = rng.normal(0, 6, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image, mask


class WingMaskDataset(Dataset):
    """Pares ala/máscara normalizados a ``size`` x ``size``."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        size: int = IMAGE_SIZE,
        augment: bool = False,
        use_clahe: bool = True,
        pairs: list[tuple[Path, Path]] | None = None,
        seed: int = 0,
    ):
        self.pairs = pairs if pairs is not None else list_pairs(image_dir, mask_dir)
        self.size = size
        self.augment = augment
        self.use_clahe = use_clahe
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        img_path, mask_path = self.pairs[index]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"No se pudo leer la máscara: {mask_path}")

        image = letterbox(image, size=self.size)
        mask = letterbox(mask, size=self.size)
        if self.use_clahe:
            image = apply_clahe(image)

        binary_mask = (mask > 127).astype(np.uint8) * 255
        return image, binary_mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._load(index)
        if self.augment:
            image, mask = _augment(image, mask, self.rng)

        x = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        y = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
        return x, y
