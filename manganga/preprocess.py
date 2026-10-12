"""Preprocesamiento de imágenes de alas.

Replica el pipeline original del upstream:

    foto -> recorte -> letterbox a 400x400 (resize_square.reshape) -> CLAHE -> U-Net

Ver ``wing_detector.detectFromPaths`` (letterbox + CLAHE) y ``unet-test.py``.
"""

from __future__ import annotations

import cv2
import numpy as np

from manganga import IMAGE_SIZE


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convierte BGR/RGB a escala de grises si hace falta."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def letterbox(img: np.ndarray, size: int = IMAGE_SIZE, border_value: int = 0) -> np.ndarray:
    """Escala preservando el aspecto y rellena con negro.

    Equivalente a ``resize_square.reshape`` del upstream.
    """
    img = to_gray(img)
    h, w = img.shape[:2]
    ratio = float(size) / max(h, w)
    new_w, new_h = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    delta_w, delta_h = size - new_w, size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_value
    )


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """CLAHE con los mismos parámetros que el upstream."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(img)


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return img


def prepare_input(
    img: np.ndarray | str,
    size: int = IMAGE_SIZE,
    use_clahe: bool = True,
    use_letterbox: bool = True,
) -> np.ndarray:
    """Imagen de ala -> tensor de entrada float32 ``(size, size)`` en [0, 1].

    Parámetros
    ----------
    img:
        Array BGR/gris o ruta a la imagen.
    size:
        Lado de la imagen de salida (400 por defecto, igual que el upstream).
    use_clahe / use_letterbox:
        Activar o desactivar cada paso (útil para tests y ablaciones).
    """
    if isinstance(img, str):
        img = load_gray(img)
    else:
        img = to_gray(img)

    if use_letterbox:
        img = letterbox(img, size=size)
    else:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    if use_clahe:
        img = apply_clahe(img)

    return (img.astype(np.float32) / 255.0)
