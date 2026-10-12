"""Recorte automático del ala en fotos crudas (sin el SSD de TensorFlow del upstream).

Segmentación clásica: umbral + contorno de mayor área + ``minAreaRect`` +
rotación a posición upright + letterbox a 400x400.

Es un reemplazo heurístico del detector de objetos original (cuyo modelo ``.pb``
nunca fue publicado). Para el flujo principal se asume una foto ya recortada;
``--crop`` es para fotos crudas de escáner/lupa.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from manganga import IMAGE_SIZE
from manganga.preprocess import letterbox, to_gray


@dataclass
class CropResult:
    image: np.ndarray | None
    ok: bool
    bbox: tuple[int, int, int, int] | None = None
    area_ratio: float = 0.0
    note: str = ""


def find_wing_contour(gray: np.ndarray, min_area_ratio: float = 0.02):
    """Contorno más probable del ala (mayor área con forma razonable)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    total = float(gray.shape[0] * gray.shape[1])
    best, best_area = None, 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area / total < min_area_ratio:
            continue
        if area > best_area:
            best, best_area = contour, area

    if best is None:
        return None, 0.0
    return best, best_area / total


def crop_wing(img: np.ndarray, size: int = IMAGE_SIZE, margin: int = 8) -> CropResult:
    """Recorta el ala y la devuelve upright, en ``size`` x ``size``."""
    gray = to_gray(img)
    contour, area_ratio = find_wing_contour(gray)
    if contour is None:
        return CropResult(None, False, note="no se encontró ningún contorno candidato")

    (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
    if w < 5 or h < 5:
        return CropResult(None, False, note="contorno demasiado pequeño")

    # Rotar para dejar el ala en horizontal (lado mayor en x).
    if h > w:
        w, h = h, w
        angle += 90
    w, h = w + 2 * margin, h + 2 * margin

    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR)

    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))
    x1 = int(round(cx + w / 2))
    y1 = int(round(cy + h / 2))
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(rotated.shape[1], x1), min(rotated.shape[0], y1)
    if x1c - x0c < 5 or y1c - y0c < 5:
        return CropResult(None, False, note="recorte fuera de la imagen")

    crop = rotated[y0c:y1c, x0c:x1c]
    # Relleno si el recorte quedó cortado por el borde.
    pad_x, pad_y = (w - crop.shape[1]) // 2, (h - crop.shape[0]) // 2
    if pad_x > 0 or pad_y > 0:
        crop = cv2.copyMakeBorder(
            crop,
            max(0, pad_y),
            max(0, pad_y),
            max(0, pad_x),
            max(0, pad_x),
            cv2.BORDER_CONSTANT,
            value=int(np.median(crop)) if crop.size else 0,
        )

    return CropResult(
        image=letterbox(crop, size=size),
        ok=True,
        bbox=(x0c, y0c, x1c, y1c),
        area_ratio=area_ratio,
    )
