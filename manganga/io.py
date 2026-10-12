"""Lectura/escritura de resultados: .tps, .csv y overlays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PALETTE = [
    (0, 0, 255),
    (0, 128, 255),
    (0, 200, 200),
    (0, 180, 0),
    (200, 120, 0),
    (255, 0, 0),
    (255, 0, 180),
    (128, 0, 255),
]


def write_tps(path: str | Path, points: np.ndarray, image_id: str | None = None) -> None:
    """Escribe un archivo TPS compatible con tpsDig/MorphoJ (LM=19)."""
    points = np.asarray(points, dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"LM={len(points)}\n")
        for x, y in points:
            fh.write(f"{x:.4f} {y:.4f}\n")
        if image_id:
            fh.write(f"IMAGE={image_id}\n")
            fh.write(f"ID={Path(image_id).stem}\n")
        fh.write("\n")


def read_tps(path: str | Path) -> tuple[np.ndarray, str | None]:
    """Lee un TPS de landmarks (primer espécimen)."""
    text = Path(path).read_text(encoding="utf-8").split()
    n_landmarks = 0
    coords: list[float] = []
    image_id: str | None = None
    i = 0
    while i < len(text):
        token = text[i]
        if token.upper().startswith("LM="):
            n_landmarks = int(token.split("=")[1])
            i += 1
            for _ in range(n_landmarks * 2):
                coords.append(float(text[i]))
                i += 1
            continue
        if token.upper().startswith("IMAGE="):
            image_id = token.split("=", 1)[1]
            i += 1
            continue
        if token.upper().startswith("ID="):
            # No pisar el IMAGE= (el ID suele ser el nombre sin extensión).
            if image_id is None:
                image_id = token.split("=", 1)[1]
            i += 1
            continue
        i += 1

    points = np.asarray(coords, dtype=float).reshape(-1, 2)
    return points, image_id


def write_csv(path: str | Path, points: np.ndarray, header: str | None = None) -> None:
    """Escribe coordenadas en CSV (x, y por fila)."""
    points = np.asarray(points, dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header or "x,y")
        fh.write("\n")
        for x, y in points:
            fh.write(f"{x:.4f},{y:.4f}\n")


def draw_overlay(
    image: np.ndarray,
    points: np.ndarray | None,
    radius: int = 7,
    numbers: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Dibuja los puntos numerados sobre la imagen (devuelve BGR)."""
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    if points is None or len(points) == 0:
        return canvas

    for index, (x, y) in enumerate(np.asarray(points, dtype=float)):
        color = PALETTE[index % len(PALETTE)]
        center = (int(round(x)), int(round(y)))
        cv2.circle(canvas, center, radius, color, thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, 1, color, -1, lineType=cv2.LINE_AA)
        if numbers:
            cv2.putText(
                canvas,
                str(index + 1),
                (center[0] + radius + 2, center[1] - radius - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas


def save_overlay(path: str | Path, image: np.ndarray, points: np.ndarray | None, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), draw_overlay(image, points, **kwargs))
