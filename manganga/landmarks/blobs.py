"""Detección de los puntos (centros de los discos/heatmaps) en la salida de la U-Net.

Mejora sobre ``BlobDetection.py`` del upstream:

- No descarta imágenes en silencio: devuelve estado y motivo.
- Probé varios umbrales y operaciones morfológicas (fallbacks).
- Usa componentes conectados (centroides exactos) en vez de ``SimpleBlobDetector``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from manganga import N_LANDMARKS

# Radio esperado de los discos con los que se generaron las máscaras (cria_mascara.py, size=4).
EXPECTED_RADIUS = 4.0

STATUS_OK = "ok"
STATUS_INCOMPLETE = "incomplete"
STATUS_EXTRA = "extra"


@dataclass
class DotResult:
    """Resultado de la detección de puntos en una imagen."""

    points: np.ndarray  # (N, 2) float, coordenadas (x, y) en píxeles
    status: str
    n_expected: int = N_LANDMARKS
    threshold: float = 0.5
    morph: str = "none"
    area_range: tuple[int, int] = (0, 0)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and len(self.points) == self.n_expected


def _components(binary: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    """Centroides de los componentes conectados dentro del rango de área."""
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    points = []
    for i in range(1, n):  # 0 es el fondo
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            points.append(centroids[i])
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def _morph(img: np.ndarray, kind: str) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    if kind == "open":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    if kind == "erode":
        return cv2.erode(img, kernel, iterations=1)
    if kind == "dilate":
        return cv2.dilate(img, kernel, iterations=1)
    return img


def detect_dots(
    prob: np.ndarray,
    n_expected: int = N_LANDMARKS,
    expected_radius: float = EXPECTED_RADIUS,
    thresholds: tuple[float, ...] = (0.5, 0.35, 0.65, 0.25, 0.75),
    morphs: tuple[str, ...] = ("none", "open", "erode", "dilate"),
) -> DotResult:
    """Extrae los centros de los discos predichos por la U-Net.

    ``prob``: mapa de probabilidad float en [0, 1] o uint8 en [0, 255].
    Estrategia: se prueban combinaciones de umbral y morfología, se prefiere la
    primera que devuelve exactamente ``n_expected`` puntos; si ninguna lo logra,
    se devuelve la más cercana con estado ``incomplete``/``extra``.
    """
    prob = np.asarray(prob)
    if prob.dtype == np.uint8:
        prob = prob.astype(np.float32) / 255.0
    else:
        prob = prob.astype(np.float32)

    expected_area = np.pi * expected_radius**2
    min_area = max(2, int(round(expected_area * 0.05)))
    max_area = int(round(expected_area * 8))
    area_range = (min_area, max_area)

    attempts: list[tuple[int, float, str, np.ndarray]] = []
    for thresh in thresholds:
        binary = (prob >= thresh).astype(np.uint8)
        for morph in morphs:
            candidate = _morph(binary, morph)
            points = _components(candidate, min_area, max_area)
            attempts.append((abs(len(points) - n_expected), thresh, morph, points))
            if len(points) == n_expected:
                return DotResult(
                    points=points,
                    status=STATUS_OK,
                    n_expected=n_expected,
                    threshold=thresh,
                    morph=morph,
                    area_range=area_range,
                )

    _, thresh, morph, points = min(attempts, key=lambda a: a[0])
    status = STATUS_INCOMPLETE if len(points) < n_expected else STATUS_EXTRA
    return DotResult(
        points=points,
        status=status,
        n_expected=n_expected,
        threshold=thresh,
        morph=morph,
        area_range=area_range,
        notes=[f"se esperaban {n_expected} puntos, se encontraron {len(points)}"],
    )
