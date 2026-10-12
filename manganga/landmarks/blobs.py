"""Detección de los puntos (centros de los discos/heatmaps) en la salida de la U-Net.

Mejora sobre ``BlobDetection.py`` del upstream:

- No descarta imágenes en silencio: devuelve estado, motivo y diagnóstico.
- Parte componentes fusionados (dos discos que se tocan son un solo componente
  conectado en la máscara): distancia-transform + picos, que es el motivo
  principal de "faltan puntos" en el ground truth del upstream.
- Filtra ruido por área y selecciona los ``n_expected`` candidatos más parecidos
  a un disco (consistencia de área) en lugar de tomar lo primero que aparezca.
- Prueba varios umbrales y morfologías, y se queda con la mejor candidata.

Calibración sobre ``masks/`` del upstream (discos de radio 4):

- área típica de un disco: 44-53 px (π·4² ≈ 50)
- ruido presente en las máscaras: componentes de 1-38 px
- pares de discos fusionados: hasta 97 px (≈ 2 discos)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from manganga import N_LANDMARKS

# Radio esperado de los discos con los que se generaron las máscaras (cria_mascara.py, size=4).
EXPECTED_RADIUS = 4.0

STATUS_OK = "ok"
STATUS_INCOMPLETE = "incomplete"
STATUS_EXTRA = "extra"

# Fracciones del área esperada de un disco.
MIN_AREA_RATIO = 0.35  # por debajo: ruido
SINGLE_MAX_RATIO = 1.6  # por encima: probablemente varios discos fusionados
HARD_MAX_RATIO = 12.0  # por encima: basura

# Cuántos candidatos extra se pasan al ordenamiento guiado por forma
# (``max_candidates = N_LANDMARKS + CANDIDATE_MARGIN``).
CANDIDATE_MARGIN = 8


@dataclass
class DotResult:
    """Resultado de la detección de puntos en una imagen."""

    points: np.ndarray  # (N, 2) float, coordenadas (x, y) en píxeles
    status: str
    n_expected: int = N_LANDMARKS
    threshold: float = 0.5
    morph: str = "none"
    areas: np.ndarray | None = None  # área de cada candidato seleccionado
    n_merged: int = 0  # componentes que hubo que partir
    n_candidates: int = 0  # candidatos crudos antes de seleccionar los 19
    area_range: tuple[int, int] = (0, 0)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and len(self.points) == self.n_expected

    def as_row(self) -> dict:
        return {
            "n_points": len(self.points),
            "status": self.status,
            "threshold": self.threshold,
            "morph": self.morph,
            "n_merged": self.n_merged,
            "n_candidates": self.n_candidates,
            "area_media": round(float(np.mean(self.areas)), 1) if self.areas is not None and len(self.areas) else "",
        }


@dataclass
class _Candidate:
    x: float
    y: float
    area: float
    n_merged: int = 1

    def deviation(self, expected_area: float) -> float:
        """Qué tan lejos está el área de la de un disco típico (log-ratio absoluto)."""
        return abs(math.log(self.area / expected_area))


def _split_component(component: np.ndarray, expected_area: float) -> list[tuple[int, int]]:
    """Centros de los discos fusionados en un componente grande.

    Usa la transformada de distancia: cada disco tiene un máximo local en su
    centro; se suprimen los máximos ya usados con un radio proporcional al pico.
    """
    total_area = float(component.sum())
    n_discs = max(1, int(round(total_area / expected_area)))

    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    work = distance.copy()
    peaks: list[tuple[int, int]] = []
    for _ in range(n_discs):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val < 1.0:
            break
        peaks.append((int(max_loc[0]), int(max_loc[1])))
        cv2.circle(work, max_loc, max(1, int(round(max_val * 1.2))), 0, -1)
    return peaks


def _candidates(
    binary: np.ndarray,
    expected_area: float,
    min_area: float,
    single_max: float,
    hard_max: float,
) -> tuple[list[_Candidate], int]:
    """Candidatos a punto a partir de los componentes conectados."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    candidates: list[_Candidate] = []
    n_merged = 0

    for i in range(1, n):  # 0 es el fondo
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > hard_max:
            continue
        if area <= single_max:
            candidates.append(_Candidate(float(centroids[i][0]), float(centroids[i][1]), area))
            continue

        peaks = _split_component((labels == i).astype(np.uint8), expected_area)
        if len(peaks) <= 1:
            candidates.append(_Candidate(float(centroids[i][0]), float(centroids[i][1]), area))
            continue

        n_merged += 1
        each = area / len(peaks)
        for px, py in peaks:
            candidates.append(_Candidate(float(px), float(py), each, n_merged=len(peaks)))

    return candidates, n_merged


def _select(
    candidates: list[_Candidate], keep: int, expected_area: float, n_expected: int
) -> tuple[list[_Candidate], float]:
    """Ordena los candidatos por parecido a un disco y devuelve hasta ``keep``.

    El puntaje (menor = mejor) se calcula siempre sobre los ``n_expected``
    mejores candidatos (para que no dependa de cuántos se devuelvan) y penaliza:

    - no llegar a ``n_expected`` puntos (el objetivo real),
    - descartar candidatos que sí parecían discos,
    - el desvío de área de los elegidos.
    """
    ranked = sorted(candidates, key=lambda c: c.deviation(expected_area))
    kept = ranked[:keep]
    core = ranked[:n_expected]

    score = 10.0 * max(0, n_expected - len(candidates))
    score += sum(c.deviation(expected_area) for c in core)
    score += 10.0 * sum(1 for c in ranked[keep:] if c.deviation(expected_area) < 0.15)
    return kept, score


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
    max_candidates: int | None = None,
) -> DotResult:
    """Extrae los centros de los discos predichos por la U-Net.

    ``prob``: mapa de probabilidad float en [0, 1] o uint8 en [0, 255].

    ``max_candidates``: si se indica, devuelve hasta esa cantidad de candidatos
    (ordenados por parecido a un disco) en lugar de recortar a ``n_expected``.
    Se usa para que ``order_shape`` pueda elegir el mejor subconjunto de 19
    guiándose por la forma de referencia.
    """
    prob = np.asarray(prob)
    prob = prob.astype(np.float32) / 255.0 if prob.dtype == np.uint8 else prob.astype(np.float32)

    expected_area = math.pi * expected_radius**2
    min_area = max(4.0, MIN_AREA_RATIO * expected_area)
    single_max = SINGLE_MAX_RATIO * expected_area
    hard_max = HARD_MAX_RATIO * expected_area
    keep = n_expected if max_candidates is None else max(max_candidates, n_expected)

    attempts: list[tuple[float, DotResult]] = []
    for threshold in thresholds:
        binary = (prob >= threshold).astype(np.uint8)
        for morph in morphs:
            candidates, n_merged = _candidates(
                _morph(binary, morph), expected_area, min_area, single_max, hard_max
            )
            kept, score = _select(candidates, keep, expected_area, n_expected)
            points = (
                np.array([[c.x, c.y] for c in kept], dtype=float)
                if kept
                else np.empty((0, 2), dtype=float)
            )
            status = STATUS_OK if len(kept) >= n_expected else STATUS_INCOMPLETE
            result = DotResult(
                points=points,
                status=status,
                n_expected=n_expected,
                threshold=threshold,
                morph=morph,
                areas=np.array([c.area for c in kept]) if kept else None,
                n_merged=n_merged,
                n_candidates=len(candidates),
                area_range=(int(min_area), int(single_max)),
            )
            attempts.append((score, result))

            clean = (
                n_expected <= len(kept) <= n_expected + 1
                and n_merged == 0
                and score < 0.1 * len(kept)
            )
            if clean:
                return result

    best_score, best = min(attempts, key=lambda a: a[0])
    if best.status != STATUS_OK:
        best.notes.append(
            f"se esperaban {n_expected} puntos, se encontraron {len(best.points)} "
            f"(umbral {best.threshold}, morfología {best.morph})"
        )
    return best
