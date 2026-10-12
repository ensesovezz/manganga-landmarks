"""Alineación Procrustes y ordenamiento de landmarks.

Reemplaza las heurísticas ``list_dot_check`` / ``adjust_element`` del upstream
(swaps a mano, frágiles con fotos nuevas) por una asignación óptima contra la
forma de referencia (algoritmo húngaro + refinamiento iterativo).

El orden de los 19 puntos queda definido por ``assets/reference_19.txt``
(media Procrustes del upstream, ``resultados/deteccoes/main-reference.txt``).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from manganga import N_LANDMARKS

REFERENCE_ASSET = "reference_19.txt"
# Umbral de disparidad (RMS por punto, en unidades de la referencia) para marcar
# una forma como sospechosa.
# Calibrado sobre el ground truth del upstream (3.000 máscaras de masks/):
# media 0.0053 · p95 0.0071 · p99 0.0083 · máximo 0.0099.
# 0.03 deja ~3x de margen sobre el peor caso del ground truth y sigue detectando
# ordenamientos malos (que suelen dar > 0.05).
DEFAULT_MAX_DISPARITY = 0.03


def load_reference(path: str | None = None) -> np.ndarray:
    """Carga la forma de referencia (19, 2).

    Si ``path`` es None usa el asset empaquetado ``assets/reference_19.txt``.
    """
    if path is None:
        asset = resources.files("manganga").joinpath("assets", REFERENCE_ASSET)
        text = asset.read_text(encoding="utf-8")
    else:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()

    values = []
    for token in text.split():
        try:
            values.append(float(token))
        except ValueError:
            continue

    arr = np.asarray(values, dtype=float)
    if arr.size != N_LANDMARKS * 2:
        raise ValueError(
            f"La referencia debe tener {N_LANDMARKS * 2} valores, tiene {arr.size}"
        )
    return arr.reshape(N_LANDMARKS, 2)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Transformación de similitud óptima (Umeyama) tal que ``dst ~= s*R@src + t``."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = len(src)

    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst

    cov = dst_c.T @ src_c / n
    u, d, vt = np.linalg.svd(cov)

    s_mat = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        # Sin reflexiones: la mejor rotación propia (Umeyama, 1991).
        s_mat[-1, -1] = -1

    rot = u @ s_mat @ vt
    var_src = float((src_c**2).sum() / n)
    scale = float((d * np.diag(s_mat)).sum() / var_src) if var_src > 0 else 1.0
    trans = mu_dst - scale * (rot @ mu_src)
    return rot, scale, trans


def apply_similarity(
    src: np.ndarray, rot: np.ndarray, scale: float, trans: np.ndarray
) -> np.ndarray:
    return (scale * (rot @ np.asarray(src, dtype=float).T).T) + trans


def align_to(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Devuelve ``src`` alineado (similitud) sobre ``dst``."""
    rot, scale, trans = umeyama(src, dst)
    return apply_similarity(src, rot, scale, trans)


def disparity(src: np.ndarray, dst: np.ndarray) -> float:
    """Distancia RMS por punto entre ``src`` alineado y ``dst``."""
    aligned = align_to(src, dst)
    return float(np.sqrt(((aligned - dst) ** 2).sum() / len(dst)))


@dataclass
class OrderedShape:
    """Resultado del ordenamiento de una forma."""

    points: np.ndarray  # (19, 2) en el orden de la referencia
    disparity: float
    mirrored: bool  # True si hubo que espejar (indica foto de ala izquierda)
    n_points: int  # puntos de entrada
    ok: bool
    dropped: int = 0  # puntos descartados (cuando había más de 19)
    note: str = ""


def _pca_candidates(points: np.ndarray, reference: np.ndarray) -> list[np.ndarray]:
    """Alineaciones iniciales usando los ejes principales (2 rotaciones propias)."""

    def axes(pts: np.ndarray) -> np.ndarray:
        centered = pts - pts.mean(axis=0)
        cov = centered.T @ centered / len(pts)
        _, vecs = np.linalg.eigh(cov)
        v1 = vecs[:, -1]
        v2 = np.array([-v1[1], v1[0]])
        return np.stack([v1, v2], axis=1)  # columnas = ejes

    w_pts, w_ref = axes(points), axes(reference)
    var_pts = float(((points - points.mean(0)) ** 2).sum() / len(points))
    var_ref = float(((reference - reference.mean(0)) ** 2).sum() / len(reference))
    scale = float(np.sqrt(var_ref / var_pts)) if var_pts > 0 else 1.0

    candidates = []
    for signs in ((1, 1), (-1, -1)):
        rot = w_ref @ np.diag(signs) @ w_pts.T
        trans = reference.mean(axis=0) - scale * (rot @ points.mean(axis=0))
        candidates.append(apply_similarity(points, rot, scale, trans))
    return candidates


def _assign_once(aligned: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cost = cdist(aligned, reference)
    rows, cols = linear_sum_assignment(cost)
    return rows, cols


def order_shape(
    points: np.ndarray,
    reference: np.ndarray,
    max_iter: int = 5,
    allow_mirror: bool = True,
    max_disparity: float = DEFAULT_MAX_DISPARITY,
    select_best: bool = True,
) -> OrderedShape:
    """Ordena los puntos según la referencia.

    - Prueba la forma tal cual y su espejo (una foto de ala izquierda llega
      invertida) y se queda con la de menor disparidad.
    - Si hay **más** puntos que landmarks (típico: motas de ruido en la máscara),
      elige el mejor subconjunto de 19 con asignación húngara rectangular contra
      la referencia (selección guiada por la forma) e informa cuántos descartó.
    - Si hay menos, no puede ordenar y devuelve ``ok=False``.

    Para formas espejadas las coordenadas devueltas están en el marco espejado
    (el eje x se invirtió); ``mirrored=True`` lo indica.
    """
    pts = np.asarray(points, dtype=float)
    n_points = len(pts)
    if n_points < len(reference):
        return OrderedShape(
            points=pts,
            disparity=float("inf"),
            mirrored=False,
            n_points=n_points,
            ok=False,
            note=f"se esperaban {len(reference)} puntos, hay {n_points}",
        )

    best: OrderedShape | None = None
    variants = [(pts, False)]
    if allow_mirror:
        mirrored = pts.copy()
        mirrored[:, 0] = -mirrored[:, 0]
        variants.append((mirrored, True))

    for variant, is_mirrored in variants:
        if len(variant) > len(reference) and select_best:
            ordered, dropped, disp = _select_and_order(variant, reference, max_iter)
        else:
            ordered = _order_variant(variant, reference, max_iter)
            dropped = 0
            disp = disparity(ordered, reference)

        if best is None or disp < best.disparity:
            best = OrderedShape(
                points=ordered,
                disparity=disp,
                mirrored=is_mirrored,
                n_points=n_points,
                ok=True,
                dropped=dropped,
            )

    assert best is not None
    best.ok = best.disparity <= max_disparity
    if not best.ok:
        best.note = (
            f"disparidad {best.disparity:.4f} > {max_disparity:.4f} "
            "(forma atípica o puntos mal detectados)"
        )
    elif best.mirrored:
        best.note = "la forma se alineó mejor espejada: ¿foto de ala izquierda?"
    elif best.dropped:
        best.note = f"se descartaron {best.dropped} puntos por no encajar en la forma"
    return best


def _select_and_order(
    points: np.ndarray, reference: np.ndarray, max_iter: int
) -> tuple[np.ndarray, int, float]:
    """Elige el mejor subconjunto de ``len(reference)`` puntos y lo ordena.

    Asignación húngara rectangular (N puntos -> 19 landmarks): en cada iteración
    se elige el subconjunto que mejor encaja con la referencia, se re-ajusta la
    similitud con ese subconjunto y se repite.
    """
    n_ref = len(reference)
    best_aligned = None
    best_cost = np.inf
    for candidate in _pca_candidates(points, reference):
        rows, cols = _assign_rect(candidate, reference)
        cost = float(cdist(candidate, reference)[rows, cols].sum())
        if cost < best_cost:
            best_aligned, best_cost = candidate, cost

    aligned = best_aligned
    ordered = np.empty((n_ref, 2), dtype=float)
    selected = np.arange(n_ref)
    for _ in range(max_iter):
        rows, cols = _assign_rect(aligned, reference)
        selected = rows
        ordered = np.empty((n_ref, 2), dtype=float)
        ordered[cols] = points[rows]
        rot, scale, trans = umeyama(ordered, reference)
        aligned = apply_similarity(points, rot, scale, trans)

    rows, cols = _assign_rect(aligned, reference)
    ordered = np.empty((n_ref, 2), dtype=float)
    ordered[cols] = points[rows]
    disp = float(np.sqrt(((aligned[rows] - reference) ** 2).sum() / n_ref))
    return ordered, len(points) - n_ref, disp


def _assign_rect(aligned: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Asignación húngara rectangular: todos los landmarks a puntos distintos."""
    rows, cols = linear_sum_assignment(cdist(aligned, reference))
    return rows, cols


def _order_variant(points: np.ndarray, reference: np.ndarray, max_iter: int) -> np.ndarray:
    """Refina la asignación puntos <-> referencia de forma iterativa."""
    # Inicialización por ejes principales (rompe la ambigüedad de rotación).
    best_init, best_cost = None, np.inf
    for candidate in _pca_candidates(points, reference):
        rows, cols = _assign_once(candidate, reference)
        cost = float(cdist(candidate, reference)[rows, cols].sum())
        if cost < best_cost:
            best_init, best_cost = candidate, cost

    aligned = best_init
    ordered = points
    for _ in range(max_iter):
        rows, cols = _assign_once(aligned, reference)
        ordered = np.empty_like(points)
        ordered[cols] = points[rows]
        rot, scale, trans = umeyama(ordered, reference)
        aligned = apply_similarity(points, rot, scale, trans)

    return ordered
