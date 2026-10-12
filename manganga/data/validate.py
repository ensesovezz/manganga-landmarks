"""Validación del ground truth del dataset de máscaras.

Las máscaras del upstream son discos dibujados sobre los 19 landmarks manuales
(``cria_mascara.py``), pero el orden de los discos no viene codificado en la
imagen. Este módulo recupera los centros, los ordena contra la referencia y
mide la disparidad Procrustes para detectar máscaras mal construidas u ordenadas.

Se usa para filtrar el set de entrenamiento (ver Fase 1 del plan).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from manganga import N_LANDMARKS
from manganga.landmarks import detect_dots, load_reference, order_shape
from manganga.landmarks.blobs import CANDIDATE_MARGIN, expected_radius_for
from manganga.landmarks.procrustes import DEFAULT_MAX_DISPARITY


@dataclass
class MaskValidation:
    name: str
    n_dots: int
    status: str
    disparity: float
    mirrored: bool
    ok: bool
    note: str


def validate_mask(
    mask: np.ndarray,
    name: str,
    reference: np.ndarray,
    max_disparity: float = DEFAULT_MAX_DISPARITY,
) -> MaskValidation:
    """Extrae, ordena y evalúa una máscara."""
    dots = detect_dots(
        mask,
        expected_radius=expected_radius_for(mask.shape[0]),
        max_candidates=N_LANDMARKS + CANDIDATE_MARGIN,
    )
    if len(dots.points) < 19:
        return MaskValidation(
            name=name,
            n_dots=len(dots.points),
            status=dots.status,
            disparity=float("inf"),
            mirrored=False,
            ok=False,
            note="; ".join(dots.notes) or dots.status,
        )

    shape = order_shape(dots.points, reference, max_disparity=max_disparity)
    return MaskValidation(
        name=name,
        n_dots=len(dots.points),
        status=dots.status,
        disparity=shape.disparity,
        mirrored=shape.mirrored,
        ok=shape.ok and not shape.mirrored,
        note=shape.note,
    )


def validate_dataset(
    mask_dir: str | Path,
    limit: int | None = None,
    max_disparity: float = DEFAULT_MAX_DISPARITY,
    reference_path: str | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Valida todas las máscaras de un directorio y devuelve un DataFrame."""
    mask_dir = Path(mask_dir)
    reference = load_reference(reference_path)
    paths = sorted(p for p in mask_dir.iterdir() if p.suffix.lower() in (".bmp", ".jpg", ".jpeg", ".png"))
    if limit is not None:
        paths = paths[:limit]

    iterator = paths
    if progress:
        from tqdm import tqdm

        iterator = tqdm(paths, desc=f"validando {mask_dir.name}")

    rows = []
    for path in iterator:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            rows.append(
                MaskValidation(str(path.name), 0, "unreadable", float("inf"), False, False, "no se pudo leer")
            )
            continue
        rows.append(asdict(validate_mask(mask, path.name, reference, max_disparity)))

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    """Resumen legible del resultado de la validación."""
    total = len(df)
    if total == 0:
        return {"total": 0}

    finite = df[np.isfinite(df["disparity"])]
    return {
        "total": int(total),
        "ok": int(df["ok"].sum()),
        "pct_ok": round(100 * df["ok"].mean(), 2),
        "sin_19_puntos": int((df["n_dots"] != 19).sum()),
        "espejadas": int(df["mirrored"].sum()),
        "disparidad_media": round(float(finite["disparity"].mean()), 4) if len(finite) else None,
        "disparidad_p95": round(float(finite["disparity"].quantile(0.95)), 4) if len(finite) else None,
        "disparidad_p99": round(float(finite["disparity"].quantile(0.99)), 4) if len(finite) else None,
        "disparidad_max": round(float(finite["disparity"].max()), 4) if len(finite) else None,
    }


def load_excluded(filter_dir: str | Path | None, mask_dir_name: str) -> set[str]:
    """Nombres de máscaras a excluir según el manifiesto de ``validate-data``.

    ``validate-data`` escribe ``validation_<nombre-del-directorio>.csv`` en el
    directorio de salida; acá se leen las filas con ``ok == False`` para no
    entrenar con máscaras rotas (discos faltantes, ruido, etc.).
    """
    if filter_dir is None:
        return set()

    path = Path(filter_dir) / f"validation_{mask_dir_name}.csv"
    if not path.exists():
        return set()

    df = pd.read_csv(path)
    if "ok" not in df.columns or "name" not in df.columns:
        return set()
    return set(df.loc[~df["ok"].astype(bool), "name"].astype(str))


def dump_worst(
    df: pd.DataFrame,
    mask_dir: str | Path,
    out_dir: str | Path,
    n: int = 12,
    reference_path: str | None = None,
) -> list[Path]:
    """Guarda overlays de las máscaras con mayor disparidad para inspección visual."""
    from manganga.io import save_overlay

    mask_dir, out_dir = Path(mask_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = load_reference(reference_path)

    worst = df[np.isfinite(df["disparity"])].nlargest(n, "disparity")
    written = []
    for _, row in worst.iterrows():
        mask = cv2.imread(str(mask_dir / row["name"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        dots = detect_dots(mask)
        points = dots.points
        if len(points) == 19:
            shape = order_shape(points, reference)
            points = shape.points
        out_path = out_dir / f"worst_{row['name']}.png"
        save_overlay(out_path, mask, points)
        written.append(out_path)
    return written
