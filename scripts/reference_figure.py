#!/usr/bin/env python3
"""Genera figuras de referencia de los 19 landmarks.

    python scripts/reference_figure.py -o docs/img

Produce:

- ``esquema_19_puntos.png``: la forma media (referencia Procrustes) con el número
  de cada landmark. Sirve para digitalizar manualmente en el mismo orden.
- ``ejemplo_sobre_ala.png``: los 19 puntos ordenados sobre un ala del dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from manganga.io import draw_overlay  # noqa: E402
from manganga.landmarks import detect_dots, load_reference, order_shape  # noqa: E402
from manganga.landmarks.blobs import CANDIDATE_MARGIN  # noqa: E402
from manganga import N_LANDMARKS  # noqa: E402


def plot_reference(reference: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(reference[:, 0], reference[:, 1], s=55, c="#c8102e", zorder=2)
    for index, (x, y) in enumerate(reference, start=1):
        ax.annotate(
            str(index),
            (x, y),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
            fontweight="bold",
            color="#1f2937",
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()  # coordenadas de imagen (y crece hacia abajo)
    ax.set_title("Esquema de los 19 landmarks (forma media de referencia)")
    ax.set_xlabel("coordenada x normalizada")
    ax.set_ylabel("coordenada y normalizada")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_example(mask_dir: Path, image_dir: Path, out_path: Path, reference: np.ndarray) -> str | None:
    reference = load_reference()
    for mask_path in sorted(mask_dir.iterdir()):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        dots = detect_dots(mask, max_candidates=N_LANDMARKS + CANDIDATE_MARGIN)
        if len(dots.points) < N_LANDMARKS:
            continue
        shape = order_shape(dots.points, reference)
        if not shape.ok:
            continue
        image_path = image_dir / mask_path.name
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        cv2.imwrite(str(out_path), draw_overlay(image, shape.points))
        return f"{mask_path.name} (disparidad {shape.disparity:.4f})"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", type=Path, default=Path("docs/img"))
    parser.add_argument("--mask-dir", type=Path, default=Path("masks"))
    parser.add_argument("--image-dir", type=Path, default=Path("train"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    reference = load_reference()

    plot_reference(reference, args.out / "esquema_19_puntos.png")
    print(f"escrito {args.out / 'esquema_19_puntos.png'}")

    used = plot_example(args.mask_dir, args.image_dir, args.out / "ejemplo_sobre_ala.png", reference)
    if used:
        print(f"escrito {args.out / 'ejemplo_sobre_ala.png'} usando {used}")
    else:
        print("no se encontró ninguna máscara válida para el ejemplo")


if __name__ == "__main__":
    main()
