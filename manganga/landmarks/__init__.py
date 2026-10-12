"""Detección y ordenamiento de landmarks."""

from manganga.landmarks.blobs import DotResult, detect_dots
from manganga.landmarks.procrustes import (
    OrderedShape,
    align_to,
    disparity,
    load_reference,
    order_shape,
)

__all__ = [
    "DotResult",
    "detect_dots",
    "OrderedShape",
    "align_to",
    "disparity",
    "load_reference",
    "order_shape",
]
