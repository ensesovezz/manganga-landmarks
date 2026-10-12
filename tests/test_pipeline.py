"""Tests del pipeline (sin modelo entrenado: usan máscaras sintéticas)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from manganga import N_LANDMARKS
from manganga.io import draw_overlay, read_tps, write_tps
from manganga.landmarks import detect_dots, load_reference, order_shape
from manganga.landmarks.procrustes import disparity
from manganga.preprocess import letterbox, prepare_input


def synthetic_mask(points: np.ndarray, size: int = 400, radius: int = 4) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    for x, y in points:
        cv2.circle(mask, (int(round(x)), int(round(y))), radius, 255, -1)
    return mask


def similarity(points: np.ndarray, angle: float, scale: float, shift: np.ndarray) -> np.ndarray:
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (scale * (rot @ points.T).T) + shift


def test_reference_has_19_landmarks():
    reference = load_reference()
    assert reference.shape == (N_LANDMARKS, 2)
    assert np.isfinite(reference).all()


def test_letterbox_preserves_aspect():
    img = np.zeros((100, 300), dtype=np.uint8)
    out = letterbox(img, size=400)
    assert out.shape == (400, 400)
    # 300x100 -> 400x133 + padding
    assert out[:133].any() or out[133:].any()


def test_prepare_input_range():
    img = (np.random.default_rng(0).random((250, 180)) * 255).astype(np.uint8)
    out = prepare_input(img, size=256)
    assert out.shape == (256, 256)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_detect_dots_finds_19_discs():
    reference = load_reference()
    points = similarity(reference, 0.4, 90.0, np.array([200.0, 210.0]))
    mask = synthetic_mask(points)
    result = detect_dots(mask)
    assert result.ok, result.notes
    assert len(result.points) == N_LANDMARKS


def test_detect_dots_reports_incomplete():
    mask = synthetic_mask(np.array([[50.0, 50.0], [100.0, 100.0]]))
    result = detect_dots(mask)
    assert not result.ok
    assert result.status == "incomplete"
    assert result.notes


def test_order_shape_recovers_order():
    reference = load_reference()
    rng = np.random.default_rng(0)
    points = similarity(reference, 0.9, 1.4, np.array([-3.0, 7.0]))
    shuffled = points[rng.permutation(N_LANDMARKS)]

    shape = order_shape(shuffled, reference)
    assert shape.ok, shape.note
    assert not shape.mirrored
    # el orden recuperado debe reproducir la forma original
    assert np.allclose(shape.points, points, atol=1e-6)


def test_order_shape_detects_mirror():
    reference = load_reference()
    mirrored = reference.copy()
    mirrored[:, 0] *= -1
    shape = order_shape(mirrored, reference)
    assert shape.ok
    assert shape.mirrored


def test_order_shape_rejects_bad_shape():
    reference = load_reference()
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.35, size=(N_LANDMARKS, 2))
    shape = order_shape(noise, reference, max_disparity=0.02)
    assert not shape.ok
    assert shape.note


def test_disparity_zero_for_identical_shape():
    reference = load_reference()
    assert disparity(reference, reference) < 1e-9


def test_tps_roundtrip(tmp_path):
    reference = load_reference() * 100 + 200
    path = tmp_path / "sample.tps"
    write_tps(path, reference, image_id="ala_01.jpg")
    points, image_id = read_tps(path)
    assert points.shape == (N_LANDMARKS, 2)
    assert np.allclose(points, reference, atol=1e-3)
    assert image_id == "ala_01.jpg"


def test_draw_overlay_returns_bgr():
    reference = load_reference() * 100 + 200
    canvas = draw_overlay(np.zeros((400, 400), dtype=np.uint8), reference)
    assert canvas.shape == (400, 400, 3)
    assert canvas.sum() > 0


def test_crop_finds_synthetic_wing():
    from manganga.crop import crop_wing

    img = np.zeros((600, 800), dtype=np.uint8)
    cv2.ellipse(img, (400, 300), (200, 80), 20, 0, 360, 200, -1)
    result = crop_wing(img, size=400)
    assert result.ok, result.note
    assert result.image is not None
    assert result.image.shape == (400, 400)
    assert result.area_ratio > 0.05


def test_crop_fails_on_blank_image():
    from manganga.crop import crop_wing

    result = crop_wing(np.zeros((200, 200), dtype=np.uint8))
    assert not result.ok


@pytest.mark.parametrize("size", [256, 400])
def test_unet_forward_shape(size):
    torch = pytest.importorskip("torch")
    from manganga.models import build_model

    model = build_model(base=4, depth=4)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, size, size))
    assert out.shape == (1, 1, size, size)
