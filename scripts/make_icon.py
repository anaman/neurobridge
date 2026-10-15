#!/usr/bin/env python3
"""Draw the NeuroBridge application icon.

Procedural so there is no binary asset to keep in sync and no licence question:
a dark rounded square with the single frontal EEG channel's signature — an
alpha rhythm that grows when the eyes close — over a faint grid.

    .venv/bin/python scripts/make_icon.py        # writes assets/neurobridge-{256,128,64}.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"

BG_TOP = (18, 22, 30)
BG_BOTTOM = (28, 36, 48)
GRID = (44, 54, 70)
WAVE = (90, 226, 160)
WAVE_GLOW = (52, 150, 110)
ACCENT = (124, 196, 255)


def _rounded_mask(size: int, radius: int) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return m


def _background(size: int) -> Image.Image:
    """Vertical gradient, so the icon does not look flat on a light panel."""
    grad = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        f = y / max(1, size - 1)
        grad[y, :, :] = [int(BG_TOP[i] + f * (BG_BOTTOM[i] - BG_TOP[i])) for i in range(3)]
    img = Image.fromarray(grad, "RGB").convert("RGBA")
    img.putalpha(_rounded_mask(size, int(size * 0.22)))
    return img


def _wave_points(size: int, n: int = 600) -> list[tuple[int, int]]:
    """An EEG-looking trace: quiet at the left, alpha bursting at the right.

    That is literally the measurement the project rests on (alpha ×~2 on eye
    closure), so the icon is a picture of the validation, not decoration.
    Returns integer pixel coordinates, which is what ImageDraw wants.
    """
    t = np.linspace(0.0, 1.0, n)
    gate = 1.0 / (1.0 + np.exp(-(t - 0.52) * 14.0))            # eyes closing
    alpha = np.sin(2 * np.pi * 9.5 * t) * (0.28 + 1.05 * gate)
    drift = 0.10 * np.sin(2 * np.pi * 1.1 * t + 0.7)
    noise = 0.045 * np.sin(2 * np.pi * 37.0 * t + 1.3)
    y = alpha + drift + noise
    y = y / (np.max(np.abs(y)) + 1e-9)
    xs = (0.10 + 0.80 * t) * float(size)
    ys = (0.50 - 0.26 * y) * float(size)
    return [(int(round(float(a))), int(round(float(b)))) for a, b in zip(xs, ys)]


def make(size: int) -> Image.Image:
    img = _background(size)
    d = ImageDraw.Draw(img, "RGBA")
    s = size / 256.0

    # faint grid: this is a recording, not a logo
    step = max(4, int(32 * s))
    for gx in range(step, size, step):
        d.line([(gx, int(size * 0.10)), (gx, int(size * 0.90))], fill=GRID + (90,), width=max(1, int(s)))
    for gy in range(step, size, step):
        d.line([(int(size * 0.08), gy), (int(size * 0.92), gy)], fill=GRID + (90,), width=max(1, int(s)))

    pts = _wave_points(size)
    # glow pass, then the trace itself
    d.line(pts, fill=WAVE_GLOW + (110,), width=max(3, int(7 * s)), joint="curve")
    d.line(pts, fill=WAVE + (255,), width=max(2, int(3 * s)), joint="curve")

    # a baseline marker where the eyes close, matching the gate in _wave_points
    x0 = int(size * 0.52)
    d.line([(x0, int(size * 0.16)), (x0, int(size * 0.84))], fill=ACCENT + (120,),
           width=max(1, int(2 * s)))
    r = max(2, int(4 * s))
    d.ellipse((x0 - r, int(size * 0.16) - r, x0 + r, int(size * 0.16) + r), fill=ACCENT + (220,))
    return img


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for size in (256, 128, 64, 48):
        p = OUT / f"neurobridge-{size}.png"
        make(size).save(p, "PNG")
        print(f"wrote {p}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
