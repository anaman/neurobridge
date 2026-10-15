"""Procedurally drawn fallback images so every image test runs offline.

These are **fallbacks, not a substitute for real photographs.**  The visual
ERP tests assume real environmental scenes (an Openverse download); when the
machine is offline or the API refuses, a session must still run end to end, and
a *structured* picture (a horizon, a grid, a schematic face) is a far better
degradation than nothing at all.  Where an effect depends on real semantic
content (N400 picture-word matching, affective valence), the report must say
that the built-in set was used and the result should be treated as a smoke test.

Design rules
------------
* Every image is 1024x768 PNG, drawn at 2x and downsampled with LANCZOS so
  edges are antialiased without any hand-rolled supersampling.
* Categories differ in *composition and colour*, not in random noise, because
  the analyses compare categories: ``bright``/``dark`` differ in mean luminance,
  ``pleasant``/``unpleasant`` differ in colour temperature and clutter,
  ``natural`` has soft organic texture, ``manmade`` has straight edges and
  repeated structure, and ``arrow_left``/``arrow_right`` are mirror images that
  are distinguishable at a glance (T704 relies on this).
* The ``face`` category is a schematic but recognisable face: eyes, nose, mouth
  and a hair-line.  A visual N170 surrogate (T403) needs *a* face-shaped
  configuration, not a realistic one.
* An **unknown** category name never raises: it is hashed into a deterministic
  palette and pattern, so a typo in a spec still produces distinct, stable
  images (and the generated label makes the mistake visible in the report).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..config import IMAGE_DIR, ensure_dirs

#: Where the fallback set lives.
BUILTIN_DIR = IMAGE_DIR / "builtin"

#: The categories named in the INTERFACES contract, plus the extras the 64
#: specs actually reference.  Anything outside this set is still resolved
#: deterministically (see :func:`_generic`).
CATEGORIES = ["bright", "dark", "natural", "manmade", "face", "object",
              "pleasant", "neutral", "unpleasant", "standard", "deviant"]

#: Extra categories referenced by specs but not in the contract.
EXTRA_CATEGORIES = [
    "flash", "flicker", "flicker_8", "flicker_10", "flicker_12",
    "arrow_left", "arrow_right", "left", "right", "percept", "imagery",
    "easy", "hard", "match", "mismatch", "nature", "urban", "quiet", "loud",
    "faces", "objects", "scenes", "fixation", "blank",
]

#: Images per category.  Two is enough to avoid immediate repetition in a
#: 200-trial block without bloating the repo.
PER_CATEGORY = 2

SIZE = (1024, 768)
SUPERSAMPLE = 2

#: A small, deliberately legible palette per family.  Kept as RGB tuples.
SKIN = (232, 190, 160)
SKIN_SHADOW = (206, 160, 132)
HAIR = (68, 48, 40)

PALETTES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "natural": ((96, 140, 92), (168, 196, 150), (60, 96, 120), (214, 206, 168)),
    "manmade": ((120, 124, 132), (176, 180, 186), (78, 84, 92), (36, 40, 46)),
    "pleasant": ((250, 214, 132), (246, 168, 130), (140, 200, 170), (255, 240, 210)),
    "neutral": ((176, 176, 172), (200, 198, 192), (140, 142, 140), (220, 218, 212)),
    "unpleasant": ((84, 70, 62), (120, 96, 72), (52, 56, 64), (96, 88, 84)),
    "bright": ((255, 250, 236), (255, 236, 190), (236, 244, 255), (250, 226, 168)),
    "dark": ((22, 24, 32), (36, 34, 44), (14, 16, 22), (48, 44, 56)),
    "face": (SKIN, SKIN_SHADOW, HAIR, (250, 246, 240)),
    "object": ((196, 86, 74), (74, 118, 172), (232, 200, 96), (86, 96, 110)),
    "standard": ((70, 110, 170), (140, 176, 214), (32, 56, 96), (214, 228, 240)),
    "deviant": ((186, 78, 62), (232, 156, 108), (110, 42, 44), (248, 216, 182)),
    "arrow": ((28, 30, 36), (238, 238, 232), (196, 64, 56), (86, 92, 104)),
}


# ---------------------------------------------------------------- helpers
def _rng_for(*parts: Any) -> "random.Random":       # type: ignore[name-defined]
    """Deterministic RNG seeded from the category name (stable across runs)."""
    import random

    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return random.Random(int(hashlib.sha1(blob).hexdigest()[:12], 16))


def _palette(family: str) -> tuple[tuple[int, int, int], ...]:
    return PALETTES.get(family, PALETTES["neutral"])


def _generic_palette(name: str) -> tuple[tuple[int, int, int], ...]:
    """Hash an unknown category name into a stable, visually distinct palette."""
    rng = _rng_for("palette", name)
    base_hue = rng.randrange(0, 360)
    out = []
    for i in range(4):
        h = (base_hue + i * rng.choice([28, 44, 60, 96, 140])) % 360
        s = rng.uniform(0.25, 0.7)
        v = rng.uniform(0.25, 0.95)
        out.append(_hsv_to_rgb(h, s, v))
    return tuple(out)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    import colorsys

    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _new_image(bg: tuple[int, int, int]):
    from PIL import Image

    w, h = SIZE[0] * SUPERSAMPLE, SIZE[1] * SUPERSAMPLE
    return Image.new("RGB", (w, h), bg)


def _save(img, path: Path) -> Path:
    """Downsample to the contract's 1024x768 and write a PNG.

    Drawing happens at 2x and is reduced with LANCZOS here: that buys antialiased
    edges (PIL's draw primitives are aliased) for a strict 2x memory cost, and it
    keeps the on-disk size at ~50 KB per image instead of ~500 KB at draft
    quality.
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if img.size != SIZE:
        img = img.resize(SIZE, Image.LANCZOS)
    img.save(path, format="PNG", optimize=True)
    return path


def _noise_field(w: int, h: int, cells: int, seed: str):
    """Smooth low-frequency luminance field in [0, 1] (value noise)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(int(hashlib.sha1(seed.encode()).hexdigest()[:12], 16))
    grid = rng.random((cells, cells)).astype("float32")
    field = Image.fromarray((grid * 255).astype("uint8"), mode="L").resize(
        (w, h), Image.BICUBIC)
    arr = np.asarray(field, dtype="float32") / 255.0
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / max(1e-6, hi - lo)


def _blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    return (int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


def _apply_noise(img, field, pal: Sequence[tuple[int, int, int]], strength: float = 0.45):
    """Tint an image by a smooth luminance field; gives organic texture."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(img, dtype="float32")
    c0 = np.array(pal[0], dtype="float32")
    c1 = np.array(pal[1], dtype="float32")
    t = np.clip(field * strength, 0.0, 1.0)[..., None]
    tint = c0[None, None, :] + (c1 - c0)[None, None, :] * t
    mixed = arr * (1.0 - strength) + tint * strength
    return Image.fromarray(np.clip(mixed, 0, 255).astype("uint8"), mode="RGB")


def _draw_label(img, text: str, pal: Sequence[tuple[int, int, int]]) -> None:
    """Burn the generator name into the corner.

    This exists so an operator reading a figure can tell a real photograph from
    a fallback at a glance; without it, a silently-failed download would look
    like data.
    """
    from PIL import ImageDraw

    d = ImageDraw.Draw(img, "RGBA")
    try:
        from PIL import ImageFont

        font = ImageFont.load_default(size=max(14, 11 * SUPERSAMPLE))
    except Exception:
        font = None
    text = f"builtin:{text}"
    pad = 6 * SUPERSAMPLE
    try:
        box = d.textbbox((pad, pad), text, font=font)
    except Exception:
        box = (pad, pad, pad + 40 * len(text), pad + 16 * SUPERSAMPLE)
    # A dark plate behind the text: the label must stay legible on `bright`
    # (mean luminance 250) as well as on `dark`.
    d.rectangle([box[0] - pad, box[1] - pad // 2, box[2] + pad, box[3] + pad // 2],
                fill=(0, 0, 0, 150))
    d.text((pad, pad), text, fill=(255, 255, 255), font=font)


# ---------------------------------------------------------------- drawings
def _gradient(img, pal: Sequence[tuple[int, int, int]], rng, vertical: bool = True):
    import numpy as np

    w, h = img.size
    t = np.linspace(0.0, 1.0, h if vertical else w, dtype="float32")
    if rng.random() < 0.5:
        t = t[::-1]
    c0 = np.array(pal[0], dtype="float32")
    c1 = np.array(pal[2] if len(pal) > 2 else pal[1], dtype="float32")
    line = c0[None, :] + (c1 - c0)[None, :] * t[:, None]
    band = np.repeat(line[:, None, :], w, axis=1) if vertical else \
        np.repeat(line[None, :, :], h, axis=0)
    from PIL import Image

    return Image.fromarray(np.clip(band, 0, 255).astype("uint8"), mode="RGB")


def _natural(name: str, n: int):
    """Horizon + foliage-ish noise: a landscape silhouette, not a photo."""
    from PIL import ImageDraw

    pal = _palette("natural")
    rng = _rng_for(name, n)
    img = _gradient(_new_image(pal[2]), pal, rng, vertical=True)
    field = _noise_field(*img.size, cells=5 + rng.randrange(4), seed=f"{name}{n}")
    img = _apply_noise(img, field, (pal[1], pal[0]), strength=0.5)
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    horizon = int(h * (0.52 + 0.14 * rng.random()))
    # far hills
    for k in range(3):
        pts = [(0, h)]
        for x in range(0, w + 1, w // 24):
            y = horizon - int((0.06 + 0.05 * k) * h
                              * (0.5 + 0.5 * math.sin(x / w * (1.5 + k) + rng.random() * 6.28)))
            pts.append((x, y))
        pts.append((w, h))
        d.polygon(pts, fill=(pal[0][0] - 24 * k, pal[0][1] - 18 * k, pal[0][2] - 10 * k))
    # foliage clumps
    for _ in range(24):
        cx = int(rng.random() * w)
        cy = int(horizon + rng.random() * (h - horizon))
        r = int((0.03 + 0.06 * rng.random()) * w)
        d.ellipse([cx - r, cy - r, cx + r, cy + r],
                  fill=(pal[1][0], pal[1][1], pal[1][2], 70))
    d.rectangle([0, horizon, w, h], fill=(pal[0][0] // 2, pal[0][1] // 2, pal[0][2] // 2))
    d.line([0, horizon, w, horizon], fill=pal[3], width=max(2, 3 * SUPERSAMPLE))
    _draw_label(img, name if n == 0 else f"{name}{n}", pal)
    return img


def _manmade(name: str, n: int):
    """Grids, rectangles and straight edges: the visual opposite of natural."""
    from PIL import ImageDraw

    pal = _palette("manmade")
    rng = _rng_for(name, n)
    img = _gradient(_new_image(pal[3]), pal, rng, vertical=True)
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    horizon = int(h * 0.62)
    d.rectangle([0, horizon, w, h], fill=pal[3])
    # a skyline of rectangles with lit windows
    x = 0
    while x < w:
        bw = int(rng.uniform(0.06, 0.16) * w)
        bh = int(rng.uniform(0.18, 0.55) * h)
        top = horizon - bh
        d.rectangle([x, top, x + bw, horizon], fill=pal[0], outline=pal[2],
                    width=max(2, 2 * SUPERSAMPLE))
        win = max(6, bw // 6)
        for wy in range(top + win, horizon - win, win * 2):
            for wx in range(x + win, x + bw - win, win * 2):
                lit = rng.random() < 0.45
                d.rectangle([wx, wy, wx + win, wy + win],
                            fill=(250, 226, 150) if lit else pal[2])
        x += bw + int(rng.uniform(0.005, 0.03) * w)
    # perspective road/grid lines
    for k in range(10):
        xx = int((k / 9.0) * w)
        d.line([w // 2, horizon, xx, h], fill=(pal[1][0], pal[1][1], pal[1][2], 120),
               width=max(1, SUPERSAMPLE))
    for k in range(1, 7):
        yy = horizon + int((h - horizon) * (k / 7.0) ** 1.7)
        d.line([0, yy, w, yy], fill=(pal[1][0], pal[1][1], pal[1][2], 120),
               width=max(1, SUPERSAMPLE))
    _draw_label(img, name if n == 0 else f"{name}{n}", pal)
    return img


def _face(name: str, n: int):
    """A schematic but recognisable face: eyes, nose, mouth, hair-line."""
    from PIL import ImageDraw

    pal = _palette("face")
    rng = _rng_for(name, n)
    bg = _gradient(_new_image((236, 236, 240)), (pal[3], (200, 206, 216), (176, 186, 200)), rng)
    img = bg
    d = ImageDraw.Draw(bg, "RGBA")
    w, h = bg.size
    cx, cy = w // 2, int(h * 0.54)
    fw, fh = int(w * 0.30), int(h * 0.40)
    # hair-line: a rounded cap slightly larger than the skull
    d.ellipse([cx - fw - 10 * SUPERSAMPLE, cy - fh - 26 * SUPERSAMPLE,
               cx + fw + 10 * SUPERSAMPLE, cy + fh * 0.35], fill=HAIR)
    # head
    d.ellipse([cx - fw, cy - fh, cx + fw, cy + fh], fill=pal[0])
    d.ellipse([cx - fw + 8 * SUPERSAMPLE, cy - fh + 8 * SUPERSAMPLE,
               cx + fw - 8 * SUPERSAMPLE, cy + fh - 8 * SUPERSAMPLE], fill=pal[0])
    # brows
    for sx in (-1, 1):
        bx = cx + sx * int(fw * 0.46)
        d.line([bx - int(fw * 0.24), cy - int(fh * 0.24),
                bx + int(fw * 0.24), cy - int(fh * 0.30)],
               fill=HAIR, width=max(3, 5 * SUPERSAMPLE))
    # eyes with visible iris and pupil (a face needs fixatable features)
    for sx in (-1, 1):
        ex = cx + sx * int(fw * 0.46)
        ey = cy - int(fh * 0.10)
        er = int(fw * 0.17)
        d.ellipse([ex - er, ey - er * 0.72, ex + er, ey + er * 0.72], fill=(252, 252, 252))
        ir = int(er * 0.52)
        d.ellipse([ex - ir, ey - ir, ex + ir, ey + ir], fill=(78, 66, 58))
        pr = max(2, int(ir * 0.45))
        d.ellipse([ex - pr, ey - pr, ex + pr, ey + pr], fill=(16, 14, 14))
    # nose
    d.polygon([(cx, cy - int(fh * 0.06)), (cx - int(fw * 0.13), cy + int(fh * 0.20)),
               (cx + int(fw * 0.15), cy + int(fh * 0.20))], fill=pal[1])
    # mouth (slight smile)
    d.polygon([(cx - int(fw * 0.30), cy + int(fh * 0.42)),
               (cx - int(fw * 0.15), cy + int(fh * 0.53)),
               (cx + int(fw * 0.15), cy + int(fh * 0.53)),
               (cx + int(fw * 0.30), cy + int(fh * 0.42)),
               (cx + int(fw * 0.15), cy + int(fh * 0.58)),
               (cx - int(fw * 0.15), cy + int(fh * 0.58))],
              fill=(168, 78, 78))
    # neck + shoulders so it reads as a person, not a balloon
    d.rectangle([cx - int(fw * 0.30), cy + int(fh * 0.86),
                 cx + int(fw * 0.30), h], fill=pal[1])
    d.ellipse([cx - int(fw * 1.5), h - int(h * 0.16), cx + int(fw * 1.5), h + int(h * 0.5)],
              fill=_blend(pal[1], (255, 255, 255), 0.25))
    _draw_label(img, name if n == 0 else f"{name}{n}", pal)
    return img


def _object(name: str, n: int):
    """Non-face shapes: circles, squares, triangles with clear contours."""
    from PIL import ImageDraw

    pal = _palette("object")
    rng = _rng_for(name, n)
    img = _gradient(_new_image((238, 238, 232)), (pal[3], (210, 212, 208)), rng)
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    d.rectangle([0, int(h * 0.78), w, h], fill=(196, 194, 186))
    shapes = ["ellipse", "rect", "triangle", "ellipse", "rect"]
    rng.shuffle(shapes)
    for i, shape in enumerate(shapes[:3]):
        cx = int(w * (0.24 + 0.26 * i) + rng.uniform(-0.03, 0.03) * w)
        cy = int(h * (0.46 + rng.uniform(-0.08, 0.08) * h / h))
        r = int(w * (0.08 + 0.05 * rng.random()))
        col = pal[i % 3]
        if shape == "ellipse":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline=(40, 40, 44),
                      width=max(2, 3 * SUPERSAMPLE))
        elif shape == "rect":
            d.rectangle([cx - r, cy - int(r * 0.8), cx + r, cy + int(r * 0.8)],
                        fill=col, outline=(40, 40, 44), width=max(2, 3 * SUPERSAMPLE))
        else:
            d.polygon([(cx, cy - r), (cx - r, cy + int(r * 0.8)),
                       (cx + r, cy + int(r * 0.8))], fill=col, outline=(40, 40, 44))
    _draw_label(img, name if n == 0 else f"{name}{n}", pal)
    return img


def _bright_dark(name: str, n: int, dark: bool):
    """High vs low mean luminance, with a soft radial falloff."""
    import numpy as np
    from PIL import Image

    pal = _palette("dark" if dark else "bright")
    rng = _rng_for(name, n)
    w, h = SIZE[0] * SUPERSAMPLE, SIZE[1] * SUPERSAMPLE
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * rng.uniform(0.35, 0.65), h * rng.uniform(0.35, 0.65)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (0.75 * w)
    g = np.clip(1.0 - r, 0.0, 1.0) ** 1.4
    c0 = np.array(pal[3] if dark else pal[0], dtype="float32")
    c1 = np.array(pal[0] if dark else pal[3], dtype="float32")
    arr = c1[None, None, :] + (c0 - c1)[None, None, :] * g[..., None]
    if not dark:
        arr = np.clip(arr + 26.0, 0, 255)
    else:
        arr = np.clip(arr * 0.55, 0, 255)
    img = Image.fromarray(arr.astype("uint8"), mode="RGB")
    field = _noise_field(*img.size, cells=6, seed=f"{name}{n}")
    img = _apply_noise(img, field, (pal[0], pal[3] if dark else pal[1]), strength=0.22)
    _draw_label(img, name if n == 0 else f"{name}{n}", pal)
    return img


def _affective(name: str, n: int, tone: str):
    """Pleasant / neutral / unpleasant: colour temperature *and* composition.

    * pleasant  -- warm palette, open sky, low clutter, round shapes.
    * neutral   -- desaturated, symmetrical, static.
    * unpleasant -- cold/dark palette, dense angular clutter, diagonal stress.

    The three are describable in words (warm/open vs flat vs cold/crowded),
    which is what T408 needs from a valence manipulation; it does **not** claim
    to elicit a real emotional response the way a photograph does.
    """
    from PIL import ImageDraw

    pal = _palette(tone)
    rng = _rng_for(name, n)
    img = _gradient(_new_image(pal[0]), pal, rng, vertical=True)
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    if tone == "pleasant":
        # open horizon, big warm sun, few soft clouds
        d.ellipse([int(w * 0.62), int(h * 0.12), int(w * 0.86), int(h * 0.40)],
                  fill=(255, 240, 190))
        d.rectangle([0, int(h * 0.66), w, h], fill=(150, 194, 150))
        for _ in range(5):
            cx = int(rng.random() * w)
            cy = int(rng.uniform(0.12, 0.42) * h)
            rw, rh = int(rng.uniform(0.07, 0.14) * w), int(rng.uniform(0.03, 0.06) * h)
            d.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=(255, 252, 244, 150))
    elif tone == "neutral":
        # flat, symmetrical: a single centred rectangle on a plain field
        d.rectangle([int(w * 0.32), int(h * 0.28), int(w * 0.68), int(h * 0.72)],
                    fill=pal[1], outline=pal[2], width=max(2, 3 * SUPERSAMPLE))
        d.line([w // 2, 0, w // 2, h], fill=pal[3], width=max(1, SUPERSAMPLE))
        d.line([0, h // 2, w, h // 2], fill=pal[3], width=max(1, SUPERSAMPLE))
    else:
        # cold, dense, angular clutter with diagonal stress
        for _ in range(46):
            x0 = int(rng.random() * w)
            y0 = int(rng.random() * h)
            dx = int(rng.uniform(-0.12, 0.12) * w)
            dy = int(rng.uniform(-0.12, 0.12) * h)
            d.polygon([(x0, y0), (x0 + dx, y0 + int(dy * 0.3)),
                       (x0 + int(dx * 0.4), y0 + dy)],
                      fill=_blend(pal[rng.randrange(3)], (0, 0, 0), rng.uniform(0.0, 0.5)))
        for _ in range(9):
            d.line([0, int(rng.random() * h), w, int(rng.random() * h)],
                   fill=(220, 210, 200, 60), width=max(1, SUPERSAMPLE))
    field = _noise_field(*img.size, cells=7, seed=f"{name}{n}")
    img = _apply_noise(img, field, (pal[0], pal[2]), strength=0.18)
    _draw_label(img, name if n == 0 else f"{name}{n}", pal)
    return img


def _standard_deviant(name: str, n: int, deviant: bool):
    """Two abstract 'gabor-like' patterns: same layout, different orientation."""
    import numpy as np
    from PIL import Image

    pal = _palette("deviant" if deviant else "standard")
    rng = _rng_for(name, n)
    w, h = SIZE[0] * SUPERSAMPLE, SIZE[1] * SUPERSAMPLE
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    freq = 0.012 if not deviant else 0.030
    ang = math.radians(0.0 if not deviant else 45.0)
    phase = (xx * math.cos(ang) + yy * math.sin(ang)) * freq
    grat = 0.5 + 0.5 * np.sin(2 * np.pi * phase)
    env = np.exp(-(((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (2 * (0.32 * w) ** 2)))
    t = np.clip(grat * env + 0.35 * (1 - env), 0, 1)
    c0 = np.array(pal[3], dtype="float32")
    c1 = np.array(pal[2] if deviant else pal[0], dtype="float32")
    arr = c0[None, None, :] + (c1 - c0)[None, None, :] * t[..., None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="RGB")
    _draw_label(img, name if n == 0 else f"{name}{n}", pal)
    return img


def _arrow(name: str, n: int, left: bool):
    """Mirrored directional arrows: unmistakable at a glance (T704)."""
    from PIL import ImageDraw

    pal = _palette("arrow")
    rng = _rng_for(name, n)
    img = _gradient(_new_image(pal[1]), (pal[1], (222, 222, 216), pal[0]), rng)
    d = ImageDraw.Draw(img)
    w, h = img.size
    cy = h // 2
    shaft = int(w * 0.60)
    head = int(w * 0.16)
    thick = int(h * 0.13)
    if left:
        x_tip, x_tail = int(w * 0.10), int(w * 0.10) + head + shaft
    else:
        x_tip, x_tail = int(w * 0.90), int(w * 0.90) - head - shaft
    sgn = -1 if left else 1
    d.polygon([(x_tip, cy),
               (x_tip + sgn * head, cy - int(h * 0.20)),
               (x_tip + sgn * head, cy - thick // 2),
               (x_tail, cy - thick // 2),
               (x_tail, cy + thick // 2),
               (x_tip + sgn * head, cy + thick // 2),
               (x_tip + sgn * head, cy + int(h * 0.20))],
              fill=pal[2], outline=pal[0])
    _draw_label(img, name if n == 0 else f"{name}{n}", pal)
    return img


def _flash(name: str, n: int, dark: bool = False):
    """Full-field high-contrast frames for the visual latency test (T005).

    A uniform field is deliberate: the latency estimate is the onset of a
    luminance step, so any spatial structure would add decoding time and
    contaminate the measurement.
    """
    from PIL import ImageDraw

    pal = _palette("arrow")
    img = _new_image((0, 0, 0) if dark else (255, 255, 255))
    d = ImageDraw.Draw(img)
    w, h = img.size
    if n == 1:                                   # the inverted frame
        d.rectangle([0, 0, w, h], fill=(0, 0, 0) if not dark else (255, 255, 255))
    # a tiny central fixation mark so the eye has somewhere to be
    cx, cy = w // 2, h // 2
    s = 10 * SUPERSAMPLE
    col = (30, 30, 30) if not dark else (240, 240, 240)
    d.rectangle([cx - s, cy - s, cx + s, cy + s], outline=col, width=max(2, 2 * SUPERSAMPLE))
    return img


def _flicker(name: str, n: int):
    """Frame sets for the alpha-entrainment flicker (T103).

    ``n=0`` is the bright frame, ``n=1`` the dark frame: the runner alternates
    them at the requested frequency, so a two-image category *is* the stimulus.
    """
    return _flash(name, n, dark=bool(n % 2))


def _generic(name: str, n: int):
    """Deterministic composition for any category the contract does not list.

    The name is hashed into a palette and a pattern choice, so ``easy`` and
    ``hard`` (or a typo) still get visually distinct, stable images instead of
    an exception -- a missing image file is a failed session, an odd-looking one
    is a note in the report.
    """
    from PIL import ImageDraw

    pal = _generic_palette(name)
    rng = _rng_for("generic", name, n)
    pattern = rng.choice(["grid", "blobs", "stripes", "mosaic"])
    img = _gradient(_new_image(pal[0]), pal, rng, vertical=rng.random() < 0.5)
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    if pattern == "grid":
        step = int(w / rng.uniform(10, 26))
        for x in range(0, w, step):
            d.line([x, 0, x, h], fill=pal[2], width=max(1, SUPERSAMPLE))
        for y in range(0, h, step):
            d.line([0, y, w, y], fill=pal[2], width=max(1, SUPERSAMPLE))
    elif pattern == "blobs":
        for _ in range(30):
            cx, cy = int(rng.random() * w), int(rng.random() * h)
            r = int(rng.uniform(0.03, 0.16) * w)
            d.ellipse([cx - r, cy - r, cx + r, cy + r],
                      fill=_blend(pal[rng.randrange(len(pal))], pal[0], rng.uniform(0, 0.5)))
    elif pattern == "stripes":
        bw = int(w / rng.uniform(6, 18))
        for i, x in enumerate(range(0, w, bw)):
            d.rectangle([x, 0, x + bw // 2, h],
                        fill=pal[i % len(pal)])
    else:
        cols, rows = rng.randrange(4, 9), rng.randrange(3, 7)
        for r in range(rows):
            for c in range(cols):
                d.rectangle([c * w / cols, r * h / rows, (c + 1) * w / cols, (r + 1) * h / rows],
                            fill=_blend(pal[(r + c) % len(pal)], pal[1], rng.uniform(0, 0.6)))
    field = _noise_field(*img.size, cells=8, seed=f"{name}{n}")
    img = _apply_noise(img, field, pal, strength=0.15)
    _draw_label(img, f"{name}{'' if n == 0 else n}", pal)
    return img


def _render(name: str, n: int):
    """Dispatch a category name to its drawing routine (never raises)."""
    key = (name or "").strip().lower()
    if key in ("face", "faces"):
        return _face(key, n)
    if key in ("object", "objects"):
        return _object(key, n)
    if key in ("natural", "nature", "scenes"):
        return _natural(key, n)
    if key in ("manmade", "urban"):
        return _manmade(key, n)
    if key == "bright":
        return _bright_dark(key, n, dark=False)
    if key == "dark":
        return _bright_dark(key, n, dark=True)
    if key in ("pleasant", "neutral", "unpleasant"):
        return _affective(key, n, key)
    if key == "standard":
        return _standard_deviant(key, n, deviant=False)
    if key == "deviant":
        return _standard_deviant(key, n, deviant=True)
    if key in ("arrow_left", "left"):
        return _arrow(key, n, left=True)
    if key in ("arrow_right", "right"):
        return _arrow(key, n, left=False)
    if key == "flash":
        return _flash(key, n)
    if key == "blank":
        return _flash(key, n, dark=False)
    if key.startswith("flicker"):
        return _flicker(key, n)
    if key in ("percept", "imagery"):
        # Imagery vs perception (T705): the same object drawn sharp vs degraded,
        # so the two conditions share content and differ in image quality.
        img = _object(key, n)
        if key == "imagery":
            from PIL import Image, ImageFilter

            img = img.filter(ImageFilter.GaussianBlur(6 * SUPERSAMPLE))
        return img
    if key in ("easy", "hard"):
        img = _natural(key, n)
        if key == "hard":
            from PIL import Image, ImageFilter

            img = img.filter(ImageFilter.GaussianBlur(3 * SUPERSAMPLE))
        return img
    return _generic(key or "unnamed", n)


# ---------------------------------------------------------------- public API
#: Process-wide memo for :func:`builtin_images`.
_IMAGE_CACHE: dict[str, list[Path]] = {}


def ensure_builtin_images(force: bool = False, categories: Iterable[str] | None = None
                          ) -> dict[str, list[Path]]:
    """Draw (or re-use) the fallback image set; returns ``{category: [paths]}``.

    Idempotent: an existing PNG is re-used unless ``force``.  Any category name
    is accepted -- unknown names are rendered deterministically rather than
    raising, because this function is the last line of defence when a download
    fails and a session is already in progress.
    """
    ensure_dirs()
    BUILTIN_DIR.mkdir(parents=True, exist_ok=True)
    names = list(categories) if categories is not None else list(CATEGORIES) + list(EXTRA_CATEGORIES)
    out: dict[str, list[Path]] = {}
    seen: set[str] = set()
    for raw in names:
        name = str(raw or "").strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        paths: list[Path] = []
        for n in range(PER_CATEGORY):
            path = BUILTIN_DIR / f"{_slug(name)}_{n}.png"
            if not path.exists() or force:
                img = _render(name, n)
                _save(img, path)
            paths.append(path)
        out[name] = paths
    return out


def builtin_images(category: str, force: bool = False) -> list[Path]:
    """Paths for one category (drawn on demand); never raises for a new name.

    Memoised per process: a curriculum run asks for the same handful of
    categories once per test, and re-globbing the directory 64 times in a row is
    pure overhead on the "start the session" path.
    """
    key = str(category or "").strip().lower()
    if not force and key in _IMAGE_CACHE:
        return list(_IMAGE_CACHE[key])
    paths = ensure_builtin_images(force=force, categories=[key]).get(key, [])
    _IMAGE_CACHE[key] = list(paths)
    return list(paths)


def _slug(name: str) -> str:
    out = "".join(c if (c.isalnum() or c in "-_") else "-" for c in str(name).lower())
    return out.strip("-") or "unnamed"


def is_known(category: str) -> bool:
    """True when ``category`` has a bespoke drawing routine (vs the generic one)."""
    key = str(category or "").strip().lower()
    return key in (set(CATEGORIES) | set(EXTRA_CATEGORIES) | {"faces", "objects", "scenes"})


__all__ = ["CATEGORIES", "EXTRA_CATEGORIES", "BUILTIN_DIR", "PALETTES", "SIZE",
           "ensure_builtin_images", "builtin_images", "is_known"]
