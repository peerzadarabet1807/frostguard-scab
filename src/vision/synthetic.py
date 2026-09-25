"""Procedural apple-leaf images with scab-like lesions and ground-truth boxes.

Used for unit tests, the dashboard's "sample leaf" button and smoke-testing the
fallback detector. These are *synthetic* renders, not field photographs.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BACKGROUND = (232, 236, 226)
LEAF_GREEN = (72, 138, 52)
VEIN_GREEN = (118, 172, 86)
LESION_COLOURS = ((84, 74, 36), (66, 58, 30), (92, 84, 44), (58, 52, 34))

Box = tuple[int, int, int, int]


def render_synthetic_leaf(
    width: int = 800,
    height: int = 600,
    n_lesions: int = 5,
    seed: int = 7,
    lesion_radius: tuple[int, int] = (9, 22),
) -> tuple[Image.Image, list[Box]]:
    """Render a leaf with ``n_lesions`` olive-brown scab spots.

    Returns:
        ``(image, boxes)`` where ``boxes`` are ``(x1, y1, x2, y2)`` lesion extents.
    """
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(img)

    # Leaf blade: an ellipse filling most of the frame, plus a petiole.
    cx, cy = width / 2, height / 2
    rx, ry = width * 0.40, height * 0.34
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=LEAF_GREEN)
    draw.line((cx - rx - 60, cy + 10, cx - rx + 4, cy), fill=(96, 128, 60), width=7)

    # Venation: midrib + alternating secondary veins.
    draw.line((cx - rx, cy, cx + rx, cy), fill=VEIN_GREEN, width=4)
    for k in range(-4, 5):
        x0 = cx + k * rx / 5.5
        for sign in (-1, 1):
            x1 = x0 + rx * 0.18
            y1 = cy + sign * ry * 0.78 * math.sqrt(max(0.0, 1 - ((x1 - cx) / rx) ** 2))
            draw.line((x0, cy, x1, y1), fill=VEIN_GREEN, width=2)

    # Lesions: irregular olive/brown blobs placed well inside the blade and apart.
    boxes: list[Box] = []
    attempts = 0
    while len(boxes) < n_lesions and attempts < 500:
        attempts += 1
        r = int(rng.integers(lesion_radius[0], lesion_radius[1] + 1))
        angle = rng.uniform(0, 2 * math.pi)
        dist = math.sqrt(rng.uniform(0.0, 0.62))
        lx = int(cx + dist * (rx - r) * math.cos(angle))
        ly = int(cy + dist * (ry - r) * math.sin(angle))
        if any(abs(lx - (b[0] + b[2]) / 2) < 70 and abs(ly - (b[1] + b[3]) / 2) < 70 for b in boxes):
            continue
        colour = LESION_COLOURS[int(rng.integers(len(LESION_COLOURS)))]
        # A lumpy outline: union of a core ellipse and a few satellite blobs.
        rxl, ryl = r * rng.uniform(0.85, 1.15), r * rng.uniform(0.75, 1.05)
        draw.ellipse((lx - rxl, ly - ryl, lx + rxl, ly + ryl), fill=colour)
        x_min, y_min, x_max, y_max = lx - rxl, ly - ryl, lx + rxl, ly + ryl
        for _ in range(3):
            sa = rng.uniform(0, 2 * math.pi)
            sr = r * rng.uniform(0.35, 0.55)
            sx, sy = lx + 0.7 * rxl * math.cos(sa), ly + 0.7 * ryl * math.sin(sa)
            draw.ellipse((sx - sr, sy - sr, sx + sr, sy + sr), fill=colour)
            x_min, y_min = min(x_min, sx - sr), min(y_min, sy - sr)
            x_max, y_max = max(x_max, sx + sr), max(y_max, sy + sr)
        boxes.append((int(x_min), int(y_min), int(math.ceil(x_max)), int(math.ceil(y_max))))

    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    noise = rng.normal(0.0, 4.0, (height, width, 3))
    arr = np.clip(np.asarray(img, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr), boxes


def render_healthy_leaf(width: int = 800, height: int = 600, seed: int = 7) -> Image.Image:
    """Same leaf without lesions (negative control)."""
    image, _ = render_synthetic_leaf(width, height, n_lesions=0, seed=seed)
    return image
