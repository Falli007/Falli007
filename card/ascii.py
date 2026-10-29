"""Convert a photo into ASCII art for the profile card.

Usage:
    python ascii.py photo.jpg --cols 40 --cutout > ascii.txt

--cutout removes the background with rembg (pip install "rembg[cpu]") and boosts
local contrast so facial features survive on darker skin tones and busy rooms.
Without it, bright pixels map to dense characters; use --invert for a bright background.
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Dense to sparse, using the character set of the reference card.
RAMP = "@@@%%%HHMWkbjl|i!;:'`.    "  # trailing spaces keep the background clean
CHAR_ASPECT = 0.5  # monospace glyphs are roughly twice as tall as wide


def cutout(img: Image.Image) -> Image.Image:
    """Remove the background and return a greyscale image with local detail lifted."""
    from rembg import new_session, remove

    rgba = remove(img.convert("RGB"), session=new_session("u2net_human_seg")).convert("RGBA")
    alpha = np.asarray(rgba.split()[3], dtype=float) / 255
    grey = rgba.convert("L")
    lum = np.asarray(grey, dtype=float) / 255
    blur = np.asarray(grey.filter(ImageFilter.GaussianBlur(18)), dtype=float) / 255
    local = np.clip(0.5 + (lum - blur) * 2.2, 0, 1)  # high-pass: eyes, beard, collar edges
    mixed = 0.55 * np.sqrt(lum) + 0.45 * local  # sqrt lifts shadows without blowing highlights
    return Image.fromarray((np.clip(mixed, 0, 1) * alpha * 255).astype(np.uint8))


def to_ascii(path: str, cols: int, invert: bool = False, contrast: float = 1.4,
             use_cutout: bool = False) -> str:
    img = ImageOps.exif_transpose(Image.open(path))
    if use_cutout:
        img = cutout(img)
    else:
        img = ImageOps.autocontrast(ImageEnhance.Contrast(img.convert("L")).enhance(contrast), cutoff=2)
        if invert:
            img = ImageOps.invert(img)
    w, h = img.size
    rows = max(1, round(h / w * cols * CHAR_ASPECT))
    img = img.resize((cols, rows), Image.Resampling.LANCZOS)
    scale = (len(RAMP) - 1) / 255
    px = img.load()
    lines = ("".join(RAMP[int((255 - px[x, y]) * scale)] for x in range(cols)).rstrip()
             for y in range(rows))
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--cols", type=int, default=40)
    p.add_argument("--invert", action="store_true")
    p.add_argument("--contrast", type=float, default=1.4)
    p.add_argument("--cutout", action="store_true")
    a = p.parse_args()
    print(to_ascii(a.image, a.cols, a.invert, a.contrast, a.cutout))
