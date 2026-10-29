"""Turn a photo into a coloured, high-resolution ASCII portrait for the profile card.

Usage (from repo root):
    pip install "rembg[cpu]" numpy pillow
    python card/portrait.py card/avatar.png --cols 128

Writes card/portrait.json: one entry per row, each a list of [text, hex colour] runs.
Character density follows brightness plus local detail; colour comes from the photo,
so the result reads as a likeness rather than a silhouette.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

RAMP = " .`':;!i|ljbkWMH%%@@@"  # sparse to dense
CHAR_ASPECT = 0.5  # glyph width / line height
OUT = Path(__file__).resolve().parent / "portrait.json"


def remove_background(img: Image.Image) -> Image.Image:
    from rembg import new_session, remove

    return remove(img.convert("RGB"), session=new_session("u2net_human_seg")).convert("RGBA")


def build(path: str, cols: int, keep_bg: bool,
          crop: tuple[float, float, float, float] | None = None) -> list[list[list[str]]]:
    src = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if crop:  # fractions of width/height: left, top, right, bottom
        w, h = src.size
        src = src.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    rgba = src.convert("RGBA") if keep_bg else remove_background(src)

    from skimage.exposure import equalize_adapthist

    # Adaptive (local) histogram equalisation at full resolution: the face is equalised against
    # its own neighbourhood, not against a white shirt, so features survive on darker skin.
    w, h = rgba.size
    full_lum = np.asarray(rgba.convert("L"), dtype=float) / 255
    local_eq = equalize_adapthist(full_lum, kernel_size=max(8, w // 6), clip_limit=0.02)
    eq_img = Image.fromarray((local_eq * 255).astype(np.uint8))

    rows = max(1, round(h / w * cols * CHAR_ASPECT))
    small = rgba.resize((cols, rows), Image.Resampling.LANCZOS)
    alpha = np.asarray(small.split()[3], dtype=float) / 255
    eq_small = eq_img.resize((cols, rows), Image.Resampling.LANCZOS)
    eq = np.asarray(eq_small, dtype=float) / 255
    blur = np.asarray(eq_small.filter(ImageFilter.GaussianBlur(1.5)), dtype=float) / 255
    detail = np.clip((eq - blur) * 3.0, -0.4, 0.4)  # edges: eyes, brows, beard line, collar
    # Mostly dense glyphs so colour carries the photograph; local tone and edges vary the
    # glyph weight on top of that, which is what gives the ASCII texture.
    density = np.clip(0.55 + 0.45 * eq + 0.6 * detail, 0.3, 1)

    # Colour: the photo's own colour, with shadows lifted (gamma) so a darker face still reads
    # on the dark panel, and local contrast mixed in to sharpen eyes, brows and beard.
    hsv = np.asarray(ImageEnhance.Color(small.convert("RGB")).enhance(1.25).convert("HSV"), dtype=float)
    v = hsv[..., 2] / 255
    hsv[..., 2] = np.clip(0.65 * v ** 0.6 + 0.35 * eq + 0.08, 0, 1) * 255
    colour = np.asarray(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"), dtype=float)

    out: list[list[list[str]]] = []
    for y in range(rows):
        runs: list[list[str]] = []
        for x in range(cols):
            if alpha[y, x] < 0.5:
                ch, hexc = " ", ""
            else:
                ch = RAMP[min(len(RAMP) - 1, int(density[y, x] * (len(RAMP) - 1) + 0.5))]
                r, g, b = (int(min(255, c)) & 0xF8 for c in colour[y, x])  # quantise to merge runs
                hexc = f"#{r:02x}{g:02x}{b:02x}"
            if runs and runs[-1][1] == hexc:
                runs[-1][0] += ch
            else:
                runs.append([ch, hexc])
        while runs and runs[-1][0].strip() == "":
            runs.pop()
        out.append(runs)
    while out and not out[0]:
        out.pop(0)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--cols", type=int, default=128)
    p.add_argument("--keep-bg", action="store_true", help="skip background removal")
    p.add_argument("--crop", type=float, nargs=4, metavar=("L", "T", "R", "B"),
                   help="crop as fractions, e.g. 0.12 0 0.92 0.72 for head and shoulders")
    a = p.parse_args()
    data = build(a.image, a.cols, a.keep_bg, a.crop)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT} ({len(data)} rows)")
