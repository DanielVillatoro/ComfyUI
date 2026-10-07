"""Add smartphone-style sensor noise and JPEG compression to an image.

Usage: venv/bin/python add_phone_grain.py INPUT.png [OUTPUT.jpg] [--strength 3.0] [--quality 82] [--seed 0]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def add_grain(img, strength, seed):
    rng = np.random.default_rng(seed)
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    # Luminance noise, stronger in shadows like a small phone sensor.
    sigma = strength * (1.4 - 0.6 * luma / 255.0)
    rgb += (rng.standard_normal(luma.shape).astype(np.float32) * sigma)[..., None]

    # Low-frequency chroma blotches: noise generated at 1/4 size and upscaled.
    h, w = luma.shape
    small = rng.standard_normal((h // 4 + 1, w // 4 + 1, 3)).astype(np.float32) * strength * 0.5
    chroma = np.stack([np.asarray(Image.fromarray(small[..., c]).resize((w, h), Image.BILINEAR)) for c in range(3)], -1)
    rgb += chroma - chroma.mean(axis=-1, keepdims=True)  # zero-mean across channels: shifts hue, not brightness

    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--strength", type=float, default=3.0)
    parser.add_argument("--quality", type=int, default=82)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    src = Path(args.input)
    dest = Path(args.output) if args.output else src.with_name(src.stem + "_grain.jpg")
    add_grain(Image.open(src), args.strength, args.seed).save(dest, "JPEG", quality=args.quality, subsampling=2)
    print(dest)


if __name__ == "__main__":
    main()
