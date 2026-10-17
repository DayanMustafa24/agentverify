"""Label-preserving perturbations for the Phase 2 robustness study (Gap 2).
Severity 1..3. Kept small and deterministic (seeded)."""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter

JPEG_Q = {1: 60, 2: 40, 3: 20}
BLUR_R = {1: 1.0, 2: 2.0, 3: 3.5}
NOISE_S = {1: 8, 2: 16, 3: 28}          # std on 0-255 scale
CROP = {1: 0.9, 2: 0.8, 3: 0.65}         # keep fraction of each side
TYPO_P = {1: 0.03, 2: 0.06, 3: 0.10}

_NEIGH = {"a": "sqzw", "e": "wrds", "i": "uojk", "o": "iplk", "u": "yihj", "n": "bmhj", "t": "ryfg", "s": "adwe", "r": "etdf"}


def jpeg(img: Image.Image, sev: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=JPEG_Q[sev])
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def blur(img: Image.Image, sev: int) -> Image.Image:
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(BLUR_R[sev]))


def noise(img: Image.Image, sev: int, seed: int = 0) -> Image.Image:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    a = a + np.random.default_rng(seed).normal(0, NOISE_S[sev], a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def center_crop(img: Image.Image, sev: int) -> Image.Image:
    w, h = img.size
    f = CROP[sev]
    cw, ch = int(w * f), int(h * f)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img.convert("RGB").crop((x0, y0, x0 + cw, y0 + ch))


def typos(text: str, sev: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    out = []
    for ch in text:
        if ch.lower() in _NEIGH and rng.random() < TYPO_P[sev]:
            out.append(rng.choice(_NEIGH[ch.lower()]))
        else:
            out.append(ch)
    return "".join(out)


def shout(text: str, sev: int, seed: int = 0) -> str:
    return text.upper() + "!" * sev


def hashtags(text: str, sev: int, seed: int = 0) -> str:
    tags = ["#breaking", "#mustsee", "#share", "#viral", "#news"]
    return text + " " + " ".join(tags[:sev + 1])


IMAGE_PERTURBATIONS = {"jpeg": jpeg, "blur": blur, "noise": noise, "crop": center_crop}
TEXT_PERTURBATIONS = {"typos": typos, "shout": shout, "hashtags": hashtags}
