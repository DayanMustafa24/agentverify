"""B1: frozen CLIP features + logistic regression (no LLM, no retrieval).

Features per post: [img, txt, img*txt, |img-txt|, cos(img,txt)] with L2-normalised embeddings.
Needs `pip install -e .[clip]`. The extractor is injectable so tests run without weights.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..contracts import SOURCE_LABELS, PostInput


class Extractor(Protocol):
    def embed(self, posts: Sequence[PostInput]) -> tuple[np.ndarray, np.ndarray]: ...


class ClipExtractor:
    def __init__(self, name: str = "openai/clip-vit-base-patch32", device: str | None = None, batch: int = 32):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.batch = batch

    def embed(self, posts):
        from PIL import Image
        torch = self.torch
        imgs, txts = [], []
        for i in range(0, len(posts), self.batch):
            chunk = posts[i:i + self.batch]
            ims = [Image.open(p.image_path).convert("RGB") for p in chunk]
            inp = self.proc(text=[p.text for p in chunk], images=ims, return_tensors="pt",
                            padding=True, truncation=True, max_length=77).to(self.device)
            with torch.no_grad():
                out = self.model(**inp)
            imgs.append(out.image_embeds.cpu().numpy())
            txts.append(out.text_embeds.cpu().numpy())
        return np.vstack(imgs), np.vstack(txts)


class DummyExtractor:
    """Hash-seeded random embeddings. For tests only."""
    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed(self, posts):
        def vec(s):
            r = np.random.default_rng(abs(hash(s)) % (2**32))
            return r.normal(size=self.dim)
        return (np.stack([vec(p.image_path) for p in posts]),
                np.stack([vec(p.text) for p in posts]))


def build_features(img: np.ndarray, txt: np.ndarray) -> np.ndarray:
    img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-9)
    txt = txt / (np.linalg.norm(txt, axis=1, keepdims=True) + 1e-9)
    cos = (img * txt).sum(1, keepdims=True)
    return np.hstack([img, txt, img * txt, np.abs(img - txt), cos])


class ClipLR:
    """One LR for binary, one for 4-way. Confidence = max class probability."""
    def __init__(self, C: float = 1.0, seed: int = 42):
        kw = dict(C=C, max_iter=2000, class_weight="balanced", random_state=seed)
        self.bin = LogisticRegression(**kw)
        self.multi = LogisticRegression(**kw)

    def fit(self, X: np.ndarray, posts: Sequence[PostInput]):
        self.bin.fit(X, [int(p.gold_is_fake) for p in posts])
        self.multi.fit(X, [p.gold_source.value for p in posts])
        return self

    def predict(self, X: np.ndarray):
        pb = self.bin.predict_proba(X)            # columns: [real, fake]
        pm = self.multi.predict_proba(X)
        mlabels = list(self.multi.classes_)
        return pb, pm, mlabels
