"""Produces a replacement-object image when the user didn't upload a
reference image. Tries a text-to-image API if configured, otherwise
synthesizes a simple labelled placeholder graphic (a coloured rounded
rectangle with the replacement object's name) so the pipeline never blocks
on a missing API key.
"""
from __future__ import annotations

import base64
import hashlib
import logging

import cv2
import httpx
import numpy as np

from app.config import settings

logger = logging.getLogger("lightnoteai.image_gen")


async def get_replacement_image(replacement_object: str, size: int = 512) -> np.ndarray:
    if settings.image_gen_provider.lower() == "openai" and settings.openai_api_key:
        try:
            return await _generate_openai(replacement_object, size)
        except Exception:  # noqa: BLE001
            logger.exception("text-to-image generation failed, using placeholder")

    return _placeholder_image(replacement_object, size)


async def _generate_openai(prompt_text: str, size: int) -> np.ndarray:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={
                "model": settings.openai_image_model,
                "prompt": f"A studio product photo of {prompt_text}, isolated on a plain white background, centered",
                "size": "1024x1024",
                "n": 1,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        b64 = data["data"][0]["b64_json"]
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.resize(img, (size, size))


def _placeholder_image(label: str, size: int) -> np.ndarray:
    """A full-bleed colour gradient with a label, used when neither a
    reference image nor an image-gen API is available.

    This is deliberately flat/edge-free rather than an outlined product
    shape: `replace_object()` composites it with Poisson blending
    (`cv2.seamlessClone`), which propagates *internal* gradients from the
    source into the destination. A drawn silhouette or a hard-edged detail
    (e.g. a cap-shaped rectangle) rarely lines up with the real object's
    actual silhouette in the frame, and the mismatch shows up as dark/light
    blotches. A smooth gradient has no internal edges to clash, so it blends
    predictably regardless of the destination shape - at the cost of not
    looking like an actual product. For a real product photo, upload a
    reference image instead."""
    h = int(hashlib.sha1(label.encode()).hexdigest()[:6], 16)
    base_color = np.array([(h & 0xFF), (h >> 8) & 0xFF, (h >> 16) & 0xFF], dtype=np.float32)
    base_color = 60 + (base_color % 160)  # BGR, kept mid-bright so it blends into scenes

    x = np.linspace(-1, 1, size, dtype=np.float32)
    y = np.linspace(-1, 1, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    sheen = 1.0 - 0.35 * np.sqrt(xx ** 2 + yy ** 2)
    img = (base_color.reshape(1, 1, 3) * sheen[:, :, None]).clip(30, 255).astype(np.uint8)

    text = (label.strip() or "object").split()[-1][:14]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(1.2, (size * 0.8) / (len(text) * 20 + 1)))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)
    org = ((size - tw) // 2, size // 2 + th // 2)
    cv2.putText(img, text, org, font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    return img
