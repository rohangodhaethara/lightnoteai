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
    """Deterministic colour per label so re-runs look consistent."""
    h = int(hashlib.sha1(label.encode()).hexdigest()[:6], 16)
    color = ((h & 0xFF), (h >> 8) & 0xFF, (h >> 16) & 0xFF)
    color = tuple(int(60 + c % 160) for c in color)

    img = np.full((size, size, 3), 255, dtype=np.uint8)
    pad = size // 8
    cv2.rectangle(img, (pad, pad), (size - pad, size - pad), color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (pad, pad), (size - pad, size - pad), (30, 30, 30), thickness=4, lineType=cv2.LINE_AA)

    text = label.strip() or "object"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(1.4, (size * 0.8) / (len(text) * 22 + 1)))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)
    org = (max(pad, (size - tw) // 2), size // 2 + th // 2)
    cv2.putText(img, text, org, font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    return img
