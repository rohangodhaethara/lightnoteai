"""Per-frame pixel edits: remove, replace-with-image, replace-text.

Kept as pure functions of (frame, mask, bbox, ...) -> edited frame so they
are easy to unit test and to swap out (e.g. for a diffusion inpainting
model) independently of the detection/tracking code.
"""
from __future__ import annotations

import cv2
import numpy as np


def remove_object(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    dilated = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    return cv2.inpaint(frame_bgr, dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)


def replace_object(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    replacement_bgr: np.ndarray,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return frame_bgr

    # Fit the replacement image into the bbox preserving its aspect ratio
    # (rather than stretching it to fill) - a naive stretch smears any
    # non-matching content (e.g. a hand or background in a candid reference
    # photo) across the whole box. The fitted crop is pasted onto a copy of
    # the original bbox pixels so seamlessClone has real image content to
    # blend at the edges instead of a hard rectangle.
    rh, rw = replacement_bgr.shape[:2]
    scale = min(bw / rw, bh / rh)
    new_w, new_h = max(1, round(rw * scale)), max(1, round(rh * scale))
    resized = cv2.resize(replacement_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    off_x, off_y = (bw - new_w) // 2, (bh - new_h) // 2

    canvas = frame_bgr[y1:y2, x1:x2].copy()
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

    fit_mask = np.zeros((bh, bw), dtype=np.uint8)
    fit_mask[off_y:off_y + new_h, off_x:off_x + new_w] = 255

    region_mask = mask[y1:y2, x1:x2]
    if region_mask.max() == 0:
        region_mask = fit_mask
    else:
        # Only blend where the pasted content AND the detected object
        # silhouette overlap, so we don't pull in the frame's own background.
        region_mask = cv2.bitwise_and(region_mask, fit_mask)
    # Shrink slightly so the clone boundary sits inside the pasted content's
    # edge rather than exactly on it.
    eroded = cv2.erode(region_mask, np.ones((5, 5), np.uint8), iterations=1)
    if eroded.max() > 0:
        region_mask = eroded

    center = (x1 + bw // 2, y1 + bh // 2)

    try:
        blended = cv2.seamlessClone(canvas, frame_bgr, region_mask, center, cv2.NORMAL_CLONE)
        return blended
    except cv2.error:
        out = frame_bgr.copy()
        mask3 = cv2.merge([region_mask, region_mask, region_mask]).astype(np.float32) / 255.0
        roi = out[y1:y2, x1:x2].astype(np.float32)
        out[y1:y2, x1:x2] = (canvas.astype(np.float32) * mask3 + roi * (1 - mask3)).astype(np.uint8)
        return out


def replace_text(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    replacement_text: str,
) -> np.ndarray:
    inpainted = remove_object(frame_bgr, mask)
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(2.0, bw / (len(replacement_text) * 18 + 1)))
    (tw, th), _ = cv2.getTextSize(replacement_text, font, font_scale, 2)
    org = (x1 + max(0, (bw - tw) // 2), y1 + (bh + th) // 2)
    cv2.putText(inpainted, replacement_text, org, font, font_scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(inpainted, replacement_text, org, font, font_scale, (20, 20, 20), 1, cv2.LINE_AA)
    return inpainted
