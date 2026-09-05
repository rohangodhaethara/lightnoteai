"""Object detection + segmentation + lightweight tracking.

Model: Ultralytics YOLOv8n-seg (COCO-pretrained). It gives us a bounding
box AND a pixel mask per detected instance in a single forward pass, which
covers both the "detect" and "segment" requirements without needing a
separate SAM pass.

"Tracking" here is a pragmatic greedy match of the chosen instance across
frames by mask-centroid distance + class, with short-gap hold-over (if the
detector misses a frame or two we reuse the last good mask) instead of a
full video object tracker (e.g. SAM2's video predictor / DeepSORT). That
keeps the pipeline CPU-friendly and dependency-light while still producing
temporally coherent edits. The interface below is intentionally narrow so
this strategy can be swapped for SAM2/DeepSORT later without touching the
rest of the pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger("lightnoteai.vision")


@dataclass
class FrameMask:
    found: bool
    mask: Optional[np.ndarray]  # uint8 HxW, 0/255
    bbox: Optional[tuple[int, int, int, int]]  # x1,y1,x2,y2
    confidence: float = 0.0


@lru_cache(maxsize=1)
def _load_model():
    from ultralytics import YOLO

    logger.info("loading YOLO segmentation model: %s", settings.yolo_seg_model)
    return YOLO(settings.yolo_seg_model)


def model_available() -> bool:
    try:
        _load_model()
        return True
    except Exception:  # noqa: BLE001
        logger.exception("failed to load YOLO seg model, will use heuristic fallback")
        return False


def _centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def isolate_product(image_bgr: np.ndarray, coco_class_hint: Optional[str] = None) -> np.ndarray:
    """Detect the product in a user-uploaded reference image and return a
    bbox-cropped version with everything outside its segmentation mask
    painted neutral white, so a candid/lifestyle photo (a hand holding the
    product, a busy background) doesn't get composited into the video along
    with the product. Falls back to the original image unchanged if nothing
    is confidently detected (e.g. an already-isolated product shot, or the
    model being unavailable) - this is a best-effort cleanup, not required
    for good results with a clean reference photo."""
    import cv2

    tracker = ObjectTracker(coco_class=coco_class_hint)
    result = tracker.locate(image_bgr)
    if not result.found or result.mask is None or result.bbox is None:
        return image_bgr

    x1, y1, x2, y2 = result.bbox
    if x2 - x1 < 8 or y2 - y1 < 8:
        return image_bgr

    crop = image_bgr[y1:y2, x1:x2]
    mask_crop = result.mask[y1:y2, x1:x2]
    mask3 = cv2.merge([mask_crop, mask_crop, mask_crop]).astype(np.float32) / 255.0
    neutral = np.full_like(crop, 255)
    isolated = (crop.astype(np.float32) * mask3 + neutral.astype(np.float32) * (1 - mask3)).astype(np.uint8)
    return isolated


class ObjectTracker:
    """Stateful per-video tracker: call `locate(frame)` for every frame in order.

    Detection/segmentation runs on a small resized copy of the frame (fast on
    CPU); the resulting mask/bbox are then scaled back up to the frame's
    actual resolution before being handed to the editor, so output quality
    isn't capped by the detector's inference resolution.
    """

    def __init__(self, coco_class: Optional[str], conf_threshold: Optional[float] = None,
                 detection_width: Optional[int] = None):
        self.coco_class = coco_class
        self.conf_threshold = conf_threshold or settings.yolo_conf_threshold
        self.detection_width = detection_width or settings.detection_max_width
        self._last_bbox: Optional[tuple[int, int, int, int]] = None
        self._misses = 0
        self._max_hold_frames = 6
        self._use_model = model_available()

    def locate(self, frame_bgr: np.ndarray) -> FrameMask:
        import cv2

        h, w = frame_bgr.shape[:2]
        if w > self.detection_width:
            scale = self.detection_width / w
            det_frame = cv2.resize(
                frame_bgr, (self.detection_width, max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            scale = 1.0
            det_frame = frame_bgr

        result = self._locate_scaled(det_frame)
        if not result.found or scale == 1.0:
            return result

        mask_full = cv2.resize(result.mask, (w, h), interpolation=cv2.INTER_NEAREST)
        inv = 1.0 / scale
        x1, y1, x2, y2 = result.bbox
        bbox_full = (
            max(0, int(round(x1 * inv))), max(0, int(round(y1 * inv))),
            min(w, int(round(x2 * inv))), min(h, int(round(y2 * inv))),
        )
        return FrameMask(found=True, mask=mask_full, bbox=bbox_full, confidence=result.confidence)

    def _locate_scaled(self, frame_bgr: np.ndarray) -> FrameMask:
        """Detection logic operating entirely in the (possibly downscaled)
        detection frame's coordinate space; `locate()` scales the result up."""
        if self._use_model:
            result = self._locate_with_model(frame_bgr)
            if result.found:
                self._last_bbox = result.bbox
                self._misses = 0
                return result
            self._misses += 1
            if self._last_bbox is not None and self._misses <= self._max_hold_frames:
                return self._mask_from_bbox(frame_bgr, self._last_bbox, confidence=0.0)
            return FrameMask(found=False, mask=None, bbox=None)

        return self._locate_heuristic(frame_bgr)

    def _locate_with_model(self, frame_bgr: np.ndarray) -> FrameMask:
        model = _load_model()
        results = model.predict(source=frame_bgr, verbose=False, conf=self.conf_threshold)
        if not results:
            return FrameMask(found=False, mask=None, bbox=None)
        r = results[0]
        if r.masks is None or len(r.boxes) == 0:
            return FrameMask(found=False, mask=None, bbox=None)

        names = r.names
        h, w = frame_bgr.shape[:2]
        candidates = []
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            cls_name = names.get(cls_id, str(cls_id))
            conf = float(r.boxes.conf[i].item())
            if self.coco_class and cls_name != self.coco_class:
                continue
            xyxy = r.boxes.xyxy[i].tolist()
            bbox = tuple(int(v) for v in xyxy)
            mask_small = r.masks.data[i].cpu().numpy()
            candidates.append((conf, bbox, mask_small))

        if not candidates:
            return FrameMask(found=False, mask=None, bbox=None)

        if self._last_bbox is not None:
            def score(c):
                conf, bbox, _ = c
                cx, cy = _centroid(bbox)
                lx, ly = _centroid(self._last_bbox)
                dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                return dist - conf * 50
            candidates.sort(key=score)
        else:
            candidates.sort(key=lambda c: -c[0])

        conf, bbox, mask_small = candidates[0]
        mask = (mask_small * 255).astype(np.uint8)
        if mask.shape != (h, w):
            import cv2
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return FrameMask(found=True, mask=mask, bbox=bbox, confidence=conf)

    def _mask_from_bbox(self, frame_bgr: np.ndarray, bbox, confidence: float) -> FrameMask:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
        return FrameMask(found=True, mask=mask, bbox=bbox, confidence=confidence)

    def _locate_heuristic(self, frame_bgr: np.ndarray) -> FrameMask:
        """Fallback when the YOLO weights can't be downloaded/loaded: pick the
        most saturated, centrally-located blob via classic CV. This keeps the
        app functional end-to-end even fully offline, at lower quality."""
        import cv2

        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        _, thresh = cv2.threshold(sat, 60, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            cx1, cy1, cx2, cy2 = int(w * 0.35), int(h * 0.25), int(w * 0.65), int(h * 0.85)
            bbox = (cx1, cy1, cx2, cy2)
            return self._mask_from_bbox(frame_bgr, bbox, confidence=0.1)

        center = np.array([w / 2, h / 2])

        def score(c):
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            cx, cy = x + cw / 2, y + ch / 2
            dist = np.linalg.norm(np.array([cx, cy]) - center)
            return area - dist * 20

        best = max(contours, key=score)
        x, y, cw, ch = cv2.boundingRect(best)
        bbox = (x, y, x + cw, y + ch)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [best], -1, 255, thickness=-1)
        return FrameMask(found=True, mask=mask, bbox=bbox, confidence=0.2)
