"""Maps a free-text object phrase (e.g. "Coca-Cola bottle") onto the closest
COCO class that our segmentation model (YOLOv8-seg) actually knows how to
detect (e.g. "bottle").

This is the practical bridge between "the LLM understood the user meant a
Coca-Cola bottle" and "the vision model can only detect generic COCO
classes" - a real constraint of using an off-the-shelf detector instead of
training a custom one, and worth calling out in the README.
"""
from __future__ import annotations

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Brand / colloquial synonyms -> nearest COCO class the detector can localize.
SYNONYMS: dict[str, str] = {
    "coca-cola": "bottle", "coca cola": "bottle", "coke": "bottle", "pepsi": "bottle",
    "soda": "bottle", "soda can": "can" if "can" in COCO_CLASSES else "bottle",
    "water bottle": "bottle", "beer": "bottle", "juice": "bottle", "drink": "bottle",
    "can": "bottle",  # COCO has no "can" class; nearest is bottle
    "mug": "cup", "coffee cup": "cup", "wine": "wine glass",
    "mobile": "cell phone", "smartphone": "cell phone", "iphone": "cell phone",
    "phone": "cell phone", "notebook": "laptop", "computer": "laptop",
    "television": "tv", "screen": "tv", "purse": "handbag", "bag": "backpack",
    "sofa": "couch", "table": "dining table", "plant": "potted plant",
    "bike": "bicycle", "motorbike": "motorcycle", "auto": "car", "vehicle": "car",
}


def best_coco_match(phrase: str) -> str | None:
    text = phrase.lower().strip()
    if not text:
        return None

    for key, coco in SYNONYMS.items():
        if key in text:
            return coco

    for cls in COCO_CLASSES:
        if cls in text:
            return cls

    words = set(text.replace("-", " ").split())
    for cls in COCO_CLASSES:
        if set(cls.split()) & words:
            return cls

    return None
