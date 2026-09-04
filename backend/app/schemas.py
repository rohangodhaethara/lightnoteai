from enum import Enum
from typing import Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    QUEUED = "queued"
    PARSING_INSTRUCTION = "parsing_instruction"
    DETECTING_OBJECT = "detecting_object"
    TRACKING_OBJECT = "tracking_object"
    EDITING_FRAMES = "editing_frames"
    RENDERING_VIDEO = "rendering_video"
    COMPLETED = "completed"
    FAILED = "failed"


class Operation(str, Enum):
    REPLACE_OBJECT = "replace_object"
    REMOVE_OBJECT = "remove_object"
    REPLACE_TEXT = "replace_text"


class ParsedInstruction(BaseModel):
    operation: Operation
    target_object: str
    replacement_object: Optional[str] = None
    target_text: Optional[str] = None
    replacement_text: Optional[str] = None
    coco_class: Optional[str] = None
    raw_prompt: str
    parsed_by: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = 0
    message: str = ""
    parsed_instruction: Optional[ParsedInstruction] = None
    error: Optional[str] = None
    input_video_url: Optional[str] = None
    reference_image_url: Optional[str] = None
    output_video_url: Optional[str] = None
    created_at: str
    updated_at: str
