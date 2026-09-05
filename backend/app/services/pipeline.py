from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2

from app.config import OUTPUTS_DIR, UPLOADS_DIR, settings
from app.job_store import Job
from app.schemas import JobStatus, Operation
from app.services import editor, image_gen
from app.services.coco_classes import best_coco_match
from app.services.instruction_parser import parse_instruction
from app.services.vision import ObjectTracker, isolate_product
from app.utils import ffmpeg_utils

logger = logging.getLogger("lightnoteai.pipeline")


def _touch(job: Job, status: JobStatus, progress: int, message: str) -> None:
    job.status = status
    job.progress = progress
    job.message = message
    job.updated_at = datetime.now(timezone.utc).isoformat()
    logger.info("job %s -> %s (%s%%) %s", job.job_id, status, progress, message)


async def process_job(job: Job) -> None:
    work_dir = UPLOADS_DIR / job.job_id
    frames_in = work_dir / "frames_in"
    frames_out = work_dir / "frames_out"
    work_dir.mkdir(parents=True, exist_ok=True)

    input_video = Path(job.input_video_path)

    _touch(job, JobStatus.PARSING_INSTRUCTION, 5, "Understanding the instruction with the LLM")
    instruction = await parse_instruction(job.prompt)
    job.parsed_instruction = instruction

    if not ffmpeg_utils.ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not found on PATH; required for video processing")

    duration = ffmpeg_utils.get_duration_seconds(input_video)
    working_video = input_video
    if duration > settings.max_video_seconds:
        trimmed = work_dir / "trimmed_input.mp4"
        await ffmpeg_utils.trim_video(input_video, trimmed, settings.max_video_seconds)
        working_video = trimmed
        _touch(job, JobStatus.PARSING_INSTRUCTION, 8,
               f"Video trimmed to first {settings.max_video_seconds}s for this CPU demo pipeline")

    _touch(job, JobStatus.DETECTING_OBJECT, 15, "Extracting frames")
    fps = await ffmpeg_utils.extract_frames(working_video, frames_in, settings.output_max_width)

    reference_bgr = None
    if job.reference_image_path:
        reference_bgr = cv2.imread(job.reference_image_path, cv2.IMREAD_COLOR)
        if reference_bgr is not None:
            replacement_class_hint = best_coco_match(instruction.replacement_object or instruction.target_object)
            # Crop to the detected product and neutralize the background so a
            # candid/lifestyle reference photo (a hand holding the product, a
            # busy background) doesn't get composited into the video too -
            # falls back to the original image untouched if nothing is
            # confidently detected in it.
            reference_bgr = await asyncio.to_thread(isolate_product, reference_bgr, replacement_class_hint)

    if instruction.operation == Operation.REPLACE_OBJECT and reference_bgr is None:
        replacement_label = instruction.replacement_object or instruction.target_object
        reference_bgr = await image_gen.get_replacement_image(replacement_label)

    _touch(job, JobStatus.TRACKING_OBJECT, 25, f"Locating '{instruction.target_object}' in the video")

    frame_paths = sorted(frames_in.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError("No frames extracted from the input video")

    frames_out.mkdir(parents=True, exist_ok=True)

    def run_edit_loop() -> tuple[int, int]:
        tracker = ObjectTracker(coco_class=instruction.coco_class)
        found_count = 0
        total = len(frame_paths)
        for idx, fp in enumerate(frame_paths):
            frame = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            located = tracker.locate(frame)

            if located.found and located.mask is not None and located.bbox is not None:
                found_count += 1
                if instruction.operation == Operation.REMOVE_OBJECT:
                    edited = editor.remove_object(frame, located.mask)
                elif instruction.operation == Operation.REPLACE_TEXT:
                    edited = editor.replace_text(
                        frame, located.mask, located.bbox,
                        instruction.replacement_text or instruction.replacement_object or "?",
                    )
                else:
                    edited = editor.replace_object(frame, located.mask, located.bbox, reference_bgr)
            else:
                edited = frame

            cv2.imwrite(str(frames_out / fp.name), edited)

            if idx % max(1, total // 20) == 0:
                pct = 25 + int(60 * (idx + 1) / total)
                job.progress = min(85, pct)
                job.updated_at = datetime.now(timezone.utc).isoformat()

        return found_count, total

    _touch(job, JobStatus.EDITING_FRAMES, 30, "Editing frames (detect -> mask -> composite)")
    found_count, total = await asyncio.to_thread(run_edit_loop)

    not_found_warning = None
    if found_count == 0:
        not_found_warning = (
            f"Warning: '{instruction.target_object}' was not confidently detected in any frame; "
            "output is close to the original. Try a clearer object description or a closer shot."
        )

    _touch(job, JobStatus.RENDERING_VIDEO, 90, "Rendering final video with ffmpeg")
    output_path = OUTPUTS_DIR / f"{job.job_id}.mp4"
    await ffmpeg_utils.frames_to_video(frames_out, fps, output_path, original_video=working_video)

    job.output_video_path = str(output_path)
    detection_rate = f"{found_count}/{total} frames"
    completion_note = not_found_warning or f"Done. Object located in {detection_rate}."
    _touch(job, JobStatus.COMPLETED, 100, completion_note)

    shutil.rmtree(frames_in, ignore_errors=True)
    shutil.rmtree(frames_out, ignore_errors=True)
