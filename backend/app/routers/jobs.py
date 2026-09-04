from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import UPLOADS_DIR
from app.job_store import job_store
from app.schemas import JobResponse

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def _url_for(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if p.is_relative_to(UPLOADS_DIR):
        return f"/media/uploads/{p.relative_to(UPLOADS_DIR).as_posix()}"
    from app.config import OUTPUTS_DIR
    if p.is_relative_to(OUTPUTS_DIR):
        return f"/media/outputs/{p.relative_to(OUTPUTS_DIR).as_posix()}"
    return None


async def _save_upload(upload: UploadFile, dest_dir: Path, allowed_ext: set[str]) -> Path:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(allowed_ext)}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"input{ext}"

    size = 0
    with dest_path.open("wb") as out_file:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out_file.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(400, "File too large (max 200MB)")
            out_file.write(chunk)
    return dest_path


@router.post("", response_model=JobResponse)
async def create_job(
    prompt: str = Form(...),
    video: UploadFile | None = None,
    video_url: Optional[str] = Form(default=None),
    reference_image: UploadFile | None = None,
):
    if not prompt or not prompt.strip():
        raise HTTPException(400, "prompt is required")
    if not video and not video_url:
        raise HTTPException(400, "either a video file or a video_url must be provided")

    import uuid
    job_id_dir = UPLOADS_DIR / uuid.uuid4().hex[:12]

    if video is not None:
        video_path = await _save_upload(video, job_id_dir, ALLOWED_VIDEO_EXT)
    else:
        video_path = await _download_video_url(video_url, job_id_dir)

    reference_path: Optional[Path] = None
    if reference_image is not None and reference_image.filename:
        reference_path = await _save_upload(reference_image, job_id_dir, ALLOWED_IMAGE_EXT)

    job = job_store.create_job(
        input_video_path=str(video_path),
        reference_image_path=str(reference_path) if reference_path else None,
        prompt=prompt.strip(),
    )
    # re-home the job under its own id so URLs are stable/predictable
    final_dir = UPLOADS_DIR / job.job_id
    shutil.move(str(job_id_dir), str(final_dir))
    job.input_video_path = str(final_dir / Path(job.input_video_path).name)
    if job.reference_image_path:
        job.reference_image_path = str(final_dir / Path(job.reference_image_path).name)

    return job.to_response(_url_for)


async def _download_video_url(url: Optional[str], dest_dir: Path) -> Path:
    if not url:
        raise HTTPException(400, "video_url is empty")
    import httpx

    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        ext = ".mp4"
    dest_path = dest_dir / f"input{ext}"

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            if resp.status_code != 200:
                raise HTTPException(400, f"Could not download video_url (status {resp.status_code})")
            size = 0
            with dest_path.open("wb") as f:
                async for chunk in resp.aiter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(400, "Downloaded video too large (max 200MB)")
                    f.write(chunk)
    return dest_path


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.to_response(_url_for)


@router.get("/{job_id}/download")
async def download_job_output(job_id: str):
    job = job_store.get(job_id)
    if job is None or not job.output_video_path:
        raise HTTPException(404, "output not ready")
    return FileResponse(job.output_video_path, media_type="video/mp4", filename=f"{job_id}_output.mp4")
