"""In-memory job store + a tiny async background worker queue.

This is intentionally simple (no Redis/Celery) so the assignment runs with
zero extra infrastructure, but it is isolated behind a small interface so it
could be swapped for a real queue (Celery/RQ/Arq + Redis) without touching
the API layer or the processing pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Coroutine, Optional

from app.schemas import JobResponse, JobStatus, ParsedInstruction

logger = logging.getLogger("lightnoteai.jobs")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Job:
    def __init__(self, job_id: str, input_video_path: str, reference_image_path: Optional[str], prompt: str):
        self.job_id = job_id
        self.input_video_path = input_video_path
        self.reference_image_path = reference_image_path
        self.prompt = prompt
        self.status = JobStatus.QUEUED
        self.progress = 0
        self.message = "Job queued"
        self.parsed_instruction: Optional[ParsedInstruction] = None
        self.output_video_path: Optional[str] = None
        self.error: Optional[str] = None
        self.created_at = _now()
        self.updated_at = _now()

    def to_response(self, url_for) -> JobResponse:
        return JobResponse(
            job_id=self.job_id,
            status=self.status,
            progress=self.progress,
            message=self.message,
            parsed_instruction=self.parsed_instruction,
            error=self.error,
            input_video_url=url_for(self.input_video_path),
            reference_image_url=url_for(self.reference_image_path) if self.reference_image_path else None,
            output_video_url=url_for(self.output_video_path) if self.output_video_path else None,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class JobStore:
    def __init__(self, max_workers: int = 2):
        self._jobs: dict[str, Job] = {}
        self._queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._max_workers = max_workers
        self._workers_started = False
        self._processor: Optional[Callable[[Job], Coroutine]] = None

    def set_processor(self, processor: Callable[[Job], Coroutine]) -> None:
        self._processor = processor

    def start_workers(self) -> None:
        if self._workers_started:
            return
        self._workers_started = True
        for i in range(self._max_workers):
            asyncio.create_task(self._worker_loop(i))

    async def _worker_loop(self, worker_index: int) -> None:
        logger.info("worker %s started", worker_index)
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if job is None or self._processor is None:
                self._queue.task_done()
                continue
            try:
                await self._processor(job)
            except Exception as exc:  # noqa: BLE001 - job errors are surfaced to the client
                logger.exception("job %s failed", job_id)
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.updated_at = _now()
            finally:
                self._queue.task_done()

    def create_job(self, input_video_path: str, reference_image_path: Optional[str], prompt: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id, input_video_path, reference_image_path, prompt)
        self._jobs[job_id] = job
        self._queue.put_nowait(job_id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)


job_store = JobStore()
