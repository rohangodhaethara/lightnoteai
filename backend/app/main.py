import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import OUTPUTS_DIR, UPLOADS_DIR, settings
from app.job_store import job_store
from app.routers import jobs
from app.services.pipeline import process_job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="LightNoteAI Video Editor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/media/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

app.include_router(jobs.router)


@app.on_event("startup")
async def startup() -> None:
    job_store.set_processor(process_job)
    job_store.start_workers()


@app.get("/api/health")
async def health():
    return {"status": "ok"}
