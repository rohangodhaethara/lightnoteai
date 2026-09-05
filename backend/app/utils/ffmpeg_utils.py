from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe_video(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def get_fps(path: Path) -> float:
    info = probe_video(path)
    for stream in info["streams"]:
        if stream.get("codec_type") == "video":
            num, den = stream["r_frame_rate"].split("/")
            den = int(den) or 1
            return int(num) / den
    return 25.0


def get_duration_seconds(path: Path) -> float:
    info = probe_video(path)
    return float(info["format"].get("duration", 0.0))


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a subprocess synchronously. Callers await this via asyncio.to_thread
    instead of asyncio.create_subprocess_exec, which requires a Proactor event
    loop on Windows and raises NotImplementedError under uvicorn --reload
    (whose watcher can leave a Selector loop installed)."""
    return subprocess.run(cmd, capture_output=True)


async def extract_frames(video_path: Path, out_dir: Path, max_width: int) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = get_fps(video_path)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"scale='min({max_width},iw)':-2",
        str(out_dir / "frame_%06d.png"),
    ]
    result = await asyncio.to_thread(_run, cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr.decode(errors='ignore')[-2000:]}")
    return fps


async def frames_to_video(frames_dir: Path, fps: float, out_path: Path, original_video: Path | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    silent_path = out_path.with_name(out_path.stem + "_silent.mp4")

    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(silent_path),
    ]
    result = await asyncio.to_thread(_run, cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg render failed: {result.stderr.decode(errors='ignore')[-2000:]}")

    has_audio = False
    if original_video is not None:
        try:
            info = probe_video(original_video)
            has_audio = any(s.get("codec_type") == "audio" for s in info["streams"])
        except Exception:
            has_audio = False

    if has_audio and original_video is not None:
        cmd = [
            "ffmpeg", "-y", "-i", str(silent_path), "-i", str(original_video),
            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
            "-shortest", str(out_path),
        ]
        result = await asyncio.to_thread(_run, cmd)
        if result.returncode != 0:
            shutil.move(str(silent_path), str(out_path))
        else:
            silent_path.unlink(missing_ok=True)
    else:
        shutil.move(str(silent_path), str(out_path))


async def trim_video(src: Path, dst: Path, max_seconds: int) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(src), "-t", str(max_seconds), "-c", "copy", str(dst)]
    result = await asyncio.to_thread(_run, cmd)
    if result.returncode != 0:
        # fall back to re-encode trim if stream copy fails on this container
        cmd = ["ffmpeg", "-y", "-i", str(src), "-t", str(max_seconds), str(dst)]
        result = await asyncio.to_thread(_run, cmd)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg trim failed: {result.stderr.decode(errors='ignore')[-2000:]}")
