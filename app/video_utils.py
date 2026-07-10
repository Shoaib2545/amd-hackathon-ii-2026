import json
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import requests


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https")


def run_command(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n" + " ".join(command) + "\n\nSTDERR:\n" + result.stderr
        )

    return result.stdout


def download_video(video_url: str, task_id: str, work_dir: Path) -> Path:
    if not video_url:
        raise ValueError(f"Missing video_url for task_id={task_id}")

    if not is_url(video_url):
        raise ValueError(
            f"video_url must be an http/https URL for now. Got: {video_url}"
        )

    videos_dir = work_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    video_path = videos_dir / f"{task_id}.mp4"

    print(f"Downloading video for task_id={task_id}")

    with requests.get(video_url, stream=True, timeout=60) as response:
        response.raise_for_status()

        with video_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded video is empty for task_id={task_id}")

    return video_path


def get_video_duration_seconds(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]

    output = run_command(command)
    data = json.loads(output)

    duration = float(data["format"]["duration"])

    if duration <= 0:
        raise RuntimeError(f"Invalid video duration: {duration}")

    return duration


def choose_frame_count(duration_seconds: float) -> int:
    if duration_seconds <= 35:
        return 6

    if duration_seconds <= 75:
        return 8

    return 10


def extract_representative_frames(
    video_path: Path, task_id: str, work_dir: Path
) -> list[Path]:
    duration = get_video_duration_seconds(video_path)
    frame_count = choose_frame_count(duration)

    frames_dir = work_dir / "frames" / task_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Extracting {frame_count} frames for task_id={task_id}, "
        f"duration={duration:.2f}s"
    )

    # Avoid the very first and very last frame.
    start_time = max(0.5, duration * 0.08)
    end_time = max(start_time + 0.5, duration * 0.92)

    if frame_count == 1:
        timestamps = [duration / 2]
    else:
        step = (end_time - start_time) / (frame_count - 1)
        timestamps = [start_time + step * index for index in range(frame_count)]

    frame_paths = []

    for index, timestamp in enumerate(timestamps, start=1):
        frame_path = frames_dir / f"frame_{index:03d}.jpg"

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(round(timestamp, 2)),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=768:-2",
            "-q:v",
            "5",
            str(frame_path),
        ]

        run_command(command)

        if frame_path.exists() and frame_path.stat().st_size > 0:
            frame_paths.append(frame_path)

    if not frame_paths:
        raise RuntimeError(f"No frames extracted for task_id={task_id}")

    return frame_paths
