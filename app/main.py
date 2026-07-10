import json
import os
import sys
from pathlib import Path

from app.caption_agent import generate_captions_for_video
from app.validator import validate_results, validate_task
from app.video_utils import download_video, extract_representative_frames

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv_file():
    dotenv_path = PROJECT_ROOT / ".env"

    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()

        if not key or key in os.environ:
            continue

        os.environ[key] = value.strip().strip("'").strip('"')


def resolve_data_path(env_name: str, container_path: str, local_relative_path: str) -> Path:
    configured = os.getenv(env_name)

    if configured:
        return Path(configured)

    container_candidate = Path(container_path)

    if container_candidate.exists():
        return container_candidate

    return PROJECT_ROOT / local_relative_path


def resolve_work_dir() -> Path:
    configured = os.getenv("STYLEFRAME_WORK_DIR")

    if configured:
        return Path(configured)

    container_candidate = Path("/tmp/styleframe-agent")

    if container_candidate.anchor and container_candidate.parent.exists():
        return container_candidate

    return PROJECT_ROOT / ".tmp" / "styleframe-agent"


load_dotenv_file()

INPUT_PATH = resolve_data_path("STYLEFRAME_INPUT_PATH", "/input/tasks.json", "input/tasks.json")
OUTPUT_PATH = resolve_data_path(
    "STYLEFRAME_OUTPUT_PATH", "/output/results.json", "output/results.json"
)
WORK_DIR = resolve_work_dir()


def load_tasks():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]

    if isinstance(data, list):
        return data

    raise ValueError(
        "tasks.json must be either a list or an object with a 'tasks' list."
    )


def process_task(task):
    styles = validate_task(task)

    task_id = task["task_id"]
    video_url = task["video_url"]

    video_path = download_video(video_url, task_id, WORK_DIR)
    frame_paths = extract_representative_frames(video_path, task_id, WORK_DIR)

    captions = generate_captions_for_video(
        task_id=task_id,
        styles=styles,
        frame_paths=frame_paths,
    )

    return {
        "task_id": task_id,
        "captions": captions,
    }


def main():
    try:
        tasks = load_tasks()

        results = []

        for task in tasks:
            results.append(process_task(task))

        validate_results(results)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with OUTPUT_PATH.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)

        print(f"Successfully wrote results to {OUTPUT_PATH}")
        return 0

    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
