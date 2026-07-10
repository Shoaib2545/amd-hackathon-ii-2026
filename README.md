# Styleframe Agent

Generates style-specific captions for videos listed in `input/tasks.json` and writes results to `output/results.json`.

## Requirements

- Python 3.11+
- `ffmpeg` and `ffprobe` available on `PATH`
- A Fireworks API key
- A vision-capable Fireworks model

## Setup

1. Create a virtual environment.
2. Install dependencies.
3. Create a `.env` file in the project root.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Example `.env`:

```env
FIREWORKS_API_KEY=your_fireworks_key_here
FIREWORKS_MODEL=accounts/fireworks/models/qwen2p5-vl-32b-instruct
```

## Input Format

Put tasks in `input/tasks.json`:

```json
[
  {
    "task_id": "v1",
    "video_url": "https://example.com/video.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

## Run

From the project root:

```powershell
python -m app.main
```

Local runs use:

- input: `input/tasks.json`
- output: `output/results.json`
- temp work dir: `.tmp/styleframe-agent`

Container-style paths are still supported when `/input/tasks.json` and `/output/results.json` exist.

## Optional Environment Variables

- `STYLEFRAME_INPUT_PATH`: override input file path
- `STYLEFRAME_OUTPUT_PATH`: override output file path
- `STYLEFRAME_WORK_DIR`: override temp working directory
- `ALLOW_FALLBACK_CAPTIONS=1`: allow generic fallback captions if the model call fails

## Troubleshooting

- If every result looks the same, the model call is failing and fallback captions are being used.
- By default, the app now stops with an explicit error instead of silently writing repeated generic captions.
- If you get `Missing FIREWORKS_API_KEY`, check that `.env` exists and contains the key.
- If you get `ffmpeg` or `ffprobe` errors, install FFmpeg and ensure both commands are on `PATH`.
