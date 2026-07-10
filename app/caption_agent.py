import json
import re
from pathlib import Path

from app.fireworks_client import chat_completion_with_images

STYLE_DESCRIPTIONS = {
    "formal": "Professional, objective, factual tone.",
    "sarcastic": "Dry, ironic, lightly mocking, but still accurate.",
    "humorous_tech": "Funny, with technology or programming references.",
    "humorous_non_tech": "Funny everyday humour with no technical jargon.",
}


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()

    # Remove markdown fences if the model ignores instructions.
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")

    json_text = cleaned[start : end + 1]
    return json.loads(json_text)


def build_prompt(task_id: str, styles: list[str]) -> str:
    style_lines = []

    for style in styles:
        style_lines.append(f"{style}: {STYLE_DESCRIPTIONS[style]}")

    requested_styles = ", ".join(styles)
    styles_text = "\n".join(style_lines)

    return f"""
You are generating captions for one video.

You will receive sampled frames from the video.
Infer the main visible scene, subjects, setting, and action.
Then generate one caption for each requested style.

TASK_ID:
{task_id}

REQUESTED_STYLES:
{requested_styles}

STYLE_RULES:
{styles_text}

STRICT OUTPUT RULES:
Return only one valid JSON object.
Do not include markdown.
Do not include ```json.
Do not include explanations.
Do not include notes.
Do not mention frames, screenshots, sampling, model, AI, or analysis.
Each caption must be one sentence.
Each caption must be accurate to the visible video.
Do not invent speech, brands, names, or exact places unless clearly visible.
humorous_non_tech must not contain coding, software, AI, app, bug, server, data, algorithm, upload, download, or programming words.

Return this exact JSON shape:
{{
  "captions": {{
    "formal": "one sentence",
    "sarcastic": "one sentence",
    "humorous_tech": "one sentence",
    "humorous_non_tech": "one sentence"
  }}
}}

Only include these requested caption keys:
{requested_styles}
""".strip()


def fallback_captions(task_id: str, styles: list[str]) -> dict:
    captions = {}

    for style in styles:
        if style == "formal":
            captions[style] = (
                "The video shows a visible scene with subjects and surroundings requiring factual description."
            )
        elif style == "sarcastic":
            captions[style] = (
                "A scene unfolds on camera, clearly committed to making ordinary motion look important."
            )
        elif style == "humorous_tech":
            captions[style] = (
                "The video processes real-world activity like a visual loop waiting for a clean output."
            )
        elif style == "humorous_non_tech":
            captions[style] = (
                "The video captures a moment that seems ready for its small claim to fame."
            )
        else:
            captions[style] = f"Caption generated for task {task_id}."

    return captions


def generate_captions_for_video(
    task_id: str, styles: list[str], frame_paths: list[Path]
) -> dict:
    prompt = build_prompt(task_id=task_id, styles=styles)

    try:
        model_text = chat_completion_with_images(prompt=prompt, image_paths=frame_paths)
        data = extract_json_object(model_text)

        captions = data.get("captions")

        if not isinstance(captions, dict):
            raise ValueError("Model JSON missing captions object.")

        final_captions = {}

        for style in styles:
            caption = captions.get(style)

            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(
                    f"Model response missing valid caption for style={style}"
                )

            final_captions[style] = caption.strip()

        return final_captions

    except Exception as error:
        print(f"WARNING: Caption model failed for task_id={task_id}: {error}")
        return fallback_captions(task_id=task_id, styles=styles)
