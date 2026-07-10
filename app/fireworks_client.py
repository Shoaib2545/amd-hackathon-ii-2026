import base64
import json
import os
from pathlib import Path

import requests

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_MODEL = "accounts/fireworks/models/kimi-k2p6"


def get_fireworks_api_key() -> str:
    api_key = os.getenv("FIREWORKS_API_KEY")

    if not api_key:
        raise RuntimeError("Missing FIREWORKS_API_KEY environment variable.")

    return api_key


def encode_image_as_data_url(image_path: Path) -> str:
    with image_path.open("rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


def build_payload(prompt: str, image_paths: list[Path], use_json_mode: bool) -> dict:
    content = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_as_data_url(image_path)},
            }
        )

    payload = {
        "model": FIREWORKS_MODEL,
        "max_tokens": 900,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict JSON API. Return only valid JSON. Do not return markdown, explanations, or code fences.",
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    }

    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}

    return payload


def send_request(payload: dict) -> requests.Response:
    api_key = get_fireworks_api_key()

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    return requests.post(
        f"{FIREWORKS_BASE_URL}/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=120,
    )


def chat_completion_with_images(prompt: str, image_paths: list[Path]) -> str:
    # First try JSON mode.
    payload = build_payload(prompt, image_paths, use_json_mode=True)
    response = send_request(payload)

    # Some models/providers may reject response_format with vision.
    # If that happens, retry without JSON mode.
    if response.status_code >= 400 and "response_format" in response.text:
        payload = build_payload(prompt, image_paths, use_json_mode=False)
        response = send_request(payload)

    if response.status_code >= 400:
        raise RuntimeError(
            f"Fireworks API error {response.status_code}: {response.text[:1200]}"
        )

    data = response.json()
    model_text = data["choices"][0]["message"]["content"]

    if os.getenv("DEBUG_FIREWORKS") == "1":
        print("----- RAW FIREWORKS MODEL RESPONSE START -----")
        print(model_text)
        print("----- RAW FIREWORKS MODEL RESPONSE END -----")

    return model_text
