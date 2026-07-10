import base64
import json
import os
from pathlib import Path

import requests

DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_FIREWORKS_MODEL = "accounts/fireworks/models/kimi-k2p6"


def get_fireworks_api_key() -> str:
    api_key = os.getenv("FIREWORKS_API_KEY")

    if not api_key:
        raise RuntimeError("Missing FIREWORKS_API_KEY environment variable.")

    return api_key


def get_fireworks_model() -> str:
    return os.getenv("FIREWORKS_MODEL", DEFAULT_FIREWORKS_MODEL).strip()


def get_fireworks_vision_model() -> str:
    return os.getenv("FIREWORKS_VISION_MODEL", get_fireworks_model()).strip()


def get_fireworks_text_model() -> str:
    return os.getenv("FIREWORKS_TEXT_MODEL", get_fireworks_model()).strip()


def get_fireworks_base_url() -> str:
    return os.getenv("FIREWORKS_BASE_URL", DEFAULT_FIREWORKS_BASE_URL).strip()


def encode_image_as_data_url(image_path: Path) -> str:
    with image_path.open("rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


def build_image_payload(prompt: str, image_paths: list[Path], max_tokens: int) -> dict:
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
        "model": get_fireworks_vision_model(),
        "max_tokens": max_tokens,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": content,
            },
        ],
    }

    return payload


def build_text_payload(prompt: str, max_tokens: int, temperature: float) -> dict:
    return {
        "model": get_fireworks_text_model(),
        "max_tokens": max_tokens,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }


def send_request(payload: dict) -> requests.Response:
    api_key = get_fireworks_api_key()

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    return requests.post(
        f"{get_fireworks_base_url()}/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=120,
    )


def extract_model_text(response: requests.Response) -> str:
    if response.status_code >= 400:
        raise RuntimeError(
            f"Fireworks API error {response.status_code}: {response.text[:1200]}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"]


def log_debug_response(debug_label: str | None, model_text: str) -> None:
    if os.getenv("DEBUG_FIREWORKS") == "1" and debug_label:
        print(f"----- {debug_label} START -----")
        print(model_text)
        print(f"----- {debug_label} END -----")


def chat_completion_with_images(
    prompt: str, image_paths: list[Path], debug_label: str | None = None
) -> str:
    payload = build_image_payload(prompt, image_paths, max_tokens=700)
    response = send_request(payload)
    model_text = extract_model_text(response)
    log_debug_response(debug_label, model_text)
    return model_text


def chat_completion_text_only(prompt: str, debug_label: str | None = None) -> str:
    payload = build_text_payload(prompt, max_tokens=500, temperature=0.1)
    response = send_request(payload)
    model_text = extract_model_text(response)
    log_debug_response(debug_label, model_text)
    return model_text
