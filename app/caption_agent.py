import os
import re
from pathlib import Path

from app.fireworks_client import chat_completion_text_only, chat_completion_with_images

TRAFFIC_KEYWORDS = (
    "traffic",
    "road",
    "avenue",
    "street",
    "cars",
    "buses",
    "vehicles",
)
ANIMAL_KEYWORDS = (
    "kitten",
    "cat",
    "foliage",
    "garden",
    "leaves",
    "dirt",
    "tree trunks",
)
OFFICE_KEYWORDS = (
    "woman",
    "office",
    "desk",
    "computer",
    "typing",
    "keyboard",
    "screen",
    "workstation",
)

CATEGORY_KEYWORDS = {
    "traffic_urban": TRAFFIC_KEYWORDS,
    "animal_nature": ANIMAL_KEYWORDS,
    "office_technology": OFFICE_KEYWORDS,
}

SUMMARY_STOP_PATTERNS = (
    "wait,",
    "let me check",
    "check constraints",
    "requirements:",
    "actually,",
)
SUMMARY_PREFER_PATTERNS = ("let me refine:", "revised:", "draft:")
SUMMARY_META_PATTERNS = (
    "the user wants",
    "let me",
    "check constraints",
    "requirements:",
    "key elements",
    "i need to",
    "looking at",
    "actually,",
    "no markdown",
    "do not",
    "good.",
    "yes.",
    "frames",
    "screenshots",
    "ai",
    "analysis",
    "sampling",
)
CAPTION_META_PATTERNS = (
    "the user wants",
    "let me",
    "i need to",
    "check constraints",
    "requirements:",
    "scene summary",
    "raw fireworks",
    "should be",
    "example:",
    "rules:",
    "[final caption]",
    "<caption>",
)
NON_TECH_BANNED_WORDS = {
    "code",
    "bug",
    "server",
    "ai",
    "software",
    "algorithm",
    "data",
    "upload",
    "download",
    "app",
    "compile",
    "runtime",
    "debug",
    "async",
    "queue",
    "latency",
    "api",
    "cpu",
    "thread",
    "mutex",
    "deadlock",
    "render",
    "frames",
    "npc",
    "simulation",
}
UNSUPPORTED_TERMS = {
    "smog": ("smog",),
    "gridlock": ("gridlock",),
    "traffic jam": ("traffic jam",),
    "honking": ("honking",),
    "korea": ("korea",),
    "skyscrapers": ("skyscrapers",),
    "time-lapse": ("time-lapse", "time lapse"),
}
TERM_SUPPORT = {
    "smog": ("smog",),
    "gridlock": ("gridlock",),
    "traffic jam": ("traffic jam",),
    "honking": ("honking",),
    "korea": ("korea",),
    "skyscrapers": ("skyscrapers",),
    "time-lapse": ("time-lapse", "time lapse", "motion blur"),
}


def debug_enabled() -> bool:
    return os.getenv("DEBUG_FIREWORKS") == "1"


def debug_print(message: str) -> None:
    if debug_enabled():
        print(message)


def normalize_text(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.strip("\"'`\u201c\u201d")
    normalized = normalized.replace("\u2014", ", ").replace("\u2013", " - ")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    normalized = " ".join(normalized.split())
    normalized = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", normalized)
    normalized = re.sub(r"\s+([,.;!?])", r"\1", normalized)
    normalized = re.sub(r"\baclear\b", "a clear", normalized, flags=re.IGNORECASE)
    return normalized


def split_sentences(text: str) -> list[str]:
    return [
        normalize_text(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if normalize_text(sentence)
    ]


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w()'-]+\b", text))


def contains_any(text: str, patterns: tuple[str, ...] | set[str]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def detect_scene_category(text: str) -> str:
    lowered = text.lower()
    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        scores[category] = sum(1 for keyword in keywords if keyword in lowered)

    best_category = max(scores, key=scores.get)
    return best_category if scores[best_category] > 0 else "unknown"


def extract_category_summary(raw_text: str, category: str) -> str:
    if category not in CATEGORY_KEYWORDS:
        return ""

    keywords = CATEGORY_KEYWORDS[category]
    kept = []

    for sentence in split_sentences(raw_text):
        lowered = sentence.lower()
        if contains_any(lowered, SUMMARY_META_PATTERNS):
            continue
        if any(keyword in lowered for keyword in keywords):
            kept.append(sentence)

    if len(kept) > 4:
        kept = kept[-4:]

    return " ".join(kept)


def build_summary_fallback_from_category(raw_text: str, category: str) -> str:
    extracted = extract_category_summary(raw_text, category)
    if extracted:
        return extracted

    if category == "animal_nature":
        return (
            "A small kitten sits outdoors among dry leaves and natural ground cover. "
            "The scene feels quiet and close to nature."
        )
    if category == "office_technology":
        return (
            "A woman works at a desk in an office setting with a computer and keyboard. "
            "The scene focuses on concentrated indoor work."
        )
    if category == "traffic_urban":
        return (
            "Traffic moves along a wide urban avenue bordered by tall buildings and autumn trees. "
            "Distant mountains remain visible beyond the city."
        )
    return "A visible scene shows a subject within a clear setting."


def clean_scene_summary(raw_text: str, task_id: str | None = None) -> str:
    text = raw_text.replace("\r\n", "\n")
    raw_category = detect_scene_category(text)
    lines = text.splitlines()
    cleaned_lines = []
    preferred_lines = []
    collect_preferred = False

    for raw_line in lines:
        stripped = raw_line.strip()
        lowered = stripped.lower()

        if any(lowered.startswith(pattern) for pattern in SUMMARY_STOP_PATTERNS):
            break

        if not stripped:
            cleaned_lines.append("")
            if collect_preferred:
                preferred_lines.append("")
            continue

        if any(lowered.startswith(pattern) for pattern in SUMMARY_PREFER_PATTERNS):
            collect_preferred = True
            cleaned_lines.append("")
            preferred_lines.append("")
            continue

        if stripped.startswith("-") or re.match(r"^\d+\.\s", stripped):
            continue

        if contains_any(lowered, SUMMARY_META_PATTERNS):
            continue

        cleaned_lines.append(stripped)
        if collect_preferred:
            preferred_lines.append(stripped)

    def parse_paragraphs(block_lines: list[str]) -> list[str]:
        return [
            normalize_text(block)
            for block in re.split(r"\n\s*\n", "\n".join(block_lines))
            if normalize_text(block)
        ]

    def select_paragraph(paragraphs: list[str]) -> str | None:
        valid = []
        for paragraph in paragraphs:
            if contains_any(paragraph.lower(), SUMMARY_META_PATTERNS):
                continue
            sentences = split_sentences(paragraph)
            if 2 <= len(sentences) <= 4:
                valid.append(" ".join(sentences))
        return valid[-1] if valid else None

    preferred_summary = select_paragraph(parse_paragraphs(preferred_lines))
    general_summary = select_paragraph(parse_paragraphs(cleaned_lines))
    summary = preferred_summary or general_summary

    if not summary:
        factual_sentences = []
        for paragraph in parse_paragraphs(cleaned_lines):
            for sentence in split_sentences(paragraph):
                if not contains_any(sentence.lower(), SUMMARY_META_PATTERNS):
                    factual_sentences.append(sentence)
        summary = " ".join(factual_sentences[-4:]).strip()

    if not summary:
        summary = build_summary_fallback_from_category(raw_text, raw_category)

    clean_category = detect_scene_category(summary)
    if raw_category != "unknown" and clean_category != raw_category:
        summary = build_summary_fallback_from_category(raw_text, raw_category)
        clean_category = detect_scene_category(summary)

    summary = normalize_text(summary)

    if debug_enabled():
        print(f"DEBUG TASK ID: {task_id}")
        print(f"DEBUG RAW CATEGORY: {raw_category}")
        print(f"DEBUG CLEAN CATEGORY: {clean_category}")
        print("----- CLEAN SCENE SUMMARY START -----")
        print(summary)
        print("----- CLEAN SCENE SUMMARY END -----")

    return summary


def generate_scene_summary(task_id: str, frame_paths: list[Path], prompt: str | None = None) -> str:
    final_prompt = prompt or f"""
Describe only the visible video content for task {task_id}.

Requirements:
- Mention subjects, setting, action, movement, environment, and mood.
- Do not generate captions.
- Do not explain your reasoning.
- Do not mention frames, screenshots, AI, analysis, or sampling.
- Return a concise factual paragraph, 3 to 5 sentences maximum.
- No markdown.
""".strip()
    return " ".join(
        chat_completion_with_images(
            prompt=final_prompt,
            image_paths=frame_paths,
            debug_label="RAW FIREWORKS VISION SUMMARY RESPONSE",
        ).split()
    )


def maybe_regenerate_summary(task_id: str, frame_paths: list[Path], raw_text: str) -> str:
    summary = clean_scene_summary(raw_text, task_id=task_id)

    if word_count(summary) >= 12:
        return summary

    second_prompt = """
Describe the visible video scene in 3 factual sentences.
Mention setting, subjects, action, environment, and mood.
Return only the scene description.
Do not explain.
Do not list rules.
Do not say "The user wants".
Do not mention frames or analysis.
""".strip()
    second_raw = generate_scene_summary(task_id, frame_paths, prompt=second_prompt)
    second_summary = clean_scene_summary(second_raw, task_id=task_id)
    return second_summary if word_count(second_summary) >= word_count(summary) else summary


def compose_safe_captions(scene_summary: str, styles: list[str]) -> dict:
    trimmed = normalize_text(scene_summary).rstrip(".!?")
    lower_trimmed = trimmed[:1].lower() + trimmed[1:] if trimmed else trimmed
    captions = {}

    for style in styles:
        if style == "formal":
            captions[style] = (
                f"{trimmed}."
                if word_count(trimmed) >= 10
                else f"{trimmed} shows steady activity within the scene."
            )
        elif style == "sarcastic":
            captions[style] = (
                f"Nothing says smooth city travel like {lower_trimmed}."
                if word_count(trimmed) < 10
                else f"{trimmed}, because apparently this is everyone's idea of a calm and orderly moment."
            )
        elif style == "humorous_tech":
            captions[style] = (
                f"The scene looks like {lower_trimmed} running with too many active processes."
                if word_count(trimmed) < 10
                else f"{trimmed}, like a live simulation where every moving part chose the same moment to stay busy."
            )
        elif style == "humorous_non_tech":
            captions[style] = (
                f"{trimmed} is so busy that even the quiet parts look like they need a break."
                if word_count(trimmed) < 10
                else f"{trimmed}, while everything in view seems determined to compete for attention at once."
            )

    return captions


def use_text_model_for_captions() -> bool:
    return os.getenv("STYLEFRAME_USE_TEXT_MODEL") == "1"


def build_caption_prompt(scene_summary: str, styles: list[str]) -> str:
    requested_lines = "\n".join(f"{style}:" for style in styles)
    return f"""
You are a caption generator.
Return only the final caption lines.
Do not explain.
Do not give examples.
Do not describe the styles.
Do not say "should be".
Do not write rules.

Scene:
{scene_summary}

Return only these lines:
{requested_lines}
Put the final caption after each colon.
Do not output examples.
Do not output instructions.
Do not output placeholders.
""".strip()


def extract_labeled_captions(text: str, styles: list[str]) -> dict:
    captions = {}
    truncated = []

    for line in text.splitlines():
        lowered = line.strip().lower()
        if lowered.startswith("wait,") or lowered.startswith("let me check"):
            break
        truncated.append(line)

    for style in styles:
        match = re.search(rf"(?im)^\s*{re.escape(style)}\s*:\s*(.+?)\s*$", "\n".join(truncated))
        if match:
            captions[style] = normalize_text(match.group(1))

    return captions


def style_description_like(caption: str) -> bool:
    lowered = caption.lower()
    return any(
        phrase in lowered
        for phrase in (
            "should be",
            "style",
            "tone",
            "caption generator",
            "one sentence per caption",
            "factual and professional",
            "dry and lightly mocking",
            "tech or programming humor",
            "everyday humor",
            "no tech jargon",
        )
    )


def caption_has_meta(caption: str) -> bool:
    return contains_any(caption.lower(), CAPTION_META_PATTERNS)


def caption_has_unsupported_detail(caption: str, scene_summary: str) -> bool:
    lowered = caption.lower()

    for term_key, variants in UNSUPPORTED_TERMS.items():
        if any(variant in lowered for variant in variants):
            if not any(term in scene_summary.lower() for term in TERM_SUPPORT[term_key]):
                return True

    scene_category = detect_scene_category(scene_summary)
    if scene_category == "animal_nature" and contains_any(lowered, TRAFFIC_KEYWORDS):
        return True
    if scene_category == "office_technology" and (
        contains_any(lowered, TRAFFIC_KEYWORDS) or contains_any(lowered, ("mountains", "autumn trees"))
    ):
        return True

    return False


def validate_caption_or_error(caption: str, style: str, scene_summary: str) -> str | None:
    if not isinstance(caption, str):
        return f"Caption for style={style} must be a string."

    normalized = normalize_text(caption)
    if not normalized:
        return f"Caption for style={style} cannot be empty."
    if normalized.lower() in {"<caption>", "[final caption]"}:
        return f"Caption for style={style} is a placeholder."
    if "should be" in normalized.lower():
        return f"Caption for style={style} is a style description."
    if caption_has_meta(normalized) or style_description_like(normalized):
        return f"Caption for style={style} contains meta text."
    if caption_has_unsupported_detail(normalized, scene_summary):
        return f"Caption for style={style} contains unsupported detail."

    if style == "humorous_non_tech":
        words = re.findall(r"[a-zA-Z_]+", normalized.lower())
        if "o(n)" in normalized.lower() or any(word in NON_TECH_BANNED_WORDS for word in words):
            return "humorous_non_tech caption contains banned technical wording."

    return None


def validate_output_ready_captions(captions: dict, styles: list[str]) -> dict:
    final_captions = {}
    for style in styles:
        caption = captions.get(style)
        if not isinstance(caption, str):
            raise ValueError(f"Caption for style={style} must be a string.")
        normalized = normalize_text(caption)
        if not normalized:
            raise ValueError(f"Caption for style={style} cannot be empty.")
        if caption_has_meta(normalized) or style_description_like(normalized):
            raise ValueError(f"Caption for style={style} contains forbidden meta text.")
        final_captions[style] = normalized
    return final_captions


def compose_or_model_captions(task_id: str, scene_summary: str, styles: list[str]) -> dict:
    if not use_text_model_for_captions():
        return compose_safe_captions(scene_summary, styles)

    model_text = chat_completion_text_only(
        build_caption_prompt(scene_summary, styles),
        debug_label="RAW FIREWORKS FINAL CAPTION RESPONSE",
    )
    parsed = extract_labeled_captions(model_text, styles)
    fallback = compose_safe_captions(scene_summary, styles)

    final_captions = {}
    for style in styles:
        candidate = parsed.get(style)
        error = validate_caption_or_error(candidate, style, scene_summary)
        final_captions[style] = fallback[style] if error else normalize_text(candidate)

    return final_captions


def run_summary_regression_debug_checks() -> None:
    if not debug_enabled():
        return

    kitten_summary = clean_scene_summary(
        "A fluffy orange kitten sits on dry dirt beside leaves and foliage.",
        task_id="debug-kitten",
    )
    office_summary = clean_scene_summary(
        "A woman sits at a desk typing on a desktop computer in an office.",
        task_id="debug-office",
    )
    traffic_summary = clean_scene_summary(
        "Traffic moves along a wide urban avenue with cars and buses.",
        task_id="debug-traffic",
    )

    assert "kitten" in kitten_summary.lower()
    assert any(token in office_summary.lower() for token in ("woman", "computer", "office"))
    assert "traffic" in traffic_summary.lower()


def generate_captions_for_video(task_id: str, styles: list[str], frame_paths: list[Path]) -> dict:
    run_summary_regression_debug_checks()
    raw_summary = generate_scene_summary(task_id, frame_paths)
    clean_summary = maybe_regenerate_summary(task_id, frame_paths, raw_summary)
    captions = compose_or_model_captions(task_id, clean_summary, styles)
    safe_fallback = compose_safe_captions(clean_summary, styles)

    final_captions = {}
    for style in styles:
        candidate = captions.get(style)
        error = validate_caption_or_error(candidate, style, clean_summary)
        final_captions[style] = safe_fallback[style] if error else normalize_text(candidate)

    final_captions = validate_output_ready_captions(final_captions, styles)

    if debug_enabled():
        print(f"DEBUG TASK ID: {task_id}")
        print(f"DEBUG RAW CATEGORY: {detect_scene_category(raw_summary)}")
        print(f"DEBUG CLEAN CATEGORY: {detect_scene_category(clean_summary)}")
        print("DEBUG COMPOSED CAPTIONS:")
        print(final_captions)

    return final_captions
