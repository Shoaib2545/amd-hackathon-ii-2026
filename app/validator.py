ALLOWED_STYLES = {
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
}

def validate_task(task):
    if not isinstance(task, dict):
        raise ValueError("Each task must be an object.")

    if not task.get("task_id"):
        raise ValueError("Every task must include task_id.")

    if not task.get("video_url"):
        raise ValueError(f"Missing video_url for task_id={task.get('task_id')}")

    styles = task.get("styles") or list(ALLOWED_STYLES)

    if not isinstance(styles, list):
        raise ValueError(f"styles must be a list for task_id={task.get('task_id')}")

    for style in styles:
        if style not in ALLOWED_STYLES:
            raise ValueError(
                f"Unsupported style '{style}' for task_id={task.get('task_id')}"
            )

    return styles
def validate_results(results):
    if not isinstance(results, list):
        raise ValueError("Output must be a list.")

    for item in results:
        if not isinstance(item, dict):
            raise ValueError("Each result item must be an object.")

        if "task_id" not in item:
            raise ValueError("Missing task_id.")

        if "captions" not in item:
            raise ValueError(f"Missing captions for task_id={item.get('task_id')}.")

        captions = item["captions"]

        if not isinstance(captions, dict):
            raise ValueError("captions must be an object.")

        for style, caption in captions.items():
            if not isinstance(caption, str):
                raise ValueError(f"Caption for {style} must be a string.")

            if not caption.strip():
                raise ValueError(f"Caption for {style} cannot be empty.")

    return True
