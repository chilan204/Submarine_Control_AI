from core.command_cache import get_command_cache


def _normalize(text):
    return " ".join((text or "").lower().split())


def parse_command(text):
    normalized_text = _normalize(text)
    if not normalized_text:
        return None

    for command in get_command_cache():
        # Vosk is already constrained to database phrases. Exact matching keeps
        # malformed/combined transcripts from becoming physical AUV commands.
        if _normalize(command["keyword"]) != normalized_text:
            continue

        result = {"action": command["action"]}
        if command.get("direction"):
            result["direction"] = command["direction"]
        return result

    return None
