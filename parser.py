import re
from command_cache import get_command_cache


def extract_number(text):
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None


def parse_command(text):
    text = text.lower().strip()

    commands = get_command_cache()

    for cmd in commands:

        keyword = cmd["keyword"]

        if keyword in text:

            result = {
                "action": cmd["action"]
            }

            if cmd.get("direction"):
                result["direction"] = cmd["direction"]

            if cmd.get("hasValue", True):
                result["value"] = extract_number(text)

            return result

    return {"action": "UNKNOWN"}