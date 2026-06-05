import re


def extract_number(text):
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None


def contains_any(text, keywords):
    return any(k in text for k in keywords)


def parse_command(text):
    text = text.lower().strip()

    # TAKE OFF
    if contains_any(text, [
        "take off",
        "takeoff",
        "start take off",
        "begin take off",
        "cất cánh",
        "bay lên"
    ]):
        return {"action": "TAKE_OFF"}

    # LAND
    if contains_any(text, [
        "land",
        "hạ cánh",
        "đáp xuống"
    ]):
        return {"action": "LAND"}

    # MOVE LEFT
    if contains_any(text, [
        "left",
        "trái",
        "qua trái",
        "sang trái"
    ]):
        return {
            "action": "MOVE",
            "direction": "LEFT",
            "value": extract_number(text)
        }

    # MOVE RIGHT
    if contains_any(text, [
        "right",
        "phải",
        "qua phải",
        "sang phải"
    ]):
        return {
            "action": "MOVE",
            "direction": "RIGHT",
            "value": extract_number(text)
        }

    # MOVE FORWARD
    if contains_any(text, [
        "forward",
        "tiến",
        "đi tới",
        "bay tới",
        "phía trước"
    ]):
        return {
            "action": "MOVE",
            "direction": "FORWARD",
            "value": extract_number(text)
        }

    # MOVE BACKWARD
    if contains_any(text, [
        "back",
        "backward",
        "lùi",
        "đi lùi",
        "phía sau"
    ]):
        return {
            "action": "MOVE",
            "direction": "BACKWARD",
            "value": extract_number(text)
        }

    # MOVE UP
    if contains_any(text, [
        "up",
        "lên",
        "bay lên",
        "cao lên"
    ]):
        return {
            "action": "MOVE",
            "direction": "UP",
            "value": extract_number(text)
        }

    # MOVE DOWN
    if contains_any(text, [
        "down",
        "xuống",
        "hạ xuống"
    ]):
        return {
            "action": "MOVE",
            "direction": "DOWN",
            "value": extract_number(text)
        }

    return {"action": "UNKNOWN"}