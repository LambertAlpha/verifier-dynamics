"""GSM8K for E007: prompt, gold answers and the strict checker (the last \\boxed{...} must equal
the reference number after normalization)."""

import re

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gold_answer(answer_field: str) -> str:
    return normalize(answer_field.split("####")[-1])


def extract_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def normalize(s: str) -> str:
    t = s.strip().replace("\\$", "").replace("$", "").replace(",", "").replace("%", "").strip()
    t = t.rstrip(".").strip()
    try:
        v = float(t)
    except ValueError:
        return t
    if v == int(v):
        return str(int(v))
    return t


def gold_reward(text: str, gold: str) -> float:
    box = extract_boxed(text)
    return 1.0 if box is not None and normalize(box) == gold else 0.0


def numbers_in(text: str) -> set[str]:
    return {normalize(m) for m in _NUM.findall(text)}
