"""Real-world GSM8K graders for the E010 gold-free scan (research/paper/). Each grader maps
(response text, reference answer string) -> {0, 1}.

lm_eval_strict / lm_eval_flexible: the two GSM8K filters of lm-evaluation-harness
  (strict: first "#### <number>"; flexible: last number-like match), scored by exact match after
  removing "," "$" and a trailing "." (the harness's regexes_to_ignore), case-insensitive.
math_verify: Hugging Face math-verify (parse + verify), if installed.
anywhere: the reference number appears anywhere (the loose-extraction bug class).
"""

import re
from collections.abc import Callable

from vdyn.e007 import task as tk

_STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
_FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_IGNORE = [re.compile(r","), re.compile(r"\$"), re.compile(r"(?s).*#### "), re.compile(r"\.$")]


def _clean(s: str) -> str:
    for r in _IGNORE:
        s = r.sub("", s)
    return s.strip().lower()


def lm_eval_strict(text: str, gold: str) -> float:
    m = _STRICT.search(text)
    return float(m is not None and _clean(m.group(1)) == _clean(gold))


def lm_eval_flexible(text: str, gold: str) -> float:
    ms = _FLEX.findall(text)
    if not ms:
        return 0.0
    last = next(x for x in ms[-1] if x)
    return float(_clean(last) == _clean(gold))


def anywhere(text: str, gold: str) -> float:
    return float(gold in tk.numbers_in(text))


def math_verify_grader() -> Callable[[str, str], float] | None:
    try:
        from math_verify import parse, verify
    except ImportError:
        return None

    def grade(text: str, gold: str) -> float:
        try:
            return float(bool(verify(parse(gold), parse(text))))
        except Exception:  # math-verify raises on unparsable input; count as reject
            return 0.0

    return grade
