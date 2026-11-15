import json
import re

_JSON_DECODER = json.JSONDecoder()

_THINK_NAMES = r"(?:think|thinking|thought|reasoning|reflection|scratchpad|cot)"

# (opening-tag regex, closing-tag regex)
_THINK_PAIRS = [
    (rf"<\s*{_THINK_NAMES}\b[^>]*>",            rf"<\s*/\s*{_THINK_NAMES}\s*>"),
    (r"\[\s*think(?:ing)?\s*\]",                r"\[\s*/\s*think(?:ing)?\s*\]"),
    (r"<\|\s*begin_of_thought\s*\|>",           r"<\|\s*end_of_thought\s*\|>"),
    (r"◁\s*think\s*▷",                          r"◁\s*/\s*think\s*▷"),
]

# Marker tokens to delete but whose content is kept (e.g. the solution part)
_MARKER_ONLY = re.compile(r"<\|\s*(?:begin|end)_of_solution\s*\|>", re.IGNORECASE)






def strip_thinking(text):
    """
    Remove chain-of-thought / reasoning sections from LLM output.
    Handles closed blocks, orphan closing tags (when the opening tag was
    in the prompt), and unclosed blocks (truncated generation).
    Returns the cleaned text (may be '' if the output was only thinking).
    """
    if not text:
        return text

    flags = re.DOTALL | re.IGNORECASE
    for open_re, close_re in _THINK_PAIRS:
        # 1) Fully closed blocks: <think> ... </think>
        text = re.sub(rf"{open_re}.*?{close_re}", "", text, flags=flags)

        # 2) Orphan closing tag: everything before it was thinking
        closers = list(re.finditer(close_re, text, flags=flags))
        if closers:
            text = text[closers[-1].end():]

        # 3) Unclosed opening tag: the rest is (truncated) thinking
        opener = re.search(open_re, text, flags=flags)
        if opener:
            text = text[:opener.start()]

    text = _MARKER_ONLY.sub("", text)
    return text.strip()


def extract_json_answer(text, key="answer"):
    """
    Robustly pull the value of `key` out of arbitrary LLM output.
    Handles: surrounding prose, ```json fences, multiple/nested braces,
    invalid escapes (e.g. SMILES backslashes), single quotes, trailing
    commas, and truncated output. Returns a string, or None if not found.
    """
    if not text:
        return None

    text = strip_thinking(text)
    def _clean(val):
        if isinstance(val, (list, tuple)):
            val = val[0] if val else None
        if val is None:
            return None
        return str(val).strip()

    # Escape lone backslashes that aren't valid JSON escapes (SMILES etc.)
    def _fix_escapes(s):
        return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)

    # 1) Try raw_decode starting at every '{'
    for variant in (text, _fix_escapes(text)):
        for m in re.finditer(r"\{", variant):
            try:
                obj, _ = _JSON_DECODER.raw_decode(variant[m.start():])
            except ValueError:
                continue
            if isinstance(obj, dict) and key in obj:
                out = _clean(obj[key])
                if out is not None:
                    return out

    # 2) Lenient regex: "answer": "...", 'answer': '...', or unquoted value
    k = re.escape(key)
    patterns = [
        rf'["\']?{k}["\']?\s*:\s*"((?:[^"\\]|\\.)*)"',    # double-quoted
        rf"[\"']?{k}[\"']?\s*:\s*'((?:[^'\\]|\\.)*)'",    # single-quoted
        rf'["\']?{k}["\']?\s*:\s*"([^"\n]*)$',            # truncated, no closing quote
        rf'["\']?{k}["\']?\s*:\s*([^,}}\n]+)',            # unquoted value
    ]
    for p in patterns:
        m = re.search(p, text, re.DOTALL | re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if val:
                return val

    return None