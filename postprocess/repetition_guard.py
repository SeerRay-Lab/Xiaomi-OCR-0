#!/usr/bin/env python3

from collections import Counter
from typing import Dict, List, Optional


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _longest_same_character_run(text: str) -> Dict[str, object]:
    best_character = ""
    best_length = 0
    current_character = ""
    current_length = 0
    for character in text:
        if character.isspace():
            current_character = ""
            current_length = 0
        elif character == current_character:
            current_length += 1
        else:
            current_character = character
            current_length = 1
        if current_length > best_length:
            best_character = current_character
            best_length = current_length
    return {"character": best_character, "length": best_length}


def _repeating_suffix(
    text: str,
    min_repeats: int,
    min_repeated_chars: int,
    min_ratio: float,
    max_unit_chars: int,
) -> Optional[Dict[str, object]]:
    if len(text) < min_repeated_chars:
        return None

    best = None
    max_unit = min(max_unit_chars, len(text) // min_repeats)
    for unit_length in range(1, max_unit + 1):
        unit = text[-unit_length:]
        cursor = len(text) - unit_length
        count = 1
        while cursor >= unit_length and text[cursor - unit_length:cursor] == unit:
            cursor -= unit_length
            count += 1

        repeated_chars = count * unit_length
        ratio = repeated_chars / len(text)
        if (
            count < min_repeats
            or repeated_chars < min_repeated_chars
            or ratio < min_ratio
        ):
            continue
        candidate = {
            "unit": unit,
            "unit_chars": unit_length,
            "repeats": count,
            "repeated_chars": repeated_chars,
            "ratio": ratio,
        }
        if best is None or repeated_chars > best["repeated_chars"]:
            best = candidate
    return best


def _dominant_repeated_line(
    text: str,
    min_repeats: int,
    min_repeated_chars: int,
    min_ratio: float,
) -> Optional[Dict[str, object]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < min_repeats:
        return None

    line, count = Counter(lines).most_common(1)[0]
    repeated_chars = len(line) * count
    ratio = count / len(lines)
    if (
        count < min_repeats
        or repeated_chars < min_repeated_chars
        or ratio < min_ratio
    ):
        return None
    return {
        "line": line,
        "repeats": count,
        "total_lines": len(lines),
        "repeated_chars": repeated_chars,
        "ratio": ratio,
    }


def analyze_repetition(
    text: str,
    finish_reason: Optional[str] = None,
    same_character_run: int = 128,
    min_repeats: int = 8,
    min_repeated_chars: int = 512,
    min_ratio: float = 0.5,
    max_unit_chars: int = 256,
) -> Dict[str, object]:
    normalized = _normalize_whitespace(text)
    non_whitespace_length = len(normalized.replace(" ", ""))
    reasons: List[str] = []

    if finish_reason == "length":
        reasons.append("finish_reason_length")

    same_character = _longest_same_character_run(text)
    same_character_ratio = (
        same_character["length"] / non_whitespace_length
        if non_whitespace_length
        else 0.0
    )
    if same_character["length"] >= same_character_run and (
        same_character["length"] >= min_repeated_chars
        or same_character_ratio >= min_ratio
    ):
        reasons.append("same_character_run")

    suffix = _repeating_suffix(
        normalized,
        min_repeats=min_repeats,
        min_repeated_chars=min_repeated_chars,
        min_ratio=min_ratio,
        max_unit_chars=max_unit_chars,
    )
    if suffix is not None:
        reasons.append("repeating_suffix")

    repeated_line = _dominant_repeated_line(
        text,
        min_repeats=min_repeats,
        min_repeated_chars=min_repeated_chars,
        min_ratio=max(min_ratio, 0.8),
    )
    if repeated_line is not None:
        reasons.append("dominant_repeated_line")

    return {
        "suspicious": bool(reasons),
        "reasons": reasons,
        "characters": len(text),
        "non_whitespace_characters": non_whitespace_length,
        "finish_reason": finish_reason,
        "same_character": same_character,
        "same_character_ratio": same_character_ratio,
        "repeating_suffix": suffix,
        "dominant_repeated_line": repeated_line,
    }
