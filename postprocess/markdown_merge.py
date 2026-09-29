#!/usr/bin/env python3
"""Configurable Markdown assembly from cached per-region recognition results."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

try:
    from wordfreq import zipf_frequency
except Exception:
    zipf_frequency = None


WORD_VALIDATION_BACKEND = "wordfreq" if zipf_frequency is not None else "regex-fallback"

LABEL_VISUALIZATION_MAPPING = {
    "image": ["chart", "image"],
    "text": [
        "abstract",
        "algorithm",
        "content",
        "doc_title",
        "figure_title",
        "paragraph_title",
        "reference",
        "reference_content",
        "text",
        "vertical_text",
        "vision_footnote",
        "seal",
        "formula_number",
    ],
    "table": ["table"],
    "formula": ["display_formula", "inline_formula"],
    "abandon": [
        "header",
        "footer",
        "number",
        "footnote",
        "aside_text",
        "footer_image",
        "header_image",
    ],
}

TITLE_RE_PATTERN = re.compile(
    r"^\s*((?:[1-9][0-9]*(?:\.[1-9][0-9]*)*[\.、]?|"
    r"[\(（](?:[1-9][0-9]*|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+)[\)）]|"
    r"[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+[、\.]?|"
    r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\s)))(\s*)(.*)$"
)


@dataclass(frozen=True)
class MergeOptions:
    text_profile: str = "v2"
    title_profile: str = "v2"
    hyphen_merge: str = "v2"
    text_compaction: str = "off"
    compact_line_threshold: int = 64
    compact_region_threshold: int = 128
    compact_min_horizontal_overlap: float = 0.45
    compact_max_vertical_gap: float = 80.0
    bullet_repair: bool = True
    include_images: bool = False
    include_footnote: bool = False
    include_aside_text: bool = False
    include_reference: bool = True
    inline_formula_policy: str = "display"
    inline_parent_min_cover: float = 0.25
    inline_parent_min_x_overlap: float = 0.5
    inline_parent_max_gap: float = 32.0
    inline_parent_score_margin: float = 0.1
    inline_parent_boundary_band: float = 24.0
    validate_tables: bool = True

    def __post_init__(self) -> None:
        if self.text_profile not in {"v2", "paddle"}:
            raise ValueError(f"unsupported text_profile: {self.text_profile}")
        if self.title_profile not in {"v2", "numbered"}:
            raise ValueError(f"unsupported title_profile: {self.title_profile}")
        if self.hyphen_merge not in {"v2", "adjacent", "off"}:
            raise ValueError(f"unsupported hyphen_merge: {self.hyphen_merge}")
        if self.text_compaction not in {"off", "adaptive"}:
            raise ValueError(f"unsupported text_compaction: {self.text_compaction}")
        if self.inline_formula_policy not in {
            "display",
            "dedupe",
            "parent-aware",
            "paddle-overlap",
        }:
            raise ValueError(
                f"unsupported inline_formula_policy: {self.inline_formula_policy}"
            )
        if self.compact_line_threshold < 2:
            raise ValueError("compact_line_threshold must be >= 2")
        if self.compact_region_threshold < 2:
            raise ValueError("compact_region_threshold must be >= 2")
        if not 0 < self.compact_min_horizontal_overlap <= 1:
            raise ValueError("compact_min_horizontal_overlap must be in (0, 1]")
        if self.compact_max_vertical_gap < 0:
            raise ValueError("compact_max_vertical_gap must be >= 0")
        if not 0 < self.inline_parent_min_cover <= 1:
            raise ValueError("inline_parent_min_cover must be in (0, 1]")
        if not 0 < self.inline_parent_min_x_overlap <= 1:
            raise ValueError("inline_parent_min_x_overlap must be in (0, 1]")
        if self.inline_parent_max_gap < 0:
            raise ValueError("inline_parent_max_gap must be >= 0")
        if self.inline_parent_score_margin < 0:
            raise ValueError("inline_parent_score_margin must be >= 0")
        if self.inline_parent_boundary_band < 0:
            raise ValueError("inline_parent_boundary_band must be >= 0")


@dataclass
class MergeOutput:
    markdown: str
    events: list[dict[str, Any]]
    stats: dict[str, Any]


def map_label(
    label: str,
    include_footnote: bool = False,
    include_aside_text: bool = False,
    include_reference: bool = True,
) -> str:
    if label == "footnote" and include_footnote:
        return "text"
    if label == "aside_text" and include_aside_text:
        return "text"
    if label == "reference" and not include_reference:
        return "abandon"
    for mapped_type, labels in LABEL_VISUALIZATION_MAPPING.items():
        if label in labels:
            return mapped_type
    return label


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _preview(content: Any, limit: int = 500) -> Any:
    if not isinstance(content, str) or len(content) <= limit:
        return content
    return content[:limit] + f"...<{len(content) - limit} chars omitted>"


def _make_event(
    pass_name: str,
    action: str,
    source_indices: list[int],
    before: str | None = None,
    after: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "pass": pass_name,
        "action": action,
        "source_indices": source_indices,
    }
    if before is not None:
        event["before"] = _preview(before)
        event["before_sha256"] = _sha256_text(before)
    if after is not None:
        event["after"] = _preview(after)
        event["after_sha256"] = _sha256_text(after)
    event.update(details)
    return event


def _find_consecutive_repeat(s: str, min_unit_len: int = 10, min_repeats: int = 10) -> str | None:
    length = len(s)
    if length < min_unit_len * min_repeats:
        return None
    max_unit_len = length // min_repeats
    if max_unit_len < min_unit_len:
        return None
    pattern = re.compile(
        r"(.{" + str(min_unit_len) + "," + str(max_unit_len) + r"}?)\1{"
        + str(min_repeats - 1)
        + ",}",
        re.DOTALL,
    )
    match = pattern.search(s)
    if match:
        return s[: match.start()] + match.group(1)
    return None


def _clean_repeated_content(
    content: str,
    min_len: int = 10,
    min_repeats: int = 10,
    line_threshold: int = 10,
) -> str:
    stripped_content = content.strip()
    if not stripped_content:
        return content
    if len(stripped_content) > min_len * min_repeats:
        result = _find_consecutive_repeat(
            stripped_content,
            min_unit_len=min_len,
            min_repeats=min_repeats,
        )
        if result is not None:
            return result
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    total_lines = len(lines)
    if total_lines >= line_threshold and lines:
        common, count = Counter(lines).most_common(1)[0]
        if count >= line_threshold and count / total_lines >= 0.8:
            for line_index, line in enumerate(lines):
                if line != common:
                    continue
                consecutive = sum(
                    1
                    for offset in range(line_index, min(line_index + 3, len(lines)))
                    if lines[offset] == common
                )
                if consecutive < 3:
                    continue
                original_lines = content.split("\n")
                non_empty_count = 0
                for original_index, original_line in enumerate(original_lines):
                    if original_line.strip():
                        non_empty_count += 1
                        if non_empty_count == line_index + 1:
                            return "\n".join(original_lines[: original_index + 1])
                break
    return content


def _clean_formula_number(number_content: str) -> str:
    number_clean = number_content.strip()
    if number_clean.startswith("(") and number_clean.endswith(")"):
        number_clean = number_clean[1:-1]
    elif number_clean.startswith("（") and number_clean.endswith("）"):
        number_clean = number_clean[1:-1]
    return number_clean


def _normalize_inline_formula(content: str) -> str:
    inline_formula_re = re.compile(r"(?<!\$)\$\s*((?:[^$\\]|\\.)+?)\s*\$(?!\$)")
    if "$" not in content:
        return content
    parts = []
    last_end = 0
    for match in inline_formula_re.finditer(content):
        formula = match.group(1).strip()
        if not formula:
            continue
        start, end = match.start(), match.end()
        before = content[last_end:start]
        if before and before[-1].isalnum():
            before += " "
        parts.append(before)
        parts.append(f"${formula}$")
        if end < len(content) and content[end].isalnum():
            parts.append(" ")
        last_end = end
    if not parts:
        return content
    parts.append(content[last_end:])
    return "".join(parts)


def _clean_content(content: Any) -> str:
    if content is None:
        return ""
    cleaned = str(content)
    cleaned = re.sub(r"^(\\t)+", "", cleaned).lstrip()
    cleaned = re.sub(r"(\\t)+$", "", cleaned).rstrip()
    cleaned = re.sub(r"(\.)\1{2,}", r"\1\1\1", cleaned)
    cleaned = re.sub(r"(·)\1{2,}", r"\1\1\1", cleaned)
    cleaned = re.sub(r"(_)\1{2,}", r"\1\1\1", cleaned)
    cleaned = re.sub(r"(\\_)\1{2,}", r"\1\1\1", cleaned)
    if len(cleaned) >= 2048:
        cleaned = _clean_repeated_content(cleaned)
    cleaned = _normalize_inline_formula(cleaned)
    return cleaned.strip()


def _format_numbered_title(content: str) -> str:
    title = content
    match = TITLE_RE_PATTERN.match(title)
    if match:
        title = match.group(1).strip() + " " + match.group(3).lstrip()
    title = title.rstrip(".")
    level = title.count(".") + 1 if "." in title else 1
    markdown_level = min(level + 1, 6)
    return "#" * markdown_level + " " + title


def _apply_title_profile(content: str, native_label: str, title_profile: str) -> str:
    if native_label == "doc_title":
        normalized = re.sub(r"^#+\s*", "", content)
        return "# " + normalized
    if native_label != "paragraph_title":
        return content
    normalized = content
    if normalized.startswith("- ") or normalized.startswith("* "):
        normalized = normalized[2:].lstrip()
    normalized = re.sub(r"^#+\s*", "", normalized).lstrip()
    if title_profile == "numbered":
        return _format_numbered_title(normalized)
    return "## " + normalized


def _format_formula(content: Any) -> str:
    if content is None:
        return ""
    formula = str(content)
    if formula.startswith("$$") and formula.endswith("$$"):
        formula = formula.strip()
    else:
        formula = _clean_content(formula)
    if formula.startswith("$$") or formula.startswith("\\[") or formula.startswith("\\("):
        formula = formula[2:].strip()
    if formula.endswith("$$") or formula.endswith("\\]") or formula.endswith("\\)"):
        formula = formula[:-2].strip()
    return "$$\n" + formula + "\n$$"


def _format_table(content: Any) -> str:
    if content is None:
        return ""
    table = str(content)
    if table.startswith("<table") and table.endswith("</table>"):
        return table.strip()
    return _clean_content(table)


def _format_text_v2(content: Any, native_label: str, title_profile: str) -> str:
    formatted = _clean_content(content)
    formatted = _apply_title_profile(formatted, native_label, title_profile)
    if formatted.startswith("```") and not formatted.endswith("```"):
        formatted += "\n```"
    if formatted.startswith("·") or formatted.startswith("•") or formatted.startswith("* "):
        formatted = "- " + formatted[1:].lstrip()
    match = re.match(r"^(\(|\（)(\d+|[A-Za-z])(\)|\）)(.*)$", formatted)
    if match:
        _, symbol, _, rest = match.groups()
        formatted = f"({symbol}) {rest.lstrip()}"
    match = re.match(r"^(\d+|[A-Za-z])(\.|\)|\）)(.*)$", formatted)
    if match:
        symbol, separator, rest = match.groups()
        separator = ")" if separator == "）" else separator
        formatted = f"{symbol}{separator} {rest.lstrip()}"
    return re.sub(r"(?<!\n)\n(?!\n)", "\n\n", formatted)


def _collapse_soft_newlines(content: str) -> str:
    return content.replace("-\n", "").replace("\n", " ")


def _format_text_paddle(content: Any, native_label: str, title_profile: str) -> str:
    formatted = str(content or "")
    if native_label in {"text", "vertical_text", "reference_content", "vision_footnote"}:
        formatted = formatted.replace("\n\n", "\n").replace("\n", "\n\n")
    elif native_label == "content":
        formatted = formatted.replace("-\n", "  \n").replace("\n", "  \n")
    elif native_label == "algorithm":
        formatted = formatted.strip("\n")
    elif native_label in {
        "doc_title",
        "paragraph_title",
        "figure_title",
        "abstract",
    }:
        formatted = _collapse_soft_newlines(formatted)
    formatted = formatted.strip()
    return _apply_title_profile(formatted, native_label, title_profile)


def _format_content(
    content: Any,
    label: str,
    native_label: str,
    options: MergeOptions,
) -> str:
    if label == "table":
        return _format_table(content)
    if label == "formula":
        return _format_formula(content)
    if options.text_profile == "paddle":
        return _format_text_paddle(content, native_label, options.title_profile)
    return _format_text_v2(content, native_label, options.title_profile)


def _is_likely_valid_merged_word(merged_word: str) -> bool:
    token = merged_word.strip().lower()
    if not token:
        return False
    if zipf_frequency is not None:
        try:
            return zipf_frequency(token, "en") >= 2.5
        except Exception:
            pass
    if not re.fullmatch(r"[a-z][a-z'\-]{2,30}", token):
        return False
    if "--" in token or "''" in token:
        return False
    return True


def _merge_text_blocks(
    results: list[dict[str, Any]],
    mode: str,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not results or mode == "off":
        return results
    merged_results = []
    skip_indices = set()
    for index, block in enumerate(results):
        if index in skip_indices:
            continue
        if block.get("label") != "text":
            merged_results.append(block)
            continue
        content = block.get("content", "")
        if not isinstance(content, str):
            merged_results.append(block)
            continue
        content_stripped = content.rstrip()
        if not content_stripped or not content_stripped.endswith("-"):
            merged_results.append(block)
            continue
        merged = False
        candidate_indices = range(index + 1, len(results))
        if mode == "adjacent":
            candidate_indices = range(index + 1, min(index + 2, len(results)))
        for next_index in candidate_indices:
            if results[next_index].get("label") != "text":
                if mode == "adjacent":
                    break
                continue
            next_content = results[next_index].get("content", "")
            if isinstance(next_content, str):
                next_stripped = next_content.lstrip()
                if next_stripped and next_stripped[0].islower():
                    words_before = content_stripped[:-1].split()
                    next_words = next_stripped.split()
                    if words_before and next_words:
                        merged_word = words_before[-1] + next_words[0]
                        if _is_likely_valid_merged_word(merged_word):
                            merged_content = content_stripped[:-1] + next_content.lstrip()
                            merged_block = deepcopy(block)
                            merged_block["content"] = merged_content
                            merged_block["source_indices"] = (
                                block.get("source_indices", [])
                                + results[next_index].get("source_indices", [])
                            )
                            merged_results.append(merged_block)
                            skip_indices.add(next_index)
                            events.append(
                                _make_event(
                                    "hyphen_merge",
                                    "merge",
                                    merged_block["source_indices"],
                                    before=content + "\n\n" + next_content,
                                    after=merged_content,
                                    mode=mode,
                                    merged_word=merged_word,
                                )
                            )
                            merged = True
                    break
        if not merged:
            merged_results.append(block)
    for index, block in enumerate(merged_results):
        block["index"] = index
    return merged_results


def _format_bullet_points(
    results: list[dict[str, Any]],
    events: list[dict[str, Any]],
    left_align_threshold: float = 10.0,
) -> list[dict[str, Any]]:
    if len(results) < 3:
        return results
    for index in range(1, len(results) - 1):
        current_block = results[index]
        previous_block = results[index - 1]
        next_block = results[index + 1]
        if current_block.get("native_label") != "text":
            continue
        if previous_block.get("native_label") != "text" or next_block.get("native_label") != "text":
            continue
        current_content = current_block.get("content", "")
        if current_content.startswith("- "):
            continue
        previous_content = previous_block.get("content", "")
        next_content = next_block.get("content", "")
        if not previous_content.startswith("- ") or not next_content.startswith("- "):
            continue
        current_bbox = current_block.get("bbox_2d", [])
        previous_bbox = previous_block.get("bbox_2d", [])
        next_bbox = next_block.get("bbox_2d", [])
        if not current_bbox or not previous_bbox or not next_bbox:
            continue
        if (
            abs(current_bbox[0] - previous_bbox[0]) <= left_align_threshold
            and abs(current_bbox[0] - next_bbox[0]) <= left_align_threshold
        ):
            updated = "- " + current_content
            current_block["content"] = updated
            events.append(
                _make_event(
                    "bullet_repair",
                    "add",
                    current_block.get("source_indices", []),
                    before=current_content,
                    after=updated,
                    left_align_threshold=left_align_threshold,
                )
            )
    return results


def _merge_formula_numbers(
    results: list[dict[str, Any]],
    events: list[dict[str, Any]],
    allow_inline_formula: bool = True,
) -> list[dict[str, Any]]:
    if not results:
        return results
    merged_results = []
    skip_indices = set()
    for index, block in enumerate(results):
        if index in skip_indices:
            continue
        native_label = block.get("native_label", "")
        if native_label == "formula_number":
            if index + 1 < len(results):
                next_block = results[index + 1]
                if next_block.get("label") == "formula" and (
                    allow_inline_formula
                    or next_block.get("native_label") == "display_formula"
                ):
                    number_clean = _clean_formula_number(block.get("content", "").strip())
                    formula_content = next_block.get("content", "")
                    merged_block = deepcopy(next_block)
                    if formula_content.endswith("\n$$"):
                        merged_block["content"] = (
                            formula_content[:-3] + f" \\tag{{{number_clean}}}\n$$"
                        )
                    merged_block["source_indices"] = (
                        block.get("source_indices", []) + next_block.get("source_indices", [])
                    )
                    merged_results.append(merged_block)
                    skip_indices.add(index + 1)
                    events.append(
                        _make_event(
                            "formula_number",
                            "merge_before_formula",
                            merged_block["source_indices"],
                            before=block.get("content", "") + "\n\n" + formula_content,
                            after=merged_block["content"],
                            number=number_clean,
                        )
                    )
                    continue
            events.append(
                _make_event(
                    "formula_number",
                    "drop_standalone",
                    block.get("source_indices", []),
                    before=block.get("content", ""),
                )
            )
            continue
        if block.get("label") == "formula":
            if index + 1 < len(results):
                next_block = results[index + 1]
                if (
                    next_block.get("native_label") == "formula_number"
                    and (
                        allow_inline_formula
                        or block.get("native_label") == "display_formula"
                    )
                ):
                    number_clean = _clean_formula_number(next_block.get("content", "").strip())
                    formula_content = block.get("content", "")
                    merged_block = deepcopy(block)
                    if formula_content.endswith("\n$$"):
                        merged_block["content"] = (
                            formula_content[:-3] + f" \\tag{{{number_clean}}}\n$$"
                        )
                    merged_block["source_indices"] = (
                        block.get("source_indices", []) + next_block.get("source_indices", [])
                    )
                    merged_results.append(merged_block)
                    skip_indices.add(index + 1)
                    events.append(
                        _make_event(
                            "formula_number",
                            "merge_after_formula",
                            merged_block["source_indices"],
                            before=formula_content + "\n\n" + next_block.get("content", ""),
                            after=merged_block["content"],
                            number=number_clean,
                        )
                    )
                    continue
            merged_results.append(block)
            continue
        merged_results.append(block)
    for index, block in enumerate(merged_results):
        block["index"] = index
    return merged_results


def _validate_table_html(content: str) -> tuple[bool, list[str]]:
    stripped = content.strip()
    issues = []
    if not re.match(r"^<table(?:\s|>)", stripped, flags=re.IGNORECASE):
        issues.append("missing_open_table")
    if not re.search(r"</table>\s*$", stripped, flags=re.IGNORECASE):
        issues.append("missing_close_table")
    for tag in ("table", "tr", "td", "th"):
        opening = len(re.findall(rf"<{tag}(?:\s|>)", stripped, flags=re.IGNORECASE))
        closing = len(re.findall(rf"</{tag}>", stripped, flags=re.IGNORECASE))
        if opening != closing:
            issues.append(f"unbalanced_{tag}:{opening}!={closing}")
    return not issues, issues


def _content_from_item(item: dict[str, Any]) -> Any:
    content = item.get("vlm_merged") or item.get("llm_merged") or item.get("merged", "")
    if content:
        return content
    content_dict = item.get("content", {})
    if isinstance(content_dict, dict):
        for value in content_dict.values():
            if value:
                return value
    return ""


_COMPACTION_UNSAFE_TOKENS = (
    "$$",
    "\\[",
    "\\]",
    "\\begin{",
    "\\end{",
    "<table",
    "</table",
    "```",
    "![",
    "<img",
)


def _nonempty_line_count(content: str) -> int:
    return sum(1 for line in content.splitlines() if line.strip())


def _is_compaction_safe(block: dict[str, Any]) -> bool:
    if block.get("label") != "text":
        return False
    content = block.get("content", "")
    if not isinstance(content, str) or not content.strip():
        return False
    lowered = content.lower()
    return not any(token.lower() in lowered for token in _COMPACTION_UNSAFE_TOKENS)


def _compact_text_content(content: str) -> str:
    return " ".join(line.strip() for line in content.splitlines() if line.strip())


def _numeric_bbox(block: dict[str, Any]) -> list[float] | None:
    bbox = block.get("bbox_2d")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        numeric = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if numeric[2] <= numeric[0] or numeric[3] <= numeric[1]:
        return None
    return numeric


def _bbox_union(first: list[float], second: list[float]) -> list[float]:
    return [
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    ]


_PADDLE_INLINE_OVERLAP_THRESHOLD = 0.5


def _filter_paddle_inline_overlap(
    items: list[dict[str, Any]],
    events: list[dict[str, Any]],
    stats: Counter,
) -> list[dict[str, Any]]:
    inline_positions = [
        position
        for position, item in enumerate(items)
        if item.get("label") == "inline_formula"
    ]
    stats["paddle_inline_regions"] += len(inline_positions)
    if not inline_positions:
        return items
    comparable_positions = [
        position
        for position, item in enumerate(items)
        if item.get("label") != "reference"
    ]
    dropped_positions = set()
    for left_offset, left_position in enumerate(comparable_positions):
        left_item = items[left_position]
        left_bbox = _numeric_bbox(left_item)
        if left_bbox is None:
            continue
        left_area = (left_bbox[2] - left_bbox[0]) * (left_bbox[3] - left_bbox[1])
        for right_position in comparable_positions[left_offset + 1 :]:
            if left_position in dropped_positions or right_position in dropped_positions:
                continue
            right_item = items[right_position]
            if (
                left_item.get("label") != "inline_formula"
                and right_item.get("label") != "inline_formula"
            ):
                continue
            right_bbox = _numeric_bbox(right_item)
            if right_bbox is None:
                continue
            right_area = (right_bbox[2] - right_bbox[0]) * (
                right_bbox[3] - right_bbox[1]
            )
            smaller_area = min(left_area, right_area)
            if smaller_area <= 0:
                continue
            overlap_ratio = _bbox_intersection_area(left_bbox, right_bbox) / smaller_area
            if overlap_ratio <= _PADDLE_INLINE_OVERLAP_THRESHOLD:
                continue
            for inline_position, counterpart_position in (
                (left_position, right_position),
                (right_position, left_position),
            ):
                inline_item = items[inline_position]
                if inline_item.get("label") != "inline_formula":
                    continue
                dropped_positions.add(inline_position)
                counterpart = items[counterpart_position]
                stats["paddle_inline_dropped"] += 1
                events.append(
                    _make_event(
                        "inline_formula",
                        "paddle_overlap_drop",
                        [inline_item.get("index", inline_position)],
                        before=_content_from_item(inline_item),
                        overlap_ratio=round(overlap_ratio, 6),
                        threshold=_PADDLE_INLINE_OVERLAP_THRESHOLD,
                        counterpart_index=counterpart.get(
                            "index", counterpart_position
                        ),
                        counterpart_label=counterpart.get("label"),
                    )
                )
    stats["paddle_inline_kept"] += len(inline_positions) - len(dropped_positions)
    return [
        item for position, item in enumerate(items) if position not in dropped_positions
    ]


_INLINE_PARENT_NATIVE_LABELS = frozenset(
    {
        "abstract",
        "algorithm",
        "content",
        "doc_title",
        "figure_title",
        "paragraph_title",
        "reference",
        "reference_content",
        "text",
        "vertical_text",
        "vision_footnote",
        "footnote",
        "aside_text",
    }
)

_MATH_SPAN_RE = re.compile(
    r"\$\$(.*?)\$\$|"
    r"\\\[(.*?)\\\]|"
    r"\\\((.*?)\\\)|"
    r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)",
    flags=re.DOTALL,
)


def _strip_formula_wrappers(content: Any) -> str:
    formula = str(content or "").strip()
    wrapper_pairs = (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$"))
    changed = True
    while formula and changed:
        changed = False
        for left, right in wrapper_pairs:
            if (
                formula.startswith(left)
                and formula.endswith(right)
                and len(formula) >= len(left) + len(right)
            ):
                formula = formula[len(left) : len(formula) - len(right)].strip()
                changed = True
                break
    return formula


def _canonical_formula(content: Any) -> str:
    formula = _strip_formula_wrappers(content)
    formula = formula.replace("∴", "\\therefore").replace("∵", "\\because")
    formula = re.sub(r"\\(?:left|right)\b", "", formula)
    formula = re.sub(r"\\(?:quad|qquad)\b|\\[,;:!]", "", formula)
    formula = re.sub(r"\^\s*\{\s*\\prime\s*\}", "'", formula)
    formula = re.sub(r"\^\s*\\prime\b", "'", formula)
    formula = re.sub(r"\{\s*\\prime\s*\}", "'", formula)
    formula = re.sub(r"([_^])\s*\{\s*([A-Za-z0-9]+)\s*\}", r"\1\2", formula)
    return re.sub(r"\s+", "", formula).strip()


def _text_formula_fingerprints(content: Any) -> set[str]:
    fingerprints = set()
    for match in _MATH_SPAN_RE.finditer(str(content or "")):
        formula = next((group for group in match.groups() if group is not None), "")
        fingerprint = _canonical_formula(formula)
        if fingerprint:
            fingerprints.add(fingerprint)
    return fingerprints


def _bbox_intersection_area(first: list[float], second: list[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _inline_parent_candidate(
    inline_block: dict[str, Any],
    inline_position: int,
    text_block: dict[str, Any],
    text_position: int,
    options: MergeOptions,
) -> dict[str, Any] | None:
    inline_bbox = _numeric_bbox(inline_block)
    text_bbox = _numeric_bbox(text_block)
    if inline_bbox is None or text_bbox is None:
        return None
    inline_width = inline_bbox[2] - inline_bbox[0]
    inline_height = inline_bbox[3] - inline_bbox[1]
    inline_area = inline_width * inline_height
    intersection_area = _bbox_intersection_area(inline_bbox, text_bbox)
    inline_cover = intersection_area / inline_area
    horizontal_overlap = max(
        0.0,
        min(inline_bbox[2], text_bbox[2]) - max(inline_bbox[0], text_bbox[0]),
    )
    x_overlap = horizontal_overlap / inline_width
    if inline_bbox[3] <= text_bbox[1]:
        vertical_gap = text_bbox[1] - inline_bbox[3]
    elif text_bbox[3] <= inline_bbox[1]:
        vertical_gap = inline_bbox[1] - text_bbox[3]
    else:
        vertical_gap = 0.0
    overlap_relation = inline_cover >= options.inline_parent_min_cover
    adjacent_relation = (
        intersection_area == 0
        and vertical_gap <= options.inline_parent_max_gap
        and x_overlap >= options.inline_parent_min_x_overlap
    )
    if not overlap_relation and not adjacent_relation:
        return None
    order_distance = abs(inline_position - text_position)
    if not order_distance:
        return None
    if overlap_relation:
        score = 2.0 + inline_cover + 0.2 * x_overlap
        relation = "overlap"
    else:
        gap_score = 1.0 - min(vertical_gap / max(options.inline_parent_max_gap, 1.0), 1.0)
        score = 1.2 + gap_score + 0.2 * x_overlap
        relation = "adjacent"
    if text_position < inline_position:
        score += 0.3 / order_distance
    else:
        score += 0.15 / order_distance
    boundary_band = max(options.inline_parent_boundary_band, inline_height * 1.5)
    inline_center_y = (inline_bbox[1] + inline_bbox[3]) / 2.0
    if inline_bbox[1] >= text_bbox[3]:
        placement = "after"
    elif inline_bbox[3] <= text_bbox[1]:
        placement = "before"
    elif inline_center_y >= text_bbox[3] - boundary_band:
        placement = "after"
    elif inline_center_y <= text_bbox[1] + boundary_band:
        placement = "before"
    else:
        placement = "internal"
    return {
        "block_position": text_position,
        "source_indices": list(text_block.get("source_indices", [])),
        "score": score,
        "relation": relation,
        "placement": placement,
        "inline_cover": inline_cover,
        "x_overlap": x_overlap,
        "vertical_gap": vertical_gap,
        "order_distance": order_distance,
    }


def _inline_parent_candidates(
    results: list[dict[str, Any]],
    inline_position: int,
    options: MergeOptions,
) -> list[dict[str, Any]]:
    inline_block = results[inline_position]
    candidates = []
    for text_position, text_block in enumerate(results):
        if text_block.get("label") != "text":
            continue
        if text_block.get("native_label") not in _INLINE_PARENT_NATIVE_LABELS:
            continue
        candidate = _inline_parent_candidate(
            inline_block,
            inline_position,
            text_block,
            text_position,
            options,
        )
        if candidate is not None:
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _select_inline_parent(
    candidates: list[dict[str, Any]],
    score_margin: float,
) -> tuple[dict[str, Any] | None, str]:
    if not candidates:
        return None, "no_parent"
    if len(candidates) > 1 and candidates[0]["score"] - candidates[1]["score"] < score_margin:
        return None, "ambiguous_parent"
    return candidates[0], "selected"


def _candidate_audit(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_source_indices": candidate["source_indices"],
        "score": round(candidate["score"], 6),
        "relation": candidate["relation"],
        "placement": candidate["placement"],
        "inline_cover": round(candidate["inline_cover"], 6),
        "x_overlap": round(candidate["x_overlap"], 6),
        "vertical_gap": round(candidate["vertical_gap"], 6),
        "order_distance": candidate["order_distance"],
    }


def _inline_formula_markdown(content: Any) -> str:
    formula = _strip_formula_wrappers(content)
    return f"${formula}$" if formula else ""


def _resolve_inline_formulas(
    results: list[dict[str, Any]],
    options: MergeOptions,
    events: list[dict[str, Any]],
    stats: Counter,
) -> list[dict[str, Any]]:
    if options.inline_formula_policy in {"display", "paddle-overlap"} or not results:
        return results
    updated_results = [deepcopy(block) for block in results]
    removals = set()
    insertions: dict[tuple[int, str], list[dict[str, Any]]] = {}
    inline_positions = [
        position
        for position, block in enumerate(updated_results)
        if block.get("native_label") == "inline_formula"
    ]
    stats["inline_formula_regions"] += len(inline_positions)
    for inline_position in inline_positions:
        inline_block = updated_results[inline_position]
        inline_source_indices = list(inline_block.get("source_indices", []))
        fingerprint = _canonical_formula(inline_block.get("content", ""))
        candidates = _inline_parent_candidates(updated_results, inline_position, options)
        matching_candidates = [
            candidate
            for candidate in candidates
            if fingerprint
            and fingerprint
            in _text_formula_fingerprints(
                updated_results[candidate["block_position"]].get("content", "")
            )
        ]
        if matching_candidates:
            parent, status = _select_inline_parent(
                matching_candidates,
                options.inline_parent_score_margin,
            )
            if parent is None:
                stats["inline_formula_fail_open"] += 1
                stats[f"inline_formula_{status}"] += 1
                events.append(
                    _make_event(
                        "inline_formula",
                        f"keep_display_{status}",
                        inline_source_indices,
                        before=inline_block.get("content", ""),
                        normalized_formula=fingerprint,
                        candidates=[_candidate_audit(item) for item in matching_candidates[:3]],
                    )
                )
                continue
            parent_block = updated_results[parent["block_position"]]
            removals.add(inline_position)
            stats["inline_formula_deduped"] += 1
            events.append(
                _make_event(
                    "inline_formula",
                    "dedupe",
                    parent_block.get("source_indices", []) + inline_source_indices,
                    before=(
                        parent_block.get("content", "")
                        + "\n\n"
                        + inline_block.get("content", "")
                    ),
                    after=parent_block.get("content", ""),
                    normalized_formula=fingerprint,
                    parent=_candidate_audit(parent),
                )
            )
            continue
        if options.inline_formula_policy == "dedupe":
            stats["inline_formula_fail_open"] += 1
            stats["inline_formula_no_content_match"] += 1
            events.append(
                _make_event(
                    "inline_formula",
                    "keep_display_no_content_match",
                    inline_source_indices,
                    before=inline_block.get("content", ""),
                    normalized_formula=fingerprint,
                    candidates=[_candidate_audit(item) for item in candidates[:3]],
                )
            )
            continue
        parent, status = _select_inline_parent(
            candidates,
            options.inline_parent_score_margin,
        )
        if parent is None:
            stats["inline_formula_fail_open"] += 1
            stats[f"inline_formula_{status}"] += 1
            events.append(
                _make_event(
                    "inline_formula",
                    f"keep_display_{status}",
                    inline_source_indices,
                    before=inline_block.get("content", ""),
                    normalized_formula=fingerprint,
                    candidates=[_candidate_audit(item) for item in candidates[:3]],
                )
            )
            continue
        placement = parent["placement"]
        inline_markdown = _inline_formula_markdown(inline_block.get("content", ""))
        if placement not in {"before", "after"} or not inline_markdown:
            reason = "internal_position" if placement == "internal" else "blank_formula"
            stats["inline_formula_fail_open"] += 1
            stats[f"inline_formula_{reason}"] += 1
            events.append(
                _make_event(
                    "inline_formula",
                    f"keep_display_{reason}",
                    inline_source_indices,
                    before=inline_block.get("content", ""),
                    normalized_formula=fingerprint,
                    parent=_candidate_audit(parent),
                )
            )
            continue
        removals.add(inline_position)
        insertions.setdefault((parent["block_position"], placement), []).append(
            {
                "inline_position": inline_position,
                "source_indices": inline_source_indices,
                "bbox_2d": inline_block.get("bbox_2d", []),
                "markdown": inline_markdown,
                "normalized_formula": fingerprint,
                "parent": parent,
            }
        )
    for (parent_position, placement), members in sorted(insertions.items()):
        parent_block = updated_results[parent_position]
        parent_before = str(parent_block.get("content", ""))
        members.sort(key=lambda item: item["inline_position"])
        inline_run = " ".join(item["markdown"] for item in members)
        inline_source_indices = [
            source_index
            for item in members
            for source_index in item["source_indices"]
        ]
        if placement == "before":
            parent_after = inline_run + "\n" + parent_before.lstrip()
            parent_block["source_indices"] = (
                inline_source_indices + parent_block.get("source_indices", [])
            )
        else:
            parent_after = parent_before.rstrip() + "\n" + inline_run
            parent_block["source_indices"] = (
                parent_block.get("source_indices", []) + inline_source_indices
            )
        parent_block["content"] = parent_after
        stats["inline_formula_inserted"] += len(members)
        stats[f"inline_formula_inserted_{placement}"] += len(members)
        stats["inline_formula_insert_groups"] += 1
        events.append(
            _make_event(
                "inline_formula",
                f"insert_{placement}",
                parent_block.get("source_indices", []),
                before=parent_before,
                after=parent_after,
                inline_source_indices=inline_source_indices,
                formulas=[item["markdown"] for item in members],
                normalized_formulas=[item["normalized_formula"] for item in members],
                parent=_candidate_audit(members[0]["parent"]),
            )
        )
    resolved = [
        block for position, block in enumerate(updated_results) if position not in removals
    ]
    for index, block in enumerate(resolved):
        block["index"] = index
    return resolved


def _same_text_column(
    previous_bbox: list[float],
    current_bbox: list[float],
    options: MergeOptions,
) -> bool:
    if current_bbox[1] + 5.0 < previous_bbox[1]:
        return False
    if current_bbox[1] - previous_bbox[3] > options.compact_max_vertical_gap:
        return False
    previous_width = previous_bbox[2] - previous_bbox[0]
    current_width = current_bbox[2] - current_bbox[0]
    overlap = max(
        0.0,
        min(previous_bbox[2], current_bbox[2])
        - max(previous_bbox[0], current_bbox[0]),
    )
    overlap_ratio = overlap / min(previous_width, current_width)
    if overlap_ratio >= options.compact_min_horizontal_overlap:
        return True
    previous_center = (previous_bbox[0] + previous_bbox[2]) / 2.0
    current_center = (current_bbox[0] + current_bbox[2]) / 2.0
    max_width = max(previous_width, current_width)
    return (
        abs(previous_center - current_center) <= max_width * 0.25
        and abs(previous_bbox[0] - current_bbox[0]) <= max_width * 0.35
    )


def _apply_adaptive_text_compaction(
    results: list[dict[str, Any]],
    options: MergeOptions,
    events: list[dict[str, Any]],
    stats: Counter,
) -> list[dict[str, Any]]:
    if options.text_compaction == "off" or not results:
        return results
    safe_text_blocks = [block for block in results if _is_compaction_safe(block)]
    long_region_count = sum(
        _nonempty_line_count(block["content"]) >= options.compact_line_threshold
        for block in safe_text_blocks
    )
    fragmented_page = len(safe_text_blocks) >= options.compact_region_threshold
    if not long_region_count and not fragmented_page:
        return results

    stats["text_compaction_pages"] += 1
    stats["text_compaction_trigger_long_regions"] += long_region_count
    stats["text_compaction_trigger_fragmented_pages"] += int(fragmented_page)
    stats["text_compaction_input_text_blocks"] += len(safe_text_blocks)
    source_indices = [
        source_index
        for block in safe_text_blocks
        for source_index in block.get("source_indices", [])
    ]
    events.append(
        _make_event(
            "text_compaction",
            "trigger",
            source_indices,
            safe_text_regions=len(safe_text_blocks),
            long_text_regions=long_region_count,
            line_threshold=options.compact_line_threshold,
            region_threshold=options.compact_region_threshold,
        )
    )

    compacted = []
    for block in results:
        compacted_block = deepcopy(block)
        if _is_compaction_safe(compacted_block):
            before = compacted_block["content"]
            line_count = _nonempty_line_count(before)
            after = _compact_text_content(before)
            compacted_block["content"] = after
            stats["text_compaction_lines_removed"] += max(0, line_count - 1)
            if after != before:
                events.append(
                    _make_event(
                        "text_compaction",
                        "collapse_lines",
                        compacted_block.get("source_indices", []),
                        before=before,
                        after=after,
                        nonempty_lines=line_count,
                    )
                )
        compacted.append(compacted_block)

    merged: list[dict[str, Any]] = []
    for block in compacted:
        current_bbox = _numeric_bbox(block)
        if (
            merged
            and _is_compaction_safe(block)
            and _is_compaction_safe(merged[-1])
            and current_bbox is not None
        ):
            previous = merged[-1]
            previous_bbox = previous.get("_compaction_tail_bbox") or _numeric_bbox(previous)
            if previous_bbox is not None and _same_text_column(
                previous_bbox,
                current_bbox,
                options,
            ):
                before = previous["content"] + "\n\n" + block["content"]
                previous["content"] = previous["content"].rstrip() + " " + block["content"].lstrip()
                previous["source_indices"] = (
                    previous.get("source_indices", []) + block.get("source_indices", [])
                )
                base_bbox = _numeric_bbox(previous)
                if base_bbox is not None:
                    previous["bbox_2d"] = _bbox_union(base_bbox, current_bbox)
                previous["_compaction_tail_bbox"] = current_bbox
                stats["text_compaction_regions_merged"] += 1
                events.append(
                    _make_event(
                        "text_compaction",
                        "merge_regions",
                        previous.get("source_indices", []),
                        before=before,
                        after=previous["content"],
                    )
                )
                continue
        new_block = deepcopy(block)
        if current_bbox is not None:
            new_block["_compaction_tail_bbox"] = current_bbox
        merged.append(new_block)

    for index, block in enumerate(merged):
        block.pop("_compaction_tail_bbox", None)
        block["index"] = index
    stats["text_compaction_output_text_blocks"] += sum(
        _is_compaction_safe(block) for block in merged
    )
    return merged


def merge_page(
    layout_result: list[dict[str, Any]],
    options: MergeOptions | None = None,
) -> MergeOutput:
    options = options or MergeOptions()
    if not layout_result:
        return MergeOutput(markdown="", events=[], stats={"input_regions": 0})
    sorted_items = sorted(layout_result, key=lambda item: item.get("index", 0))
    events: list[dict[str, Any]] = []
    stats = Counter()
    if options.inline_formula_policy == "paddle-overlap":
        sorted_items = _filter_paddle_inline_overlap(sorted_items, events, stats)
    processed = []
    valid_index = 0
    for ordinal, item in enumerate(sorted_items):
        source_index = item.get("index", ordinal)
        native_label = item.get("label", "text")
        mapped_label = map_label(
            native_label,
            options.include_footnote,
            options.include_aside_text,
            options.include_reference,
        )
        stats["input_regions"] += 1
        stats[f"mapped_{mapped_label}"] += 1
        if mapped_label == "abandon":
            stats["dropped_abandon"] += 1
            continue
        if mapped_label == "image":
            asset_markdown = item.get("_asset_markdown") if options.include_images else None
            if not asset_markdown:
                stats["dropped_image"] += 1
                continue
            processed.append(
                {
                    "index": valid_index,
                    "label": "image",
                    "native_label": native_label,
                    "content": asset_markdown,
                    "bbox_2d": item.get("bbox_2d", []),
                    "source_indices": [source_index],
                }
            )
            valid_index += 1
            stats["included_image"] += 1
            continue
        raw_content = _content_from_item(item)
        formatted = _format_content(raw_content, mapped_label, native_label, options)
        if options.text_profile == "paddle" and mapped_label == "text":
            baseline_formatted = _format_content(
                raw_content,
                mapped_label,
                native_label,
                MergeOptions(title_profile=options.title_profile),
            )
            if formatted != baseline_formatted:
                events.append(
                    _make_event(
                        "text_profile",
                        "paddle_format",
                        [source_index],
                        before=baseline_formatted,
                        after=formatted,
                        native_label=native_label,
                    )
                )
        elif options.title_profile == "numbered" and native_label == "paragraph_title":
            baseline_formatted = _format_content(
                raw_content,
                mapped_label,
                native_label,
                MergeOptions(),
            )
            if formatted != baseline_formatted:
                events.append(
                    _make_event(
                        "title_profile",
                        "numbered",
                        [source_index],
                        before=baseline_formatted,
                        after=formatted,
                    )
                )
        if formatted is None or (isinstance(formatted, str) and not formatted.strip()):
            stats["dropped_blank"] += 1
            continue
        include_optional_text = (
            native_label == "footnote" and options.include_footnote
        ) or (
            native_label == "aside_text" and options.include_aside_text
        )
        if include_optional_text:
            stats["included_paratext"] += 1
            stats[f"included_{native_label}"] += 1
            events.append(
                _make_event(
                    "paratext",
                    "include",
                    [source_index],
                    after=formatted,
                    native_label=native_label,
                )
            )
        if native_label == "reference" and options.include_reference:
            stats["included_reference"] += 1
            events.append(
                _make_event(
                    "reference",
                    "include",
                    [source_index],
                    after=formatted,
                    native_label=native_label,
                )
            )
        if mapped_label == "table" and options.validate_tables:
            valid_table, issues = _validate_table_html(formatted)
            stats["table_valid" if valid_table else "table_invalid"] += 1
            if not valid_table:
                events.append(
                    _make_event(
                        "table_validation",
                        "invalid",
                        [source_index],
                        before=formatted,
                        issues=issues,
                    )
                )
        processed.append(
            {
                "index": valid_index,
                "label": mapped_label,
                "native_label": native_label,
                "content": formatted,
                "bbox_2d": item.get("bbox_2d", []),
                "source_indices": [source_index],
            }
        )
        valid_index += 1
    processed = _resolve_inline_formulas(processed, options, events, stats)
    processed = _merge_formula_numbers(
        processed,
        events,
        allow_inline_formula=options.inline_formula_policy == "display",
    )
    processed = _merge_text_blocks(processed, options.hyphen_merge, events)
    if options.bullet_repair:
        processed = _format_bullet_points(processed, events)
    processed = _apply_adaptive_text_compaction(
        processed,
        options,
        events,
        stats,
    )
    parts = [block["content"] for block in processed if block.get("content")]
    stats["output_blocks"] = len(parts)
    stats["events"] = len(events)
    return MergeOutput(markdown="\n\n".join(parts), events=events, stats=dict(stats))


def merge_page_to_markdown(
    layout_result: list[dict[str, Any]],
    options: MergeOptions | None = None,
) -> str:
    return merge_page(layout_result, options=options).markdown
