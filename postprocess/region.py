"""Shared region assembly, preserved from the evaluation pipeline."""
import re
from collections import Counter
from copy import deepcopy

try:
    from wordfreq import zipf_frequency
except ImportError:
    zipf_frequency = None

LABEL_VISUALIZATION_MAPPING = {
    "image": ["chart", "image"],
    "text": [
        "abstract", "algorithm", "content", "doc_title", "figure_title",
        "paragraph_title", "reference_content", "text", "vertical_text",
        "vision_footnote", "seal", "formula_number",
    ],
    "table": ["table"],
    "formula": ["display_formula", "inline_formula"],
    "abandon": [
        "header", "footer", "number", "footnote",
        "aside_text", "reference", "footer_image", "header_image",
    ],
}


def _map_label(label: str) -> str:
    for mapped_type, labels in LABEL_VISUALIZATION_MAPPING.items():
        if label in labels:
            return mapped_type
    return label


def _find_consecutive_repeat(s, min_unit_len=10, min_repeats=10):
    n = len(s)
    if n < min_unit_len * min_repeats:
        return None
    max_unit_len = n // min_repeats
    if max_unit_len < min_unit_len:
        return None
    pattern = re.compile(
        r"(.{" + str(min_unit_len) + "," + str(max_unit_len) + r"}?)\1{"
        + str(min_repeats - 1) + ",}", re.DOTALL,
    )
    match = pattern.search(s)
    if match:
        return s[: match.start()] + match.group(1)
    return None


def _clean_repeated_content(content, min_len=10, min_repeats=10, line_threshold=10):
    stripped_content = content.strip()
    if not stripped_content:
        return content
    if len(stripped_content) > min_len * min_repeats:
        result = _find_consecutive_repeat(stripped_content, min_unit_len=min_len, min_repeats=min_repeats)
        if result is not None:
            return result
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    total_lines = len(lines)
    if total_lines >= line_threshold and lines:
        common, count = Counter(lines).most_common(1)[0]
        if count >= line_threshold and (count / total_lines) >= 0.8:
            for i, line in enumerate(lines):
                if line == common:
                    consecutive = sum(1 for j in range(i, min(i + 3, len(lines))) if lines[j] == common)
                    if consecutive >= 3:
                        original_lines = content.split("\n")
                        non_empty_count = 0
                        for idx, orig_line in enumerate(original_lines):
                            if orig_line.strip():
                                non_empty_count += 1
                                if non_empty_count == i + 1:
                                    return "\n".join(original_lines[: idx + 1])
                        break
    return content


def _clean_formula_number(number_content):
    number_clean = number_content.strip()
    if number_clean.startswith("(") and number_clean.endswith(")"):
        number_clean = number_clean[1:-1]
    elif number_clean.startswith("（") and number_clean.endswith("）"):
        number_clean = number_clean[1:-1]
    return number_clean


def _normalize_inline_formula(content):
    INLINE_FORMULA_RE = re.compile(r"(?<!\$)\$\s*((?:[^$\\]|\\.)+?)\s*\$(?!\$)")
    if "$" not in content:
        return content
    parts = []
    last_end = 0
    for m in INLINE_FORMULA_RE.finditer(content):
        formula = m.group(1).strip()
        if not formula:
            continue
        start, end = m.start(), m.end()
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


def _clean_content(content):
    if content is None:
        return ""
    content = re.sub(r"^(\\t)+", "", content).lstrip()
    content = re.sub(r"(\\t)+$", "", content).rstrip()
    content = re.sub(r"(\.)\1{2,}", r"\1\1\1", content)
    content = re.sub(r"(·)\1{2,}", r"\1\1\1", content)
    content = re.sub(r"(_)\1{2,}", r"\1\1\1", content)
    content = re.sub(r"(\\_)\1{2,}", r"\1\1\1", content)
    if len(content) >= 2048:
        content = _clean_repeated_content(content)
    content = _normalize_inline_formula(content)
    return content.strip()


def _format_content(content, label, native_label):
    if content is None:
        return content
    if label == "table":
        if content.startswith("<table") and content.endswith("</table>"):
            content = content.strip()
        else:
            content = _clean_content(str(content))
    elif label == "formula":
        if content.startswith("$$") and content.endswith("$$"):
            content = content.strip()
        else:
            content = _clean_content(str(content))
    else:
        content = _clean_content(str(content))
    if native_label == "doc_title":
        content = re.sub(r"^#+\s*", "", content)
        content = "# " + content
    elif native_label == "paragraph_title":
        if content.startswith("- ") or content.startswith("* "):
            content = content[2:].lstrip()
        content = re.sub(r"^#+\s*", "", content)
        content = "## " + content.lstrip()
    if label == "formula":
        if content.startswith("$$") or content.startswith("\\[") or content.startswith("\\("):
            content = content[2:].strip()
        if content.endswith("$$") or content.endswith("\\]") or content.endswith("\\)"):
            content = content[:-2].strip()
        content = "$$\n" + content + "\n$$"
    if label == "text":
        if content.startswith("```") and (not content.endswith("```")):
            content = content + "\n```"
        if content.startswith("·") or content.startswith("•") or content.startswith("* "):
            content = "- " + content[1:].lstrip()
        match = re.match(r"^(\(|\（)(\d+|[A-Za-z])(\)|\）)(.*)$", content)
        if match:
            _, symbol, _, rest = match.groups()
            content = f"({symbol}) {rest.lstrip()}"
        match = re.match(r"^(\d+|[A-Za-z])(\.|\)|\）)(.*)$", content)
        if match:
            symbol, sep, rest = match.groups()
            sep = ")" if sep == "）" else sep
            content = f"{symbol}{sep} {rest.lstrip()}"
        content = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", content)
    return content


def _is_likely_valid_merged_word(merged_word):
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


def _merge_text_blocks(results):
    if not results:
        return results
    merged_results = []
    skip_indices = set()
    for i, block in enumerate(results):
        if i in skip_indices:
            continue
        if block.get("label") != "text":
            merged_results.append(block)
            continue
        content = block.get("content", "")
        if not isinstance(content, str):
            merged_results.append(block)
            continue
        content_stripped = content.rstrip()
        if not content_stripped:
            merged_results.append(block)
            continue
        if not content_stripped.endswith("-"):
            merged_results.append(block)
            continue
        merged = False
        for j in range(i + 1, len(results)):
            if results[j].get("label") == "text":
                next_content = results[j].get("content", "")
                if isinstance(next_content, str):
                    next_stripped = next_content.lstrip()
                    if next_stripped and next_stripped[0].islower():
                        words_before = content_stripped[:-1].split()
                        next_words = next_stripped.split()
                        if words_before and next_words:
                            word_fragment_before = words_before[-1]
                            word_fragment_after = next_words[0]
                            merged_word = word_fragment_before + word_fragment_after
                            if _is_likely_valid_merged_word(merged_word):
                                merged_content = content_stripped[:-1] + next_content.lstrip()
                                merged_block = deepcopy(block)
                                merged_block["content"] = merged_content
                                merged_results.append(merged_block)
                                skip_indices.add(j)
                                merged = True
                        break
        if not merged:
            merged_results.append(block)
    for idx, block in enumerate(merged_results):
        block["index"] = idx
    return merged_results


def _format_bullet_points(results, left_align_threshold=10.0):
    if len(results) < 3:
        return results
    for i in range(1, len(results) - 1):
        current_block = results[i]
        prev_block = results[i - 1]
        next_block = results[i + 1]
        if current_block.get("native_label") != "text":
            continue
        if prev_block.get("native_label") != "text" or next_block.get("native_label") != "text":
            continue
        current_content = current_block.get("content", "")
        if current_content.startswith("- "):
            continue
        prev_content = prev_block.get("content", "")
        next_content = next_block.get("content", "")
        if not (prev_content.startswith("- ") and next_content.startswith("- ")):
            continue
        current_bbox = current_block.get("bbox_2d", [])
        prev_bbox = prev_block.get("bbox_2d", [])
        next_bbox = next_block.get("bbox_2d", [])
        if not (current_bbox and prev_bbox and next_bbox):
            continue
        if (abs(current_bbox[0] - prev_bbox[0]) <= left_align_threshold
                and abs(current_bbox[0] - next_bbox[0]) <= left_align_threshold):
            current_block["content"] = "- " + current_content
    return results


def _merge_formula_numbers(results):
    if not results:
        return results
    merged_results = []
    skip_indices = set()
    for i, block in enumerate(results):
        if i in skip_indices:
            continue
        native_label = block.get("native_label", "")
        if native_label == "formula_number":
            if i + 1 < len(results):
                next_block = results[i + 1]
                if next_block.get("label") == "formula":
                    number_content = block.get("content", "").strip()
                    number_clean = _clean_formula_number(number_content)
                    formula_content = next_block.get("content", "")
                    merged_block = deepcopy(next_block)
                    if formula_content.endswith("\n$$"):
                        merged_block["content"] = formula_content[:-3] + f" \\tag{{{number_clean}}}\n$$"
                    merged_results.append(merged_block)
                    skip_indices.add(i + 1)
                    continue
            continue
        if block.get("label") == "formula":
            if i + 1 < len(results):
                next_block = results[i + 1]
                if next_block.get("native_label") == "formula_number":
                    number_content = next_block.get("content", "").strip()
                    number_clean = _clean_formula_number(number_content)
                    formula_content = block.get("content", "")
                    merged_block = deepcopy(block)
                    if formula_content.endswith("\n$$"):
                        merged_block["content"] = formula_content[:-3] + f" \\tag{{{number_clean}}}\n$$"
                    merged_results.append(merged_block)
                    skip_indices.add(i + 1)
                    continue
            merged_results.append(block)
            continue
        merged_results.append(block)
    for idx, block in enumerate(merged_results):
        block["index"] = idx
    return merged_results


def merge_page_to_markdown(layout_result: list) -> str:
    """Convert layout_result to markdown (self-contained, no external imports)."""
    if not layout_result:
        return ""
    sorted_items = sorted(layout_result, key=lambda x: x.get("index", 0))
    processed = []
    valid_idx = 0
    for item in sorted_items:
        native_label = item.get("label", "text")
        mapped_label = _map_label(native_label)
        if mapped_label in ("image", "abandon"):
            continue
        content = item.get("vlm_merged") or item.get("llm_merged") or item.get("merged", "")
        if not content:
            content_dict = item.get("content", {})
            if isinstance(content_dict, dict):
                for v in content_dict.values():
                    if v:
                        content = v
                        break
        formatted = _format_content(content, mapped_label, native_label)
        if formatted is None or (isinstance(formatted, str) and formatted.strip() == ""):
            continue
        processed.append({
            "index": valid_idx,
            "label": mapped_label,
            "native_label": native_label,
            "content": formatted,
            "bbox_2d": item.get("bbox_2d", []),
        })
        valid_idx += 1
    processed = _merge_formula_numbers(processed)
    processed = _merge_text_blocks(processed)
    processed = _format_bullet_points(processed)
    parts = [b["content"] for b in processed if b.get("content")]
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Layout detection
# ═══════════════════════════════════════════════════════════════════════════════
