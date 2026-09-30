"""Shared whole-page postprocessing, preserved from the evaluation pipeline."""
import re
from postprocess.otsl import convert_otsl_to_html

OTSL_TAG_RE = re.compile(r"<(?:fcel|ecel|nl|lcel|ucel|xcel)>")


def postprocess_math_delimiters(markdown: str) -> str:
    """Convert LaTeX delimiters in whole-page markdown to markdown math delimiters."""
    if not markdown:
        return markdown
    markdown = markdown.replace("\\[\\[", "\\[").replace("\\]\\]", "\\]")
    markdown = re.sub(
        r"\$\$?\s*\\\[\s*(.*?)\s*\\\]\s*\$\$?",
        r" $$ \1 $$ ",
        markdown,
        flags=re.DOTALL,
    )
    markdown = re.sub(
        r"\$\s*\\\(\s*(.*?)\s*\\\)\s*\$",
        r" $ \1 $",
        markdown,
        flags=re.DOTALL,
    )
    markdown = re.sub(
        r"\\\[\s*(.*?)\s*\\\]",
        r" $$ \1 $$ ",
        markdown,
        flags=re.DOTALL,
    )
    markdown = re.sub(
        r"\\\(\s*(.*?)\s*\\\)",
        r" $ \1 $",
        markdown,
        flags=re.DOTALL,
    )
    return markdown


def _split_markdown_blocks(markdown: str):
    parts = re.split(r"(\n\s*\n)", markdown)
    blocks = []
    for i in range(0, len(parts), 2):
        text = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        blocks.append([text, sep])
    return blocks


def postprocess_otsl_tables(markdown: str) -> str:
    """Convert OTSL-looking markdown blocks to HTML tables; keep original on failure."""
    if not markdown or not OTSL_TAG_RE.search(markdown):
        return markdown

    blocks = _split_markdown_blocks(markdown)
    in_fence = False
    converted = []
    for text, sep in blocks:
        stripped = text.strip()
        fence_count = stripped.count("```")
        should_skip = in_fence
        if fence_count % 2 == 1:
            in_fence = not in_fence
            should_skip = True

        if not should_skip and OTSL_TAG_RE.search(stripped):
            try:
                html = convert_otsl_to_html(stripped)
                if html:
                    text = html
            except Exception:
                pass
        converted.append(text + sep)
    return "".join(converted)


def postprocess_e2e_markdown(result: str, *, convert_otsl: bool = False) -> str:
    result = postprocess_math_delimiters(result)
    if convert_otsl:
        result = postprocess_otsl_tables(result)
    return result


def _clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """截断尾部周期性重复：保留一个周期 + 余数部分。

    只对 >= min_text_len 的长输出生效; 在最终写盘前对采用版 (初始或重试) 调用,
    作为 repetition retry 之外的兜底止损, 不影响重试判定 (判定基于原始输出)。
    """
    n = len(text)
    if n < min_text_len:
        return text

    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue

        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1

        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len

        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len:]

    return text


def finalize_document(text):
    return postprocess_e2e_markdown(_clean_truncated_repeats(text), convert_otsl=True)
