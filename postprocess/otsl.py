"""OTSL table markup to HTML conversion.

OTSL is the compact table markup the OCR service emits for a table region.
Every cell is introduced by a tag and the text following that tag belongs to the
cell; ``<nl>`` ends a row:

    <fcel>   cell carrying text          <ecel>   empty cell
    <lcel>   continues the cell at left  <ucel>   continues the cell above
    <xcel>   continues both              <nl>     end of row

Only ``<fcel>`` and ``<ecel>`` start a cell. The continuation tags carry no text
of their own: they extend an origin cell, which becomes ``rowspan``/``colspan``
in the emitted HTML.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ROW_BREAK = "<nl>"
CELL = "<fcel>"
EMPTY = "<ecel>"
LEFT = "<lcel>"
UP = "<ucel>"
BOTH = "<xcel>"

ALL_TAGS = (CELL, EMPTY, LEFT, UP, BOTH, ROW_BREAK)
TAG_RE = re.compile("(" + "|".join(re.escape(tag) for tag in ALL_TAGS) + ")")

ORIGIN_TAGS = frozenset({CELL, EMPTY})
RIGHT_CONTINUATION = frozenset({LEFT, BOTH})
DOWN_CONTINUATION = frozenset({UP, BOTH})


@dataclass
class Cell:
    """A resolved cell placed on the rectangular grid."""

    text: str
    row: int
    column: int
    rowspan: int = 1
    colspan: int = 1


def _tag_of(chunk: str) -> str:
    match = TAG_RE.match(chunk)
    return match.group(0) if match else ""


def _split_chunks(line: str) -> list[str]:
    """Split a row into chunks, each holding one tag plus the text after it."""
    starts = [match.start() for match in TAG_RE.finditer(line)]
    ends = starts[1:] + [len(line)]
    return [line[start:end] for start, end in zip(starts, ends)]


def _required_width(chunks: list[str]) -> int:
    """Number of leading cells a row must keep to preserve its last content."""
    width = 0
    for position, chunk in enumerate(chunks, start=1):
        if _tag_of(chunk) == CELL:
            width = position
    return width


def _target_width(rows: list[list[str]]) -> int:
    """Pick one cell count every row is padded or truncated to.

    Every row must keep at least the cells it declared content for, and the
    cheapest width balances the total amount of truncation and padding.
    """
    lower = max(_required_width(row) for row in rows)
    upper = max(len(row) for row in rows)
    best_width = upper
    best_cost = None
    for width in range(lower, upper + 1):
        cost = sum(abs(len(row) - width) for row in rows)
        if best_cost is None or cost < best_cost:
            best_width = width
            best_cost = cost
    return best_width


def pad_rows(otsl: str) -> str:
    """Give every row the same cell count so the grid can be indexed."""
    text = otsl.strip()
    if ROW_BREAK not in text:
        return text + ROW_BREAK

    rows = []
    for line in text.split(ROW_BREAK):
        chunks = _split_chunks(line)
        if chunks:
            rows.append(chunks)
    if not rows:
        return ROW_BREAK

    width = _target_width(rows)
    padded = []
    for chunks in rows:
        fixed = chunks[:width]
        fixed += [EMPTY] * (width - len(fixed))
        padded.append("".join(fixed))
    return ROW_BREAK.join(padded) + ROW_BREAK


def parse_grid(otsl: str) -> tuple[list[list[str]], list[list[str]]]:
    """Return the padded table as parallel tag and text grids."""
    tag_rows: list[list[str]] = []
    text_rows: list[list[str]] = []

    for line in otsl.split(ROW_BREAK):
        chunks = _split_chunks(line)
        if not chunks:
            continue
        tags = []
        texts = []
        for chunk in chunks:
            tag = _tag_of(chunk)
            tags.append(tag)
            texts.append(chunk[len(tag):] if tag == CELL else "")
        tag_rows.append(tags)
        text_rows.append(texts)

    width = max((len(row) for row in tag_rows), default=0)
    for tags, texts in zip(tag_rows, text_rows):
        tags += [EMPTY] * (width - len(tags))
        texts += [""] * (width - len(texts))
    return tag_rows, text_rows


def resolve_cells(
    tag_rows: list[list[str]], text_rows: list[list[str]]
) -> list[Cell]:
    """Turn origin tags into cells, measuring spans from the continuation tags."""
    height = len(tag_rows)
    cells: list[Cell] = []

    for row_index, tags in enumerate(tag_rows):
        for column, tag in enumerate(tags):
            if tag not in ORIGIN_TAGS:
                continue

            colspan = 1
            while (
                column + colspan < len(tags)
                and tags[column + colspan] in RIGHT_CONTINUATION
            ):
                colspan += 1

            rowspan = 1
            while (
                row_index + rowspan < height
                and column < len(tag_rows[row_index + rowspan])
                and tag_rows[row_index + rowspan][column] in DOWN_CONTINUATION
            ):
                rowspan += 1

            text = text_rows[row_index][column] if tag == CELL else ""
            cells.append(
                Cell(
                    text=text.strip(),
                    row=row_index,
                    column=column,
                    rowspan=rowspan,
                    colspan=colspan,
                )
            )
    return cells


def render_html(cells: list[Cell], num_rows: int, num_cols: int) -> str:
    """Render resolved cells as an HTML table element.

    Layout is walked column by column, so every row emits exactly one cell per
    grid column and the result stays a well-formed table even when the input
    OTSL contradicts itself. Spans are measured against the ownership grid, so
    a span never claims a position owned by another cell.
    """
    if not cells:
        return ""

    grid: list[list[Cell | None]] = [
        [None] * num_cols for _ in range(num_rows)
    ]
    for cell in cells:
        for row in range(cell.row, min(cell.row + cell.rowspan, num_rows)):
            for column in range(cell.column, min(cell.column + cell.colspan, num_cols)):
                if row >= 0 and column >= 0:
                    grid[row][column] = cell

    covered_until: dict[int, int] = {}
    parts = []
    for row in range(num_rows):
        parts.append("<tr>")
        column = 0
        while column < num_cols:
            if covered_until.get(column, 0) > row:
                column += 1
                continue

            cell = grid[row][column]
            if cell is None or cell.row != row or cell.column != column:
                # A continuation tag with no origin of its own: keep the column
                # so the row still spans the full grid width.
                parts.append("<td></td>")
                column += 1
                continue

            colspan = 1
            while (
                column + colspan < num_cols
                and grid[row][column + colspan] is cell
            ):
                colspan += 1

            rowspan = 1
            while row + rowspan < num_rows and all(
                grid[row + rowspan][offset] is cell
                for offset in range(column, column + colspan)
            ):
                rowspan += 1

            attributes = ""
            if rowspan > 1:
                attributes += f' rowspan="{rowspan}"'
            if colspan > 1:
                attributes += f' colspan="{colspan}"'
            parts.append(f"<td{attributes}>{cell.text}</td>")

            for offset in range(column, column + colspan):
                covered_until[offset] = max(
                    covered_until.get(offset, 0), row + rowspan
                )
            column += colspan
        parts.append("</tr>")
    return "<table>" + "".join(parts) + "</table>"


def convert_otsl_to_html(otsl_content: str) -> str:
    """Convert an OTSL table into an HTML ``<table>`` element."""
    if not isinstance(otsl_content, str):
        raise TypeError("otsl_content must be a string")

    tag_rows, text_rows = parse_grid(pad_rows(otsl_content))
    cells = resolve_cells(tag_rows, text_rows)
    num_rows = len(tag_rows)
    num_cols = max((len(row) for row in tag_rows), default=0)
    return render_html(cells, num_rows, num_cols)