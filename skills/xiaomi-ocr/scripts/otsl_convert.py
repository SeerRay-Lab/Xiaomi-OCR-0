#!/usr/bin/env python3
"""Convert OTSL table markup (emitted by the OCR service) into Markdown or HTML.

OTSL tags seen from the deployed model:
  <fcel> cell with content           <ecel> empty cell
  <lcel> merged with the cell left   <ucel> merged with the cell above
  <xcel> merged both ways            <nl>   end of row

Markdown cannot express spans, so continuation cells are filled with the origin
value by default (`--merge repeat`) or left blank (`--merge blank`). HTML keeps
real rowspan/colspan.

Usage:
  python3 otsl_convert.py table.otsl                 # Markdown to stdout
  python3 otsl_convert.py --format html table.otsl
  cat table.otsl | python3 otsl_convert.py --format md
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

TAGS = ("fcel", "ecel", "lcel", "ucel", "xcel", "nl")
TAG_RE = re.compile(r"<(" + "|".join(TAGS) + r")>")
FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


@dataclass
class Cell:
    text: str
    row: int
    col: int
    colspan: int = 1
    rowspan: int = 1
    origin: bool = True


@dataclass
class Grid:
    cells: list[Cell] = field(default_factory=list)
    occupied: dict[tuple[int, int], Cell] = field(default_factory=dict)
    row: int = 0

    def first_free_col(self, row: int) -> int:
        col = 0
        while (row, col) in self.occupied:
            col += 1
        return col

    def place(self, cell: Cell) -> None:
        self.cells.append(cell)
        for r in range(cell.row, cell.row + cell.rowspan):
            for c in range(cell.col, cell.col + cell.colspan):
                self.occupied[(r, c)] = cell

    def n_rows(self) -> int:
        return max((c.row + c.rowspan for c in self.cells), default=0)

    def n_cols(self) -> int:
        return max((c.col + c.colspan for c in self.cells), default=0)


def parse(otsl: str) -> Grid:
    text = FENCE_RE.sub("", otsl.strip())
    grid = Grid()
    matches = list(TAG_RE.finditer(text))
    if not matches:
        raise ValueError("no OTSL tags found; expected <fcel>/<ecel>/<lcel>/<ucel>/<xcel>/<nl>")

    for index, match in enumerate(matches):
        tag = match.group(1)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.end():end].strip()

        if tag == "nl":
            grid.row += 1
            continue

        row = grid.row
        col = grid.first_free_col(row)

        if tag == "lcel":
            left = grid.occupied.get((row, col - 1))
            if left is None:
                raise ValueError(f"<lcel> at row {row} has no cell to its left")
            left.colspan += 1
            grid.place(Cell(text=left.text, row=row, col=col, origin=False))
            continue
        if tag in ("ucel", "xcel"):
            above = grid.occupied.get((row - 1, col))
            if above is None:
                raise ValueError(f"<{tag}> at row {row}, col {col} has no cell above")
            above.rowspan += 1
            if tag == "xcel":
                above.colspan += 1
            cell = Cell(text=above.text, row=row, col=col, origin=False)
            if tag == "xcel":
                cell.colspan = 1
            grid.place(cell)
            continue

        grid.place(Cell(text=content, row=row, col=col))

    if not grid.cells:
        raise ValueError("no cells parsed from OTSL input")
    return grid


def to_markdown(grid: Grid, merge: str = "repeat") -> str:
    n_cols = grid.n_cols()
    lines: list[str] = []
    for row in range(grid.n_rows()):
        values: list[str] = []
        for col in range(n_cols):
            cell = grid.occupied.get((row, col))
            if cell is None:
                values.append("")
            elif merge == "repeat" or cell.origin:
                values.append(cell.text.replace("|", "\\|").replace("\n", " "))
            else:
                values.append("")
        lines.append("| " + " | ".join(values) + " |")
        if row == 0:
            lines.append("| " + " | ".join("---" for _ in range(n_cols)) + " |")
    return "\n".join(lines)


def to_html(grid: Grid) -> str:
    def esc(value: str) -> str:
        return (
            value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    lines = ["<table>"]
    for row in range(grid.n_rows()):
        lines.append("  <tr>")
        for cell in grid.cells:
            if cell.row != row or not cell.origin:
                continue
            attrs = ""
            if cell.colspan > 1:
                attrs += f' colspan="{cell.colspan}"'
            if cell.rowspan > 1:
                attrs += f' rowspan="{cell.rowspan}"'
            lines.append(f"    <td{attrs}>{esc(cell.text)}</td>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", help="file with OTSL markup (default: stdin)")
    parser.add_argument("-f", "--format", choices=("md", "html"), default="md")
    parser.add_argument("-m", "--merge", choices=("repeat", "blank"), default="repeat",
                        help="how Markdown renders merged continuation cells (default: repeat)")
    args = parser.parse_args()

    raw = open(args.path, encoding="utf-8").read() if args.path else sys.stdin.read()
    try:
        grid = parse(raw)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(to_html(grid) if args.format == "html" else to_markdown(grid, args.merge))
    return 0


if __name__ == "__main__":
    sys.exit(main())