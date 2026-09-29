#!/usr/bin/env python3
"""Assemble Markdown and optional cropped image assets from cached results.jsonl."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from postprocess.markdown_merge import (
    WORD_VALIDATION_BACKEND,
    MergeOptions,
    MergeOutput,
    map_label,
    merge_page,
)


ASSET_DIR_NAME = "assets"


@dataclass
class ImageAsset:
    image_path: str
    source_path: Path
    markdown_name: str
    region_ordinal: int
    region_index: int
    label: str
    requested_bbox: list[float]
    relative_path: str
    actual_bbox: list[int] | None = None
    image_size: list[int] | None = None


@dataclass
class PreparedPage:
    line_number: int
    image_path: str
    markdown_name: str
    layout_result: list[dict[str, Any]]
    stored_markdown: str | None
    baseline_v2_markdown: str
    assembled_markdown: str
    merge_output: MergeOutput
    assets: list[ImageAsset]
    source_row: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".assemble.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".assemble.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_save_png(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=".crop.",
            suffix=".png.tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            image.save(file, format="PNG")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _source_image_path(image_path: str, image_dir: Path | None) -> Path:
    if image_dir is None:
        return Path(image_path)
    return image_dir / Path(image_path).name


def _asset_filename(image_path: str, region_ordinal: int, label: str) -> str:
    page_hash = hashlib.sha1(image_path.encode("utf-8")).hexdigest()[:16]
    safe_label = "chart" if label == "chart" else "image"
    return f"{page_hash}_r{region_ordinal:04d}_{safe_label}.png"


def _prepare_layout_and_assets(
    image_path: str,
    markdown_name: str,
    layout_result: list[dict[str, Any]],
    image_mode: str,
    image_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[ImageAsset]]:
    prepared_layout = deepcopy(layout_result)
    assets = []
    if image_mode == "none":
        return prepared_layout, assets
    for ordinal, region in enumerate(prepared_layout):
        native_label = region.get("label", "text")
        if map_label(native_label) != "image":
            continue
        bbox = region.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(
                f"{image_path}: image region {ordinal} has invalid bbox_2d: {bbox!r}"
            )
        try:
            requested_bbox = [float(value) for value in bbox]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{image_path}: image region {ordinal} has non-numeric bbox_2d: {bbox!r}"
            ) from error
        filename = _asset_filename(image_path, ordinal, native_label)
        relative_path = f"{ASSET_DIR_NAME}/{filename}"
        alt_text = "chart" if native_label == "chart" else "image"
        region["_asset_markdown"] = f"![{alt_text}]({relative_path})"
        assets.append(
            ImageAsset(
                image_path=image_path,
                source_path=_source_image_path(image_path, image_dir),
                markdown_name=markdown_name,
                region_ordinal=ordinal,
                region_index=int(region.get("index", ordinal)),
                label=native_label,
                requested_bbox=requested_bbox,
                relative_path=relative_path,
            )
        )
    return prepared_layout, assets


def parse_row(
    line: str,
    line_number: int,
    merge_options: MergeOptions,
    image_mode: str,
    image_placement: str,
    image_dir: Path | None,
) -> PreparedPage:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"line {line_number}: invalid JSON: {error}") from error
    if not isinstance(row, dict):
        raise ValueError(f"line {line_number}: top-level value must be an object")
    image_path = row.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError(f"line {line_number}: image_path must be a non-empty string")
    layout_result = row.get("layout_result")
    if not isinstance(layout_result, list):
        raise ValueError(f"line {line_number}: layout_result must be a list")
    for region_number, region in enumerate(layout_result):
        if not isinstance(region, dict):
            raise ValueError(
                f"line {line_number}: layout_result[{region_number}] must be an object"
            )
    stored_markdown = row.get("markdown_result")
    if stored_markdown is not None and not isinstance(stored_markdown, str):
        raise ValueError(f"line {line_number}: markdown_result must be a string or null")
    markdown_name = Path(image_path).stem + ".md"
    prepared_layout, assets = _prepare_layout_and_assets(
        image_path,
        markdown_name,
        layout_result,
        image_mode,
        image_dir,
    )
    baseline_v2_markdown = merge_page(
        layout_result,
        MergeOptions(include_reference=False),
    ).markdown
    merge_output = merge_page(prepared_layout, merge_options)
    if image_mode == "crop" and image_placement == "append" and assets:
        image_markdown = "\n\n".join(
            f"![{'chart' if asset.label == 'chart' else 'image'}]({asset.relative_path})"
            for asset in assets
        )
        if merge_output.markdown:
            merge_output.markdown += "\n\n" + image_markdown
        else:
            merge_output.markdown = image_markdown
    return PreparedPage(
        line_number=line_number,
        image_path=image_path,
        markdown_name=markdown_name,
        layout_result=layout_result,
        stored_markdown=stored_markdown,
        baseline_v2_markdown=baseline_v2_markdown,
        assembled_markdown=merge_output.markdown,
        merge_output=merge_output,
        assets=assets,
        source_row=row,
    )


def load_pages(
    results_jsonl: Path,
    merge_options: MergeOptions,
    image_mode: str,
    image_placement: str,
    image_dir: Path | None,
) -> list[PreparedPage]:
    if not results_jsonl.is_file():
        raise FileNotFoundError(f"results.jsonl does not exist: {results_jsonl}")
    pages = []
    with results_jsonl.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if line.strip():
                pages.append(
                    parse_row(
                        line,
                        line_number,
                        merge_options,
                        image_mode,
                        image_placement,
                        image_dir,
                    )
                )
    if not pages:
        raise ValueError(f"results.jsonl has no records: {results_jsonl}")
    image_counts = Counter(page.image_path for page in pages)
    duplicate_images = [path for path, count in image_counts.items() if count > 1]
    if duplicate_images:
        raise ValueError(f"duplicate image_path values: {duplicate_images[:20]}")
    markdown_counts = Counter(page.markdown_name for page in pages)
    duplicate_names = [name for name, count in markdown_counts.items() if count > 1]
    if duplicate_names:
        raise ValueError(f"duplicate Markdown filenames: {duplicate_names[:20]}")
    asset_counts = Counter(asset.relative_path for page in pages for asset in page.assets)
    duplicate_assets = [path for path, count in asset_counts.items() if count > 1]
    if duplicate_assets:
        raise ValueError(f"duplicate asset paths: {duplicate_assets[:20]}")
    return pages


def selected_markdown(page: PreparedPage, markdown_source: str) -> str:
    if markdown_source == "stored":
        if page.stored_markdown is None:
            raise ValueError(
                f"line {page.line_number}: stored markdown_result is missing"
            )
        return page.stored_markdown
    return page.assembled_markdown


def output_rows(
    pages: list[PreparedPage], markdown_source: str
) -> Iterable[dict[str, Any]]:
    for page in pages:
        row = dict(page.source_row)
        row["markdown_result"] = selected_markdown(page, markdown_source)
        yield row


def change_rows(pages: list[PreparedPage]) -> Iterable[dict[str, Any]]:
    for page in pages:
        for event in page.merge_output.events:
            yield {
                "image_path": page.image_path,
                "markdown_name": page.markdown_name,
                **event,
            }


def asset_rows(pages: list[PreparedPage]) -> Iterable[dict[str, Any]]:
    for page in pages:
        for asset in page.assets:
            yield {
                "image_path": asset.image_path,
                "resolved_source_path": str(asset.source_path),
                "markdown_name": asset.markdown_name,
                "region_ordinal": asset.region_ordinal,
                "region_index": asset.region_index,
                "label": asset.label,
                "requested_bbox": asset.requested_bbox,
                "actual_bbox": asset.actual_bbox,
                "image_size": asset.image_size,
                "relative_path": asset.relative_path,
            }


def check_output_directory(
    results_jsonl: Path,
    output_dir: Path,
    pages: list[PreparedPage],
    overwrite: bool,
) -> None:
    if output_dir.resolve() == results_jsonl.resolve().parent:
        raise ValueError("output-dir must differ from the source results.jsonl directory")
    expected_markdown = {page.markdown_name for page in pages}
    expected_assets = {
        Path(asset.relative_path).name for page in pages for asset in page.assets
    }
    if not output_dir.is_dir():
        return
    existing_markdown = {path.name for path in output_dir.glob("*.md")}
    extra_markdown = sorted(existing_markdown - expected_markdown)
    if extra_markdown:
        raise ValueError(
            f"output-dir contains {len(extra_markdown)} stale Markdown files: "
            f"{extra_markdown[:20]}"
        )
    asset_dir = output_dir / ASSET_DIR_NAME
    existing_assets = (
        {path.name for path in asset_dir.glob("*.png")} if asset_dir.is_dir() else set()
    )
    extra_assets = sorted(existing_assets - expected_assets)
    if extra_assets:
        raise ValueError(
            f"output-dir contains {len(extra_assets)} stale image assets: {extra_assets[:20]}"
        )
    occupied = sorted(existing_markdown & expected_markdown)
    occupied.extend(sorted(existing_assets & expected_assets))
    for filename in (
        "results.jsonl",
        "assembly_report.json",
        "changes.jsonl",
        "assets_manifest.jsonl",
    ):
        if (output_dir / filename).exists():
            occupied.append(filename)
    if occupied and not overwrite:
        raise FileExistsError(
            f"output files already exist; use --overwrite: {occupied[:20]}"
        )


def _merge_config_dict(
    options: MergeOptions,
    image_mode: str,
    image_padding: int,
    bbox_space: str,
    image_placement: str,
) -> dict[str, Any]:
    return {
        **asdict(options),
        "image_mode": image_mode,
        "image_padding": image_padding,
        "bbox_space": bbox_space,
        "image_placement": image_placement,
        "asset_dir": ASSET_DIR_NAME,
    }


def build_report(
    results_jsonl: Path,
    pages: list[PreparedPage],
    markdown_source: str,
    merge_options: MergeOptions,
    image_mode: str,
    image_padding: int,
    bbox_space: str,
    image_placement: str,
) -> dict[str, Any]:
    regions = [region for page in pages for region in page.layout_result]
    task_counts = Counter(region.get("task_type") for region in regions)
    cached_regions = [region for region in regions if "merged" in region]
    blank_cached_regions = [
        region for region in cached_regions if not str(region.get("merged") or "").strip()
    ]
    stored_missing = [page for page in pages if page.stored_markdown is None]
    baseline_different = [
        page
        for page in pages
        if page.stored_markdown is not None
        and page.stored_markdown != page.baseline_v2_markdown
    ]
    output_different_v2 = [
        page for page in pages if page.assembled_markdown != page.baseline_v2_markdown
    ]
    output_different_stored = [
        page
        for page in pages
        if page.stored_markdown is not None
        and page.assembled_markdown != page.stored_markdown
    ]
    blank_baseline_pages = [page for page in pages if not page.baseline_v2_markdown.strip()]
    blank_output_pages = [
        page for page in pages if not selected_markdown(page, markdown_source).strip()
    ]
    event_counts = Counter(
        f"{event['pass']}:{event['action']}"
        for page in pages
        for event in page.merge_output.events
    )
    merge_stats = Counter()
    for page in pages:
        merge_stats.update(page.merge_output.stats)
    merge_path = Path(__file__).with_name("markdown_merge.py")
    assembler_path = Path(__file__)
    config = _merge_config_dict(
        merge_options,
        image_mode,
        image_padding,
        bbox_space,
        image_placement,
    )
    config_json = json.dumps(config, ensure_ascii=False, sort_keys=True)
    return {
        "source_results_jsonl": str(results_jsonl.resolve()),
        "source_sha256": sha256(results_jsonl),
        "merge_module": str(merge_path.resolve()),
        "merge_module_sha256": sha256(merge_path),
        "assembler_module": str(assembler_path.resolve()),
        "assembler_module_sha256": sha256(assembler_path),
        "config": config,
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "word_validation_backend": WORD_VALIDATION_BACKEND,
        "markdown_source": markdown_source,
        "pages": len(pages),
        "regions": len(regions),
        "task_counts": dict(task_counts),
        "cached_merged_regions": len(cached_regions),
        "blank_cached_regions": len(blank_cached_regions),
        "blank_v2_baseline_pages": len(blank_baseline_pages),
        "blank_output_markdown_pages": len(blank_output_pages),
        "stored_markdown_missing": len(stored_missing),
        "v2_baseline_matches_stored": len(pages) - len(stored_missing) - len(baseline_different),
        "v2_baseline_differs_stored": len(baseline_different),
        "output_matches_v2_baseline": len(pages) - len(output_different_v2),
        "output_differs_v2_baseline": len(output_different_v2),
        "output_matches_stored": len(pages) - len(stored_missing) - len(output_different_stored),
        "output_differs_stored": len(output_different_stored),
        "planned_image_assets": sum(len(page.assets) for page in pages),
        "event_counts": dict(event_counts),
        "merge_stats": dict(merge_stats),
        "baseline_difference_examples": [page.image_path for page in baseline_different[:20]],
        "output_difference_examples": [page.image_path for page in output_different_v2[:20]],
        "blank_page_examples": [page.image_path for page in blank_output_pages[:20]],
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"pages                    : {report['pages']}")
    print(f"regions                  : {report['regions']}")
    print(f"task counts              : {report['task_counts']}")
    print(f"cached merged regions    : {report['cached_merged_regions']}")
    print(f"blank cached regions     : {report['blank_cached_regions']}")
    print(f"Markdown source          : {report['markdown_source']}")
    print(f"v2 baseline == stored    : {report['v2_baseline_matches_stored']}")
    print(f"v2 baseline != stored    : {report['v2_baseline_differs_stored']}")
    print(f"output == v2 baseline    : {report['output_matches_v2_baseline']}")
    print(f"output != v2 baseline    : {report['output_differs_v2_baseline']}")
    print(f"blank output pages       : {report['blank_output_markdown_pages']}")
    print(f"planned image assets     : {report['planned_image_assets']}")
    print(f"word validation          : {report['word_validation_backend']}")
    print(f"event counts             : {report['event_counts']}")
    if report["baseline_difference_examples"]:
        print("baseline diff examples   :", report["baseline_difference_examples"][:5])
    if report["output_difference_examples"]:
        print("output diff examples     :", report["output_difference_examples"][:5])
    if report["blank_page_examples"]:
        print("blank page examples      :", report["blank_page_examples"])


def _preflight_image_sources(pages: list[PreparedPage]) -> None:
    missing = sorted(
        {
            str(asset.source_path)
            for page in pages
            for asset in page.assets
            if not asset.source_path.is_file()
        }
    )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} source images are missing; use --image-dir to remap by basename "
            f"or --image-mode none. Examples: {missing[:20]}"
        )


def _crop_box(
    requested_bbox: list[float],
    width: int,
    height: int,
    padding: int,
    bbox_space: str,
) -> list[int]:
    x1, y1, x2, y2 = requested_bbox
    if not all(math.isfinite(value) for value in requested_bbox):
        raise ValueError(f"bbox contains a non-finite coordinate: {requested_bbox}")
    if bbox_space == "normalized-1000":
        x_scale = width / 1000.0
        y_scale = height / 1000.0
    elif bbox_space == "pixel":
        x_scale = 1.0
        y_scale = 1.0
    else:
        raise ValueError(f"unsupported bbox space: {bbox_space}")
    left = max(0, math.floor(min(x1, x2) * x_scale) - padding)
    top = max(0, math.floor(min(y1, y2) * y_scale) - padding)
    right = min(width, math.ceil(max(x1, x2) * x_scale) + padding)
    bottom = min(height, math.ceil(max(y1, y2) * y_scale) + padding)
    if right <= left or bottom <= top:
        raise ValueError(
            "empty crop after coordinate conversion and clamping: "
            f"requested={requested_bbox}, bbox_space={bbox_space}, "
            f"image_size={[width, height]}"
        )
    return [left, top, right, bottom]


def write_image_assets(
    output_dir: Path,
    pages: list[PreparedPage],
    image_padding: int,
    bbox_space: str,
) -> None:
    total_assets = sum(len(page.assets) for page in pages)
    if not total_assets:
        return
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required for --image-mode crop") from error
    _preflight_image_sources(pages)
    completed = 0
    for page in pages:
        if not page.assets:
            continue
        source_path = page.assets[0].source_path
        with Image.open(source_path) as image:
            image.load()
            width, height = image.size
            for asset in page.assets:
                try:
                    actual_bbox = _crop_box(
                        asset.requested_bbox,
                        width,
                        height,
                        image_padding,
                        bbox_space,
                    )
                except ValueError as error:
                    raise ValueError(
                        f"{asset.image_path}: region ordinal={asset.region_ordinal} "
                        f"index={asset.region_index} label={asset.label}: {error}"
                    ) from error
                crop = image.crop(actual_bbox)
                if crop.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
                    crop = crop.convert("RGB")
                atomic_save_png(output_dir / asset.relative_path, crop)
                asset.actual_bbox = actual_bbox
                asset.image_size = [width, height]
                completed += 1
        if completed and completed % 250 == 0:
            print(f"cropped image assets      : {completed}/{total_assets}")
    print(f"cropped image assets      : {completed}/{total_assets}")


def write_outputs(
    output_dir: Path,
    pages: list[PreparedPage],
    report: dict[str, Any],
    markdown_source: str,
    image_padding: int,
    bbox_space: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_image_assets(output_dir, pages, image_padding, bbox_space)
    for page in pages:
        atomic_write_text(
            output_dir / page.markdown_name,
            selected_markdown(page, markdown_source),
        )
    atomic_write_jsonl(
        output_dir / "results.jsonl",
        output_rows(pages, markdown_source),
    )
    atomic_write_jsonl(output_dir / "changes.jsonl", change_rows(pages))
    atomic_write_jsonl(output_dir / "assets_manifest.jsonl", asset_rows(pages))
    atomic_write_text(
        output_dir / "assembly_report.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--markdown-source",
        choices=("stored", "rebuild"),
        default="rebuild",
    )
    parser.add_argument(
        "--image-mode",
        choices=("crop", "none"),
        default="crop",
    )
    parser.add_argument(
        "--image-placement",
        choices=("append", "ordered"),
        default="append",
        help="append assets after scored content, or insert them at layout positions",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="remap source image paths to this directory by basename",
    )
    parser.add_argument("--image-padding", type=int, default=0)
    parser.add_argument(
        "--bbox-space",
        choices=("normalized-1000", "pixel"),
        default="normalized-1000",
        help="coordinate space used by layout_result[*].bbox_2d",
    )
    parser.add_argument("--text-profile", choices=("v2", "paddle"), default="v2")
    parser.add_argument(
        "--text-compaction",
        choices=("off", "adaptive"),
        default="off",
        help="collapse extreme plain-text fragmentation without changing evaluator code",
    )
    parser.add_argument("--title-profile", choices=("v2", "numbered"), default="v2")
    parser.add_argument(
        "--hyphen-merge",
        choices=("v2", "adjacent", "off"),
        default="v2",
    )
    parser.add_argument(
        "--bullet-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validate-tables",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-footnote",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include cached footnote regions in Markdown",
    )
    parser.add_argument(
        "--include-aside-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include cached aside_text regions in Markdown",
    )
    parser.add_argument(
        "--include-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include cached reference regions in Markdown",
    )
    parser.add_argument(
        "--inline-formula-policy",
        choices=("display", "dedupe", "parent-aware", "paddle-overlap"),
        default="display",
        help="keep inline formulas as display blocks, use parent-aware policies, or apply PaddleOCR-VL overlap filtering",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-v2-match", action="store_true")
    parser.add_argument("--require-stored-baseline-match", action="store_true")
    parser.add_argument("--require-rebuild-match", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.image_padding < 0:
        raise ValueError("--image-padding must be >= 0")
    if args.image_dir is not None and not args.image_dir.is_dir():
        raise FileNotFoundError(f"--image-dir does not exist: {args.image_dir}")
    if args.markdown_source == "stored" and args.image_mode != "none":
        raise ValueError("--markdown-source stored requires --image-mode none")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    merge_options = MergeOptions(
        text_profile=args.text_profile,
        text_compaction=args.text_compaction,
        title_profile=args.title_profile,
        hyphen_merge=args.hyphen_merge,
        bullet_repair=args.bullet_repair,
        include_images=args.image_mode == "crop" and args.image_placement == "ordered",
        include_footnote=args.include_footnote,
        include_aside_text=args.include_aside_text,
        include_reference=args.include_reference,
        inline_formula_policy=args.inline_formula_policy,
        validate_tables=args.validate_tables,
    )
    pages = load_pages(
        args.results_jsonl,
        merge_options,
        args.image_mode,
        args.image_placement,
        args.image_dir,
    )
    report = build_report(
        args.results_jsonl,
        pages,
        args.markdown_source,
        merge_options,
        args.image_mode,
        args.image_padding,
        args.bbox_space,
        args.image_placement,
    )
    print_report(report)
    require_stored = args.require_stored_baseline_match or args.require_rebuild_match
    if require_stored and (
        report["stored_markdown_missing"] or report["v2_baseline_differs_stored"]
    ):
        raise SystemExit("[FAIL] v2 baseline does not exactly match stored Markdown")
    if args.require_v2_match and report["output_differs_v2_baseline"]:
        raise SystemExit("[FAIL] v9 output does not exactly match v2 baseline")
    if args.dry_run:
        print("[OK] dry-run completed; no files written")
        return
    check_output_directory(args.results_jsonl, args.output_dir, pages, args.overwrite)
    write_outputs(
        args.output_dir,
        pages,
        report,
        args.markdown_source,
        args.image_padding,
        args.bbox_space,
    )
    print(f"[OK] Markdown assembled: {args.output_dir}")


if __name__ == "__main__":
    main()
