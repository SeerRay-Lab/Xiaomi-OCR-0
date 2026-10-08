# Shared postprocessing

`document.py` normalizes whole-page math, trims periodic repeated tails and converts OTSL.
`region.py` contains the original evaluation region assembly, including label filtering,
formula-number merging, text joining and bullet formatting. MCP and the browser Demo
use these same functions; streaming previews are replaced by server-assembled results.

MCP postprocessing paths:

- Whole-page OCR (including PDF pages and the region-mode fallback) calls
  `finalize_document`: repeated-tail cleanup → math delimiter normalization →
  OTSL-to-HTML conversion, matching `pipeline/e2e_img2md.py`.
- Region OCR converts each table from OTSL to HTML, keeping the raw output if
  conversion fails, then calls `region.merge_page_to_markdown`, matching
  `pipeline/twostage_img2md.py`. Assembly sorts by reading-order index, filters
  image/header/footer labels, cleans and formats region content, merges formula
  numbers, joins text blocks, and formats bullet points.
- MCP layout detection uses PaddleX native postprocessing followed by
  `pipeline.layout_detect_paddlex.serialize_regions`. The two-stage script's
  `apply_layout_postprocess` is for its Transformers layout backend and is not
  applied again to PaddleX results.

The pipelines' optional `--retry-repetitive` inference pass is not enabled in MCP;
the shared output cleanup above still runs. MCP rejects responses truncated by
the token limit before assembly.

`otsl.py` converts OTSL tables. `repetition_guard.py` detects suspicious repetition for
optional evaluation retries. `markdown_merge.py` and `assemble_markdown.py` preserve the
configurable cached-result assembly tools for offline evaluation.
