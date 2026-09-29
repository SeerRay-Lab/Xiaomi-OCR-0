# Shared postprocessing

`document.py` normalizes whole-page math, trims periodic repeated tails and converts OTSL.
`region.py` contains the original evaluation region assembly, including label filtering,
formula-number merging, text joining and bullet formatting. MCP and the browser Demo
use these same functions; streaming previews are replaced by server-assembled results.

`otsl.py` converts OTSL tables. `repetition_guard.py` detects suspicious repetition for
optional evaluation retries. `markdown_merge.py` and `assemble_markdown.py` preserve the
configurable cached-result assembly tools for offline evaluation.
