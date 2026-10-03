/* Xiaomi-OCR-0 · split compare + typewriter */

const DEFAULT_CFG = {
  max_tokens: 4096,
  temperature: 0,
};
const LS_KEY = "xiaomi_ocr_demo_cfg";
const THEME_KEY = "xiaomi_ocr_demo_theme";

const state = {
  items: [],
  activeId: null,
  typingTimer: null,
  typedDone: false,
  custom: null,
  busy: false,
  mode: "page", // page | region
  task: "document", // document | kie | vqa
  pdfUrl: null,
};

const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* theme */
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  localStorage.setItem(THEME_KEY, t);
  $("btnTheme").textContent = t === "dark" ? "☀" : "◐";
}
(function () {
  let t = localStorage.getItem(THEME_KEY);
  if (!t) t = "light";
  applyTheme(t);
})();
$("btnTheme").onclick = function () {
  applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
};

/* config */
function loadCfg() {
  try {
    const s = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
    const o = {};
    Object.keys(DEFAULT_CFG).forEach(function (k) {
      o[k] = s[k] != null && s[k] !== "" ? s[k] : DEFAULT_CFG[k];
    });
    return o;
  } catch (e) { return Object.assign({}, DEFAULT_CFG); }
}
function saveCfg(c) { localStorage.setItem(LS_KEY, JSON.stringify(c)); }
function cfg() { return loadCfg(); }

/* ── optional result views: RAW / JSON stay hidden unless enabled in settings ── */
const VIEW_KEY = "xiaomi-ocr-show-views";
function showViews() { return localStorage.getItem(VIEW_KEY) !== "0"; }
function recordResult(text, meta) {
  state.lastText = String(text || "");
  state.lastMeta = meta || null;
  state.lastTask = state.task;
  $("btnCopy").disabled = !state.lastText;
  $("btnDownload").disabled = !state.lastText;
}
function paintResult() {
  const el = $("resText");
  if (!el || !state.lastText) return;
  const v = state.view || "pretty";
  if (v === "raw") {
    el.classList.remove("markdown");
    el.classList.remove("typing");
    el.classList.add("source-view");
    el.textContent = state.lastText;
  } else if (v === "json") {
    el.classList.remove("markdown");
    el.classList.remove("typing");
    el.classList.add("source-view");
    el.textContent = JSON.stringify({
      model: (state.lastMeta && state.lastMeta.model) || "SeerRay-Lab/Xiaomi-OCR-0",
      latency_ms: state.lastMeta ? state.lastMeta.latency_ms : null,
      usage: (state.lastMeta && state.lastMeta.usage) || null,
      content: state.lastText,
    }, null, 2);
  } else {
    el.classList.remove("typing", "source-view");
    el.classList.add("markdown");
    el.innerHTML = state.lastTask === "kie" ? "<pre><code>" + escapeHtml(state.lastText) + "</code></pre>" : renderMarkdown(normalizeOcrOutput(state.lastText));
  }
  const pane = $("resPane");
  if (pane) pane.scrollTop = 0;
}
function setView(v) {
  state.view = v;
  document.querySelectorAll("#viewSeg button").forEach(function (b) {
    b.classList.toggle("active", b.dataset.view === v);
    b.setAttribute("aria-pressed", String(b.dataset.view === v));
  });
  paintResult();
}
function applyViewVisibility() {
  const seg = $("viewSeg");
  if (!seg) return;
  const on = showViews();
  seg.hidden = !on;
  if (!on) {
    state.view = "pretty";
    document.querySelectorAll("#viewSeg button").forEach(function (b) {
      b.classList.toggle("active", b.dataset.view === "pretty");
    });
  }
}
(function initViewSeg() {
  const seg = $("viewSeg");
  if (seg) {
    seg.addEventListener("click", function (e) {
      const b = e.target.closest("button[data-view]");
      if (b) setView(b.dataset.view);
    });
  }
  const box = $("cfgShowViews");
  if (box) box.checked = showViews();
  applyViewVisibility();
})();

function openDrawer() {
  const c = cfg();
  $("cfgMaxTokens").value = c.max_tokens;
  $("drawer").hidden = false;
  $("backdrop").hidden = false;
  document.querySelector(".shell").inert = true;
  $("cfgMaxTokens").focus();
  $("drawer").classList.add("open");
  $("backdrop").classList.add("open");
}
function closeDrawer() {
  $("drawer").hidden = true;
  $("backdrop").hidden = true;
  document.querySelector(".shell").inert = false;
  $("btnSettings").focus();
  $("drawer").classList.remove("open");
  $("backdrop").classList.remove("open");
}
$("btnSettings").onclick = openDrawer;
$("btnCloseSettings").onclick = closeDrawer;
$("btnCancelCfg").onclick = closeDrawer;
$("backdrop").onclick = closeDrawer;
$("btnSaveCfg").onclick = function () {
  if (!$("cfgMaxTokens").reportValidity()) return;
  saveCfg({
    max_tokens: Number($("cfgMaxTokens").value || 4096),
    temperature: 0,
  });
  localStorage.setItem(VIEW_KEY, $("cfgShowViews") && $("cfgShowViews").checked ? "1" : "0");
  applyViewVisibility();
  paintResult();
  closeDrawer();
};


function setLed(mode) {
  const led = $("ledLive");
  if (!led) return;
  led.classList.remove("on", "err");
  if (mode === "on") led.classList.add("on");
  else if (mode === "err") led.classList.add("err");
}

/* parse mode: page (whole-image stream) | region (layout + parallel) */
(function initModeSeg() {
  const seg = $("modeSeg");
  if (!seg) return;
  seg.addEventListener("click", function (e) {
    const b = e.target.closest("button");
    if (!b || state.busy || state.task !== "document" || b.disabled) return;
    state.mode = b.dataset.mode;
    seg.querySelectorAll("button").forEach(function (x) {
      x.classList.toggle("active", x.dataset.mode === state.mode);
    });
    $("stageHint").textContent = state.mode === "region"
      ? "REGION · layout + parallel"
      : "CLICK TO PARSE";
    updateTaskInputs();
    resetResult();
    clearError();
  });
})();

function updateTaskInputs() {
  const isDoc = state.task === "document";
  const isPdf = !!(state.custom && state.custom.isPdf);
  const region = isDoc && state.mode === "region";
  $("docPromptWrap").hidden = !isDoc || region;
  $("kieFieldsWrap").hidden = state.task !== "kie";
  $("vqaQuestionWrap").hidden = state.task !== "vqa";
  $("regionPromptNote").hidden = !region;
  document.querySelectorAll("#modeSeg button").forEach(function (b) {
    b.disabled = !isDoc || state.busy;
    b.setAttribute("aria-pressed", String(b.dataset.mode === state.mode));
    b.classList.toggle("active", b.dataset.mode === state.mode);
  });
  if (isPdf && !isDoc) {
    state.task = "document";
    document.querySelectorAll("#taskSeg button").forEach(function (b) { b.classList.toggle("active", b.dataset.task === "document"); });
    updateTaskInputs();
    showError("PDF 目前用于文档解析；KIE 和 VQA 请上传或选择图像。");
    return;
  }
  document.querySelectorAll("#taskSeg button").forEach(b => { b.classList.toggle("active", b.dataset.task === state.task); b.setAttribute("aria-pressed", String(b.dataset.task === state.task)); });
  $("stageHint").textContent = isPdf ? (region ? "PDF · REGION PARSING" : "PDF · PAGE PARSING") : (region ? "REGION · layout + parallel" : "CLICK TO PARSE");
}

(function initTaskSeg() {
  const seg = $("taskSeg");
  if (!seg) return;
  seg.addEventListener("click", function (e) {
    const b = e.target.closest("button[data-task]");
    if (!b || state.busy) return;
    if (state.custom && state.custom.isPdf && b.dataset.task !== "document") { showError("PDF 支持文档解析；字段提取和问答请选择图像。"); return; }
    state.task = b.dataset.task;
    if (state.task !== "document") state.mode = "page";
    seg.querySelectorAll("button").forEach(function (x) { x.classList.toggle("active", x === b); });
    document.querySelectorAll("#modeSeg button").forEach(function (x) { x.classList.toggle("active", x.dataset.mode === state.mode); });
    updateTaskInputs();
    seg.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x.dataset.task === state.task)));
    resetResult();
    clearError();
  });
  $("docPrompt").addEventListener("input", function () { if (state.task === "document") resetResult(); });
  $("kieFields").addEventListener("input", resetResult);
  $("vqaQuestion").addEventListener("input", resetResult);
})();

async function runRegionOcr(dataUrl) {
  const c = cfg();
  const body = {
    image_base64: dataUrl,
    max_tokens: Math.max(1, Math.min(c.max_tokens || 2048, 4096)),
    concurrency: 16,
  };
  const res = await fetch("/api/parse/regions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || ("HTTP " + res.status));
  return data;
}

function firePulse() {
  const p = $("pulse");
  p.classList.remove("go");
  void p.offsetWidth;
  p.classList.add("go");
}

/* ── markdown + katex ── */
function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/**
 * OTSL (Optical Structure Recognition Language) → HTML table
 *
 * Tags (as emitted by dots.ocr / PaddleOCR-VL / qwen OCR pipelines):
 *   <fcel>  filled cell — content follows until next tag
 *   <ecel>  empty cell
 *   <lcel>  null-left cell — this grid slot is covered by the cell to its left (colspan++)
 *   <ucel>  null-up cell   — this grid slot is covered by the cell above (rowspan++)
 *   <nl>    row break
 *
 * Some models also emit tab-separated rows with tags glued to content
 * (e.g. `本集团<lcel>` or `<ucel><ucel>\t2024-12-31`). Both shapes are accepted.
 */
function parseOtsl(text) {
  // Normalize tags to a canonical token stream.
  // Also honor real newlines as <nl> when the model used tabs+newlines.
  let src = String(text || "");
  src = src
    .replace(/<nl\s*\/?>/gi, "\n")
    .replace(/\r\n?/g, "\n");

  // Tokenize: tags first, then runs of plain text / tabs / newlines
  const TAG = /<(fcel|ecel|lcel|ucel)>/i;
  const tokens = [];
  let i = 0;
  while (i < src.length) {
    if (src[i] === "\n") {
      tokens.push({ t: "nl" });
      i++;
      continue;
    }
    if (src[i] === "\t") {
      // Tab is a cell boundary in hybrid mode (text\ttext, or text<tag>\ttext).
      tokens.push({ t: "softbreak" });
      i++;
      continue;
    }
    const rest = src.slice(i);
    const m = rest.match(/^<(fcel|ecel|lcel|ucel)>/i);
    if (m) {
      tokens.push({ t: m[1].toLowerCase() });
      i += m[0].length;
      continue;
    }
    // plain text run until next tag / newline / tab
    let j = i;
    while (j < src.length && src[j] !== "\n" && src[j] !== "\t" && !TAG.test(src.slice(j, j + 6))) {
      j++;
    }
    if (j === i) { i++; continue; } // safety
    tokens.push({ t: "text", v: src.slice(i, j) });
    i = j;
  }

  // Rebuild grid with rowspan/colspan
  // cell: {text, r, c, rowspan, colspan}
  const cells = [];
  let row = 0, col = 0;
  let pendingText = "";
  // occupancy map to find the master cell covering (r,c)
  const occ = new Map(); // "r,c" -> cell ref

  function key(r, c) { return r + "," + c; }

  function placeFilled(text, isEmpty) {
    const cell = {
      text: isEmpty ? "" : text.trim(),
      r: row, c: col,
      rowspan: 1, colspan: 1,
    };
    cells.push(cell);
    occ.set(key(row, col), cell);
    col++;
    pendingText = "";
  }

  function markLeft() {
    // covered by cell to the left — find nearest master on this row
    let c = col - 1;
    while (c >= 0 && !occ.has(key(row, c))) c--;
    const master = c >= 0 ? occ.get(key(row, c)) : null;
    if (master) master.colspan++;
    occ.set(key(row, col), master || { text: "", r: row, c: col, rowspan: 1, colspan: 1, phantom: true });
    col++;
  }

  function markUp() {
    // covered by cell above — walk up in this column
    let r = row - 1;
    while (r >= 0 && !occ.has(key(r, col))) r--;
    const master = r >= 0 ? occ.get(key(r, col)) : null;
    if (master) master.rowspan++;
    occ.set(key(row, col), master || { text: "", r: row, c: col, rowspan: 1, colspan: 1, phantom: true });
    col++;
  }

  function flushPendingAsCell() {
    if (pendingText.trim()) placeFilled(pendingText, false);
  }

  for (const tok of tokens) {
    if (tok.t === "nl") {
      flushPendingAsCell();
      // if the row had no cells at all, still advance
      row++;
      col = 0;
      continue;
    }
    if (tok.t === "softbreak") {
      // Tab boundary: pending text becomes its own cell; consecutive tabs → empty cells
      if (pendingText.trim()) {
        placeFilled(pendingText, false);
      } else if (col > 0 || cells.length) {
        // empty cell between tabs, but not a leading empty at row start
        // only emit if we already have content on this row or previous cell just placed
        // Heuristic: if last placed cell was on this row, emit empty
        const last = cells[cells.length - 1];
        if (last && last.r === row) {
          placeFilled("", true);
        }
      }
      continue;
    }
    if (tok.t === "text") {
      pendingText += tok.v;
      continue;
    }
    if (tok.t === "fcel") {
      // <fcel> means "a filled cell starts; content follows until next tag".
      // If there is pending text (tab-mode: text BEFORE the tag), that text
      // is the cell. If pending is empty, the cell content comes AFTER this
      // tag — do not emit a spurious empty cell.
      if (pendingText.trim()) {
        placeFilled(pendingText, false);
      } else {
        pendingText = ""; // content will accumulate from following text tokens
      }
      continue;
    }
    if (tok.t === "ecel") {
      // empty cell; if there was pending text, treat as filled (tab-mode quirk)
      if (pendingText.trim()) placeFilled(pendingText, false);
      else placeFilled("", true);
      continue;
    }
    if (tok.t === "lcel") {
      // If pending text exists, this is a filled cell that ALSO extends right
      // (e.g. `本集团<lcel>` → header spanning two date columns).
      if (pendingText.trim()) {
        placeFilled(pendingText, false);
        // placeFilled already advanced col by 1; extend one more for the lcel
        markLeft();
      } else {
        markLeft();
      }
      continue;
    }
    if (tok.t === "ucel") {
      flushPendingAsCell();
      markUp();
      continue;
    }
  }
  flushPendingAsCell();

  // Drop phantoms
  const real = cells.filter((c) => !c.phantom);
  if (!real.length) return null;

  // Normalize: max row/col
  const maxR = Math.max(...real.map((c) => c.r + c.rowspan));
  const maxC = Math.max(...real.map((c) => c.c + c.colspan));

  // Emit HTML
  const byRC = new Map();
  real.forEach((c) => byRC.set(key(c.r, c.c), c));

  let html = "<table>";
  for (let r = 0; r < maxR; r++) {
    html += "<tr>";
    let c = 0;
    while (c < maxC) {
      const cell = byRC.get(key(r, c));
      if (cell && cell.r === r && cell.c === c) {
        const rs = cell.rowspan > 1 ? ` rowspan="${cell.rowspan}"` : "";
        const cs = cell.colspan > 1 ? ` colspan="${cell.colspan}"` : "";
        const txt = escapeHtml(cell.text || "");
        html += `<td${rs}${cs}>${txt}</td>`;
        c += cell.colspan;
      } else if (cell) {
        // covered by a master elsewhere — skip
        c += 1;
      } else {
        // hole: emit empty cell to keep grid rectangular
        html += "<td></td>";
        c += 1;
      }
    }
    html += "</tr>";
  }
  html += "</table>";
  return html;
}

function otslToMarkdownTable(t) {
  let src = String(t || "");

  // 1) Keep any well-formed HTML tables the model already emitted
  const htmlTables = [];
  src = src.replace(/<table[\s\S]*?<\/table>/gi, function (m) {
    htmlTables.push(m);
    return "\u0000HT" + (htmlTables.length - 1) + "\u0000";
  });

  // 2) Only convert contiguous OTSL regions, leave surrounding prose alone.
  //    A region = one or more consecutive lines that contain a cell tag.
  const cellTag = /<(?:fcel|ecel|lcel|ucel)>/i;
  const nlTag = /<nl\s*\/?>/i;
  const isOtslLine = function (line) {
    return cellTag.test(line) || nlTag.test(line);
  };

  const lines = src.split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    if (!isOtslLine(lines[i])) {
      out.push(lines[i]);
      i++;
      continue;
    }
    // gather the whole OTSL block
    const block = [];
    while (i < lines.length && isOtslLine(lines[i])) {
      block.push(lines[i]);
      i++;
    }
    const blockText = block.join("\n");
    const tbl = parseOtsl(blockText);
    if (tbl) {
      const id = htmlTables.length;
      htmlTables.push(tbl);
      out.push("\u0000HT" + id + "\u0000");
    } else {
      // fallback: keep raw
      out.push(blockText);
    }
  }

  return out.join("\n").replace(/\u0000HT(\d+)\u0000/g, function (_, i) {
    return htmlTables[+i] || "";
  });
}

function unescapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'").replace(/&amp;/g, "&");
}

function inlineTex(s) {
  // safety: if entities leaked in, restore before KaTeX
  const tex = unescapeHtml(s);
  if (typeof katex === "undefined") return '<span class="math">' + escapeHtml(tex) + "</span>";
  try {
    return katex.renderToString(tex, { displayMode: false, throwOnError: false, output: "html" });
  } catch (e) {
    return '<span class="math">' + escapeHtml(tex) + "</span>";
  }
}

/**
 * 修复 OCR 常见 LaTeX 污染，让 KaTeX 可解析：
 *   \{r\} → {r}          （多余转义花括号）
 *   \{top\} → \top       （OCR 把 \top 读成花括号包字）
 *   \] 单独成行 + (5)  → 合并进 math 块为 \tag{5}
 *   \left\{ \right\}   → 保护，不被误改
 */
function sanitizeLatex(t) {
  let s = String(t || "");
  // 1) protect legitimate \left\{ \right\} delimiters
  s = s.replace(/\\left\\([{\[(])/g, "\x00L$1");
  s = s.replace(/\\right\\([}\])])/g, "\x00R$1");
  // 2) \{top\} → \top
  s = s.replace(/\\\{top\\\}/g, "\\top");
  s = s.replace(/\\\{bot\\\}/g, "\\bot");
  // 3) \{ \} → { }   (extra escapes OCR emits around grouping)
  s = s.replace(/\\\{/g, "{").replace(/\\\}/g, "}");
  // 4) restore
  s = s.replace(/\x00L/g, "\\left\\").replace(/\x00R/g, "\\right\\");
  // 5) merge equation number that OCR put on the next line after \]
  //    "...\]\n\n(5)\n"  →  "...\\tag{5}\n\]\n"
  s = s.replace(/\\]\s*\n+\s*\((\d{1,3})\)\s*\n/g, "\\tag{$1}\n\\]\n");
  return s;
}

// Rebuild only table markup. Model output must never become executable HTML.
function sanitizeTable(html) {
  const template = document.createElement("template");
  template.innerHTML = html;
  const allowed = new Set(["TABLE", "THEAD", "TBODY", "TFOOT", "TR", "TH", "TD", "CAPTION", "BR", "B", "STRONG", "EM", "I", "SUB", "SUP"]);
  function clean(node) {
    if (node.nodeType === 3) return escapeHtml(node.textContent);
    if (node.nodeType !== 1) return "";
    if (!allowed.has(node.tagName)) return escapeHtml(node.textContent);
    const tag = node.tagName.toLowerCase();
    let attrs = "";
    if (tag === "td" || tag === "th") ["colspan", "rowspan"].forEach(key => {
      const value = node.getAttribute(key);
      if (/^[1-9][0-9]{0,2}$/.test(value || "")) attrs += " " + key + '=\"' + value + '\"';
    });
    if (tag === "br") return "<br>";
    return "<" + tag + attrs + ">" + Array.from(node.childNodes).map(clean).join("") + "</" + tag + ">";
  }
  return Array.from(template.content.childNodes).map(clean).join("");
}

function renderMarkdown(md) {
  md = sanitizeLatex(md);
  let src = String(md || "");
  if (/<(?:fcel|ecel|lcel|ucel|nl)\b/i.test(src)) src = otslToMarkdownTable(src);

  const fenced = [];
  src = src.replace(/```([\w-]*)\n([\s\S]*?)```/g, function (_, lang, code) {
    fenced.push(code);
    return "\u0000FENCE" + (fenced.length - 1) + "\u0000";
  });

  const htmls = [];
  src = src.replace(/(<table[\s\S]*?<\/table>)/gi, function (m) {
    htmls.push(sanitizeTable(m));
    return "\u0000HTML" + (htmls.length - 1) + "\u0000";
  });

  // display math
  const mathBlocks = [];
  src = src.replace(/\\\[([\s\S]*?)\\\]/g, function (_, b) {
    mathBlocks.push(b.trim());
    return "\n\u0000MATH" + (mathBlocks.length - 1) + "\u0000\n";
  });
  src = src.replace(/\$\$([\s\S]*?)\$\$/g, function (_, b) {
    mathBlocks.push(b.trim());
    return "\n\u0000MATH" + (mathBlocks.length - 1) + "\u0000\n";
  });

  function inline(s) {
    // Extract math FIRST so < > & are not HTML-escaped before KaTeX.
    const mathBits = [];
    let t = String(s == null ? "" : s);
    t = t.replace(/\\\(([\s\S]*?)\\\)/g, function (_, tex) {
      mathBits.push(inlineTex(tex));
      return "\u0000M" + (mathBits.length - 1) + "\u0000";
    });
    t = t.replace(/\$([^$\n]+)\$/g, function (_, tex) {
      mathBits.push(inlineTex(tex));
      return "\u0000M" + (mathBits.length - 1) + "\u0000";
    });
    t = escapeHtml(t);
    t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
    t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // restore rendered math (already HTML-safe from KaTeX)
    t = t.replace(/\u0000M(\d+)\u0000/g, function (_, i) { return mathBits[+i] || ""; });
    return t;
  }

  const lines = src.split(/\r?\n/);
  const out = [];
  let i = 0, inTable = false, trows = [], list = null;

  function flushList() { if (list) { out.push(list === "ul" ? "</ul>" : "</ol>"); list = null; } }
  function flushTable() {
    if (!inTable) return;
    if (trows.length) {
      const head = trows[0];
      const body = trows.slice(1).filter(function (r) {
        return !r.every(function (c) { return /^:?-{2,}:?$/.test(c.trim()); });
      });
      let html = "<table><thead><tr>";
      head.forEach(function (c) { html += "<th>" + inline(c.trim()) + "</th>"; });
      html += "</tr></thead><tbody>";
      body.forEach(function (r) {
        html += "<tr>";
        r.forEach(function (c) { html += "<td>" + inline(c.trim()) + "</td>"; });
        html += "</tr>";
      });
      out.push(html + "</tbody></table>");
    }
    trows = []; inTable = false;
  }

  while (i < lines.length) {
    const line = lines[i];

    const mathM = line.match(/^\s*\u0000MATH(\d+)\u0000\s*$/);
    if (mathM) {
      flushList(); flushTable();
      const tex = unescapeHtml(mathBlocks[+mathM[1]] || "");
      let rendered;
      if (typeof katex !== "undefined") {
        try {
          rendered = katex.renderToString(tex, { displayMode: true, throwOnError: false, output: "html" });
        } catch (e) { rendered = "<code>" + escapeHtml(tex) + "</code>"; }
      } else {
        rendered = "<code>" + escapeHtml(tex) + "</code>";
      }
      out.push('<div class="math-block">' + rendered + "</div>");
      i++; continue;
    }

    const fenceM = line.match(/^\u0000FENCE(\d+)\u0000$/);
    if (fenceM) {
      flushList(); flushTable();
      out.push("<pre><code>" + escapeHtml(fenced[+fenceM[1]].replace(/\n$/, "")) + "</code></pre>");
      i++; continue;
    }
    const htmlM = line.match(/^\u0000HTML(\d+)\u0000$/);
    if (htmlM) {
      flushList(); flushTable();
      out.push(htmls[+htmlM[1]]);
      i++; continue;
    }

    if (/^\s*\|.*\|\s*$/.test(line)) {
      flushList(); inTable = true;
      trows.push(line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|"));
      i++; continue;
    } else if (inTable) flushTable();

    if (!line.trim()) { flushList(); i++; continue; }

    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      flushList();
      const lv = Math.min(h[1].length, 4);
      out.push("<h" + lv + ">" + inline(h[2]) + "</h" + lv + ">");
      i++; continue;
    }
    if (/^>\s?/.test(line)) {
      flushList();
      out.push("<blockquote>" + inline(line.replace(/^>\s?/, "")) + "</blockquote>");
      i++; continue;
    }
    if (/^[-*+]\s+/.test(line)) {
      if (list !== "ul") { flushList(); list = "ul"; out.push("<ul>"); }
      out.push("<li>" + inline(line.replace(/^[-*+]\s+/, "")) + "</li>");
      i++; continue;
    }
    if (/^\d+\.\s+/.test(line)) {
      if (list !== "ol") { flushList(); list = "ol"; out.push("<ol>"); }
      out.push("<li>" + inline(line.replace(/^\d+\.\s+/, "")) + "</li>");
      i++; continue;
    }
    flushList();
    out.push("<p>" + inline(line) + "</p>");
    i++;
  }
  flushList(); flushTable();
  return out.join("\n");
}

function showError(msg) {
  const b = $("errorBanner");
  b.textContent = msg;
  b.classList.add("show");

}
function clearError() {
  $("errorBanner").classList.remove("show");
}

function stopTyping() {
  if (state.typingTimer) { clearTimeout(state.typingTimer); state.typingTimer = null; }
  $("resText").classList.remove("typing");
}

/**
 * Typewriter over markdown source, then swap to rendered HTML.
 * Looks more "live" for demo; final state is fully rendered (KaTeX etc).
 */
function typeResult(rawText) {
  stopTyping();
  const el = $("resText");
  const full = String(rawText || "");
  recordResult(full, null);
  el.style.display = "block";
  $("resEmpty").style.display = "none";
  $("badgeRight").textContent = "TYPING";
  $("badgeRight").className = "badge on";

  if (reduceMotion) {
    el.classList.remove("typing");
    el.classList.add("markdown");
    el.innerHTML = renderMarkdown(full);
    state.typedDone = true;
    $("badgeRight").textContent = "DONE";
    return;
  }

  el.classList.add("typing");
  el.textContent = "";
  state.typedDone = false;

  // chunk size: finish in ~2.5–4s
  const total = full.length;
  const steps = Math.min(140, Math.max(30, Math.ceil(total / 18)));
  const chunk = Math.max(1, Math.ceil(total / steps));
  let i = 0;

  function tick() {
    i = Math.min(total, i + chunk);
    el.textContent = full.slice(0, i);
    const pane = $("resPane");
    pane.scrollTop = pane.scrollHeight;
    if (i < total) {
      state.typingTimer = setTimeout(tick, 22);
    } else {
      // swap to rendered HTML
      el.classList.remove("typing");
      el.classList.add("markdown");
      el.innerHTML = renderMarkdown(full);
      state.typedDone = true;
      $("badgeRight").textContent = "DONE";
      pane.scrollTop = 0;
    }
  }
  tick();
}

function resetResult() {
  paintPills();
  stopTyping();
  state.typedDone = false;
  state.lastText = "";
  state.lastMeta = null;
  $("btnCopy").disabled = true;
  $("btnDownload").disabled = true;
  $("resText").classList.remove("source-view", "typing");
  $("resText").style.display = "none";
  $("resText").innerHTML = "";
  $("resEmpty").style.display = "flex";
  $("badgeRight").textContent = "待解析";
  $("badgeRight").className = "badge";
  $("sideHint").textContent = "点击画面开始解析";
  const hint = $("clickHint");
  if (hint) { hint.textContent = "点击图片 · 开始解析"; hint.style.opacity = ""; }
}

function paintPills() {
  $("pillLat").textContent = "耗时 —";
  $("pillTok").textContent = "TOKENS —";
  $("pillModeText").textContent = "待解析";
  setLed("off");
}

function renderThumbs() {
  const host = $("thumbs");
  host.innerHTML = "";

  // upload tile first
  const up = document.createElement("button");
  up.type = "button";
  up.className = "film-upload" + (state.activeId === "custom" ? " active" : "");
  up.innerHTML =
    '<span class="plus">+</span>' +
    "<span>上传图像 / PDF</span>" +
    '<span class="sub">PNG · JPG · WEBP · PDF</span>';
  up.onclick = function () { $("imgUpload").click(); };
  host.appendChild(up);

  state.items.forEach(function (item) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "film" + (state.activeId === item.id ? " active" : "");
    b.dataset.id = item.id;
    b.innerHTML =
      '<img src="' + item.image + '" alt="' + escapeHtml(item.title) + '" loading="lazy" />' +
      '<span class="cap">' + escapeHtml(item.title) + '<small>' + escapeHtml(item.desc || '') + '</small></span>';
    b.onclick = function () { selectItem(item.id); };
    host.appendChild(b);
  });
}

$("imgUpload").onchange = function () {
  const f = this.files && this.files[0];
  if (!f || state.busy) return;
  if (f.size > 55 * 1024 * 1024) { showError("文件超过 55 MiB，请压缩或拆分后重试。"); this.value = ""; return; }
  const isPdf = f.type === "application/pdf" || /\.pdf$/i.test(f.name);
  if (!isPdf && (!f.type || f.type.indexOf("image/") !== 0)) {
    showError("请选择图像或 PDF 文件");
    return;
  }
  clearError();
  if (state.pdfUrl) URL.revokeObjectURL(state.pdfUrl);
  if (isPdf) {
    state.pdfUrl = URL.createObjectURL(f);
    state.custom = { name: f.name, file: f, isPdf: true, size: f.size };
    state.activeId = "custom";
    document.querySelectorAll(".film").forEach(function (el) { el.classList.remove("active"); });
    document.querySelectorAll(".film-upload").forEach(function (el) { el.classList.add("active"); });
    $("stageTitle").textContent = f.name;
    $("stageImg").hidden = true;
    $("stagePdf").hidden = false;
    $("stagePdf").src = state.pdfUrl;
    $("badgeLeft").textContent = "PDF";
    $("pillLat").textContent = "—"; $("pillTok").textContent = "—";
    $("pillModeText").textContent = "PDF · LOCAL"; setLed("on");
    state.task = "document";
    document.querySelectorAll("#taskSeg button").forEach(function (b) { b.classList.toggle("active", b.dataset.task === "document"); });
    updateTaskInputs(); resetResult();
    $("sideHint").textContent = "点击画面解析 PDF 全部页面";
    this.value = "";
    return;
  }
  const r = new FileReader();
  r.onload = function () {
    if (state.busy) return;
    state.custom = { name: f.name, dataUrl: r.result, size: f.size, isPdf: false };
    state.activeId = "custom";
    document.querySelectorAll(".film").forEach(function (el) { el.classList.remove("active"); });
    document.querySelectorAll(".film-upload").forEach(function (el) { el.classList.add("active"); });
    $("stageTitle").textContent = f.name;
    $("stageImg").hidden = false;
    $("stagePdf").hidden = true; $("stagePdf").removeAttribute("src");
    $("stageImg").src = r.result;
    $("stageImg").alt = f.name;
    $("pillLat").textContent = "—";
    $("pillTok").textContent = "—";
    $("pillModeText").textContent = "LIVE"; setLed("on");
    $("badgeLeft").textContent = "IMAGE";
    updateTaskInputs();
    resetResult();
    $("sideHint").textContent = "点击画面开始实时解析";
  };
  r.readAsDataURL(f);
  this.value = "";
};

function itemById(id) {
  return state.items.filter(function (x) { return x.id === id; })[0];
}

function selectItem(id) {
  const item = itemById(id);
  if (!item || state.busy) return;
  $("stageTitle").textContent = item.title;
  clearError();
  stopTyping();
  state.activeId = id;
  state.custom = null;
  if (state.pdfUrl) { URL.revokeObjectURL(state.pdfUrl); state.pdfUrl = null; }
  document.querySelectorAll(".film-upload").forEach(function (el) { el.classList.remove("active"); });
  document.querySelectorAll(".film").forEach(function (el) {
    el.classList.toggle("active", el.dataset.id === id);
  });
  $("stageImg").src = item.image;
  $("stageImg").hidden = false;
  $("stagePdf").hidden = true; $("stagePdf").removeAttribute("src");
  $("badgeLeft").textContent = "IMAGE";
  $("stageImg").alt = item.title;
  paintPills(item);
  updateTaskInputs();
  resetResult();

}

/* Live OCR for custom upload */
const LIVE_PROMPT =
  "Extract all information from the main body of the document image and represent it in markdown format, " +
  "ignoring headers and footers. Tables should be expressed in OTSL format, formulas in the document should " +
  "be represented using LATEX format, and the parsing should be organized according to the reading order.";

/**
 * Real streaming OCR via SSE proxy.
 * onDelta(textChunk) fires as tokens arrive.
 * Resolves { latency_ms, usage, model, content }.
 */
async function streamLiveOcr(dataUrl, onDelta, prompt, task) {
  const c = cfg();
  const body = {
    image_base64: dataUrl,
    prompt: prompt || LIVE_PROMPT,
    task: task || "document",
    fields: $("kieFields").value || "",
    question: $("vqaQuestion").value || "",
    temperature: 0,
    max_tokens: task === "vqa" ? 1024 : Math.max(1, Math.min(16384, c.max_tokens || 4096)),
  };
  const ac = new AbortController();
  const timer = setTimeout(function () { ac.abort(); }, 90000);
  let res;
  try {
    res = await fetch("/api/ocr/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal: ac.signal,
    });
  } catch (e) {
    clearTimeout(timer);
    if (e.name === "AbortError") throw new Error("请求超时（90s）");
    throw e;
  }
  if (!res.ok) {
    clearTimeout(timer);
    let msg = "HTTP " + res.status;
    try {
      const j = await res.json();
      if (j.error) msg = j.error;
    } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let content = "";
  let doneMeta = null;
  let err = null;

  try {
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() || "";
    for (const part of parts) {
      let ev = "message";
      let data = "";
      for (const line of part.split("\n")) {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      let obj;
      try { obj = JSON.parse(data); } catch (e) { continue; }
      if (ev === "delta" && obj.t) {
        content += obj.t;
        if (onDelta) onDelta(obj.t, content);
      } else if (ev === "done") {
        doneMeta = obj;
      } else if (ev === "error") {
        err = obj.error || "stream error";
      }
    }
  }
  } finally {
    clearTimeout(timer);
    try { reader.cancel(); } catch (e) { /* ignore */ }
  }
  if (err) throw new Error(err);
  if (!doneMeta) throw new Error("推理流中断，未收到完成事件");
  return {
    latency_ms: doneMeta ? doneMeta.latency_ms : 0,
    usage: (doneMeta && doneMeta.usage) || {},
    model: (doneMeta && doneMeta.model) || "SeerRay-Lab/Xiaomi-OCR-0",
    content: doneMeta && typeof doneMeta.content === "string" ? doneMeta.content : content,
  };
}

/**
 * Region SSE stream: yields events as regions complete.
 *   event layout {count, regions, layout_ms}
 *   event region {index, bucket, ok, ms, text}
 *   event done   {latency_ms, concurrency, usage}
 */
async function streamRegionOcr(dataUrl, handlers) {
  const c = cfg();
  const body = {
    image_base64: dataUrl,
    max_tokens: Math.max(1, Math.min(c.max_tokens || 2048, 4096)),
    concurrency: 16,
  };
  const ac = new AbortController();
  const timer = setTimeout(function () { ac.abort(); }, 120000);
  let res;
  try {
    res = await fetch("/api/parse/regions/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal: ac.signal,
    });
  } catch (e) {
    clearTimeout(timer);
    if (e.name === "AbortError") throw new Error("区域解析超时（120s）");
    throw e;
  }
  if (!res.ok) {
    clearTimeout(timer);
    let msg = "HTTP " + res.status;
    try { const j = await res.json(); if (j.error) msg = j.error; } catch (e) {}
    throw new Error(msg);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let done = null;
  let err = null;
  try {
    while (true) {
      const { done: d, value } = await reader.read();
      if (d) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop() || "";
      for (const part of parts) {
        let ev = "message";
        let data = "";
        for (const line of part.split("\n")) {
          if (line.startsWith("event:")) ev = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        let obj;
        try { obj = JSON.parse(data); } catch (e) { continue; }
        if (ev === "layout" && handlers.onLayout) handlers.onLayout(obj);
        else if (ev === "region" && handlers.onRegion) handlers.onRegion(obj);
        else if (ev === "done") done = obj;
        else if (ev === "error") err = obj.error || "stream error";
      }
    }
  } finally {
    clearTimeout(timer);
    try { reader.cancel(); } catch (e) {}
  }
  if (err) throw new Error(err);
  if (!done) throw new Error("区域推理流中断，未收到完成事件");
  return done;
}

/* region 拼装器：按阅读序保序上屏 */
function makeRegionAssembler(onFlush) {
  const buf = {};
  let next = 0;
  return {
    push: function (r) {
      buf[r.index] = r;
    },
    flush: function () {
      const out = [];
      while (buf[next] !== undefined) {
        out.push(buf[next]);
        delete buf[next];
        next++;
      }
      return out;
    },
    pending: function () { return Object.keys(buf).length; },
  };
}

/** 追加式打字机：单一全局串行队列 —— 多个调用不会互相打断文本顺序 */
const _typeQueue = [];
let _typeTimer = null;

function _typePump(el) {
  if (_typeTimer || !_typeQueue.length) return;
  const entry = _typeQueue.shift();
  const text = entry.text;
  const onFinish = entry.onFinish;
  let i = 0;
  const step = Math.max(2, Math.ceil(text.length / 60));
  function tick() {
    i = Math.min(text.length, i + step);
    el.textContent += text.slice(i - step, i);
    const pane = $("resPane");
    pane.scrollTop = pane.scrollHeight;
    if (i < text.length) {
      _typeTimer = setTimeout(tick, entry.ms || 12);
    } else {
      _typeTimer = null;
      if (onFinish) onFinish();
      // 队列里还有就继续
      if (_typeQueue.length) _typePump(el);
    }
  }
  tick();
}

function appendTyped(el, text, perChunkMs, onFinish) {
  _typeQueue.push({ text: String(text || ""), ms: perChunkMs, onFinish: onFinish });
  if (!_typeTimer) _typePump(el);
}

/** 等待打字机队列清空（渲染前调用） */
function waitTypedIdle() {
  return new Promise(function (resolve) {
    (function check() {
      if (!_typeTimer && !_typeQueue.length) return resolve();
      setTimeout(check, 40);
    })();
  });
}

/** Resolve a data URL for whatever is on stage (custom or showcase). */
async function stageImageDataUrl() {
  if (state.custom && state.custom.dataUrl) return state.custom.dataUrl;
  if (state.custom && state.custom.isPdf) throw new Error("PDF 不能用于图像任务");
  const item = itemById(state.activeId);
  if (!item || !item.image) throw new Error("无可用图像");
  const res = await fetch(item.image);
  if (!res.ok) throw new Error("图像加载失败");
  return await blobToDataUrl(await res.blob());
}

function fileToDataUrl(file) { return blobToDataUrl(file); }

async function runPdfParse(file) {
  if (file.size > 55 * 1024 * 1024) throw new Error("PDF 超过 55 MiB；请拆分文件后重试。");
  $("taskProgress").hidden = false;
  $("taskProgressFill").style.width = "0%";
  const c = cfg();
  const pdfData = await fileToDataUrl(file);
  const start = await fetch("/api/book", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pdf_base64: pdfData, pdf_name: file.name, mode: state.mode,
      prompt: $("docPrompt").value || LIVE_PROMPT, max_tokens: c.max_tokens || 4096,
      concurrency: state.mode === "region" ? 2 : 4, dpi: 150 }),
  });
  const started = await start.json();
  if (!start.ok || started.error) throw new Error(started.error || ("HTTP " + start.status));
  let status;
  while (true) {
    await new Promise(function (resolve) { setTimeout(resolve, 800); });
    const res = await fetch("/api/book/" + encodeURIComponent(started.id));
    status = await res.json();
    if (!res.ok) throw new Error(status.error || "读取 PDF 任务状态失败");
    const total = status.total || 0;
    const done = status.done || 0;
    const percent = total ? Math.round(done * 100 / total) : 0;
    $("taskProgressFill").style.width = percent + "%";
    $("stageHint").textContent = (status.phase_label || "PDF 处理中") + (total ? " · " + percent + "%" : "");
    $("sideHint").textContent = "PDF " + done + "/" + total + " 页 · " + (state.mode === "region" ? "区域模式" : "整页模式");
    $("pillModeText").textContent = "PDF · " + done + "/" + (total || "…");
    if (["done", "error", "cancelled"].includes(status.status)) break;
  }
  if (status.status !== "done" && !status.has_markdown) throw new Error(status.error || "PDF 解析未完成");
  $("taskProgressFill").style.width = "100%";
  const rawRes = await fetch("/api/book/" + encodeURIComponent(started.id) + "/raw");
  const raw = await rawRes.json();
  if (!rawRes.ok || raw.error) throw new Error(raw.error || "读取解析结果失败");
  return { content: raw.markdown || "", latency_ms: raw.elapsed_ms || status.elapsed_ms || 0,
    usage: status.usage || {}, model: status.model || "SeerRay-Lab/Xiaomi-OCR-0", pages: status.total || 0, warning: status.status !== "done" ? (status.error || "部分页面未完成") : "" };
}

function formatKieResult(rawText) {
  const match = String(rawText || "").match(/\{[\s\S]*\}/);
  if (!match) return { json: null, raw: rawText, warning: "Model output did not contain a JSON object" };
  try {
    let data = JSON.parse(match[0]);
    const fields = $("kieFields").value.split(/[\n,，]+/).map(function (x) { return x.trim(); }).filter(Boolean);
    if (fields.length && data && typeof data === "object" && !Array.isArray(data)) {
      data = Object.fromEntries(fields.map(function (field) { return [field, data[field] == null ? "" : data[field]]; }));
    }
    return { json: data };
  } catch (e) { return { json: null, raw: rawText, warning: "Model output contained invalid JSON" }; }
}

$("srcPane").onclick = async function () {
  if (state.busy) return;
  const item = itemById(state.activeId);

  const isPdf = !!(state.custom && state.custom.isPdf);
  const task = state.task;
  const useRegion = task === "document" && state.mode === "region";

  if (!item && !state.custom) { showError("请先选择样例或上传文件。"); return; }
  if (task === "vqa" && !$("vqaQuestion").value.trim()) { showError("请先输入你的问题。"); $("vqaQuestion").focus(); return; }
  clearError();
  resetResult();
  setBusy(true);
  firePulse();
  $("imgWrap").classList.add("thinking");
  const hint = $("clickHint");
  if (hint) { hint.textContent = isPdf ? "PDF 解析中…" : (useRegion ? "区域解析中…" : "思考中…"); hint.style.opacity = "1"; }
  $("sideHint").textContent = isPdf ? "PDF 页面渲染与解析中" : (useRegion ? "layout + 并发 region OCR" : "流式输出中");
  $("stageHint").textContent = isPdf ? "PDF…" : (useRegion ? "REGION…" : "STREAMING…");
  $("badgeRight").textContent = "LIVE";
  $("badgeRight").className = "badge on";

  stopTyping();
  const el = $("resText");
  el.classList.remove("markdown", "source-view");
  el.classList.add("typing");
  el.style.display = "block";
  el.textContent = "";
  $("resEmpty").style.display = "none";

  try {
    let data;
    if (isPdf) {
      data = await runPdfParse(state.custom.file);
      el.classList.remove("typing");
      el.classList.add("markdown");
      recordResult(data.content, data);
      if (data.warning) showError("部分 PDF 页面失败，已保留可下载的结果：" + data.warning);
      el.innerHTML = renderMarkdown(normalizeOcrOutput(data.content));
      $("pillLat").innerHTML = "LATENCY <b>" + ((data.latency_ms || 0) / 1000).toFixed(2) + "s</b>";
      $("pillTok").innerHTML = "TOKENS <b>" + ((data.usage && data.usage.completion_tokens) || "—") + "</b>";
      $("pillModeText").textContent = "PDF · " + data.pages + " PAGES"; setLed("on");
      $("sideHint").textContent = "点击画面可重新解析 PDF";
      $("stageHint").textContent = "PDF · DONE";
      if (hint) { hint.textContent = "点击解析"; hint.style.opacity = ""; }
    } else {
      const dataUrl = await stageImageDataUrl();
      if (useRegion) {
      // SSE：layout 先到，随后每个 region 完成即推 —— 串行打字机队列，保持阅读序
      const asm = makeRegionAssembler();
      let count = 0;
      let okN = 0;
      let layoutInfo = null;
      const pump = function () {
        const ready = asm.flush();
        if (!ready.length) return;
        ready.forEach(function (r) {
          const t = (r.text || "").trim();
          if (t) appendTyped(el, t + "\n\n", 8);
        });
      };
      data = await streamRegionOcr(dataUrl, {
        onLayout: function (o) {
          count = o.count || 0;
          layoutInfo = o;
          $("stageHint").textContent = "REGION · " + count + " boxes";
          $("sideHint").textContent = "layout " + ((o.layout_ms || 0) + "ms") + " · 开始识别…";
        },
        onRegion: function (o) {
          if (o.ok) okN++;
          asm.push(o);
          pump();
        },
      });
      // 收尾：flush 残余 + 等打字机队列清空再渲染（保证 OTSL 表/公式完整）
      pump();
      await waitTypedIdle();
      el.classList.remove("typing");
      el.classList.add("markdown");
      recordResult(data.content, data);
      el.innerHTML = renderMarkdown(data.content || "");
      $("pillLat").innerHTML = "LATENCY <b>" + ((data.latency_ms || 0) / 1000).toFixed(2) + "s</b>";
      $("pillTok").innerHTML = "TOKENS <b>" + ((data.usage && data.usage.completion_tokens) || "—") + "</b>";
      $("pillModeText").textContent = "REGION · " + okN + "/" + count;
      setLed("on");
      $("sideHint").textContent = "layout " + ((layoutInfo && layoutInfo.layout_ms) || 0) + "ms · 并发 " + (data.concurrency || 16);
      $("stageHint").textContent = "REGION · " + count + " · " + okN + " ok";
      if (hint) { hint.textContent = "点击解析"; hint.style.opacity = ""; }
      } else {
      const chosenPrompt = task === "document" ? ($("docPrompt").value || LIVE_PROMPT) : "";
      data = await streamLiveOcr(dataUrl, function (chunk, full) {
        el.textContent = full;
        const pane = $("resPane");
        pane.scrollTop = pane.scrollHeight;
      }, chosenPrompt, task);
      let finalContent = (data.content || "").trim();
      if (task === "kie") {
        const kie = formatKieResult(finalContent);
        finalContent = JSON.stringify(kie.json === null ? kie : kie.json, null, 2);
      }
      el.classList.remove("typing");
      el.classList.add("markdown");
      recordResult(finalContent, data);
      el.innerHTML = task === "kie"
        ? "<pre><code>" + escapeHtml(finalContent) + "</code></pre>"
        : renderMarkdown(normalizeOcrOutput(finalContent));
      const lat = (data.latency_ms / 1000).toFixed(2) + "s";
      const u = data.usage || {};
      $("pillLat").innerHTML = "LATENCY <b>" + lat + "</b>";
      $("pillTok").innerHTML = "TOKENS <b>" + (u.completion_tokens != null ? u.completion_tokens : "—") + "</b>";
      $("pillModeText").textContent = task === "document" ? "LIVE" : (task === "kie" ? "KIE · LIVE" : "VQA · LIVE"); setLed("on");
      $("sideHint").textContent = "点击画面可重新解析";
      $("stageHint").textContent = (task === "document" ? "PAGE" : task.toUpperCase()) + " · LIVE";
      if (hint) { hint.textContent = "点击解析"; hint.style.opacity = ""; }
      }
    }
  } catch (e) {
      el.classList.remove("typing");
      el.textContent = "";
      $("resEmpty").style.display = "flex";
      showError(e.message || String(e));
      $("badgeRight").textContent = "ERROR";
      $("badgeRight").className = "badge";
      setLed("err");
      $("sideHint").textContent = isPdf ? "确认 PDF 依赖与本机模型已就绪" : "确认本机模型和区域识别依赖已启动";
      $("stageHint").textContent = "ERROR";
      if (hint) { hint.textContent = "点击重试"; hint.style.opacity = ""; }
  } finally {
    setBusy(false);
    if (state.lastText) { $("badgeRight").textContent = state.lastMeta && state.lastMeta.warning ? "部分完成" : "已完成"; paintResult(); }
    $("imgWrap").classList.remove("thinking");
    if (isPdf) setTimeout(function () { $("taskProgress").hidden = true; }, 1100);
  }
};

/* load */
function loadShowcase() {
  return fetch("/showcase.json")
    .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
    .then(function (data) {
      state.items = (data && data.items) || [];
      renderThumbs();
      $("sampleCount").textContent = state.items.length + " 个样例";
      const first = state.items[0];
      if (first) selectItem(first.id);
    })
    .catch(function (e) {
      $("stageTitle").textContent = "Showcase 加载失败";
      showError(e.message);
    });
}

(function init() {
  if (!localStorage.getItem(LS_KEY)) localStorage.setItem(LS_KEY, JSON.stringify(DEFAULT_CFG));
  loadShowcase();
  checkHealth();
  setInterval(checkHealth, 30000);
})();

function blobToDataUrl(blob) {
  return new Promise(function (resolve, reject) {
    const r = new FileReader();
    r.onload = function () { resolve(r.result); };
    r.onerror = function () { reject(r.error || new Error("FileReader failed")); };
    r.readAsDataURL(blob);
  });
}

/** Convert OTSL markers into markdown tables when present. */
function normalizeOcrOutput(text) {
  const t = String(text || "");
  if (/<(?:fcel|ecel|lcel|ucel)\b/i.test(t) || /<nl\s*\/?>/i.test(t)) {
    return otslToMarkdownTable(t);
  }
  return t;
}


function setBusy(busy) {
  state.busy = busy;
  document.querySelectorAll("#thumbs button, #taskSeg button, #modeSeg button, #btnUpload, #btnRun, #btnSettings, #docPrompt, #kieFields, #vqaQuestion, #imgUpload").forEach(el => { el.disabled = busy; });
  $("btnRun").textContent = busy ? "解析中…" : "开始解析 ↗";
  $("srcPane").setAttribute("aria-busy", String(busy));
  if (!busy) document.querySelectorAll("#modeSeg button").forEach(el => { el.disabled = state.task !== "document"; });
}
$("btnRun").onclick = () => $("srcPane").onclick();
$("btnUpload").onclick = () => $("imgUpload").click();
$("btnCopy").onclick = async function () {
  try {
    const field = document.createElement("textarea");
    field.value = state.lastText; field.className = "sr-only";
    document.body.appendChild(field); field.select();
    let copied = false;
    try { copied = document.execCommand("copy"); } finally { field.remove(); }
    if (!copied) {
      await Promise.race([
        navigator.clipboard.writeText(state.lastText),
        new Promise((_, reject) => setTimeout(() => reject(new Error("Clipboard timeout")), 2500))
      ]);
    }
    this.textContent = "已复制";
    setTimeout(() => { this.textContent = "复制"; }, 1500);
  } catch (_) { showError("无法访问剪贴板，请切换原文视图复制，或下载结果。"); }
};
$("btnDownload").onclick = () => {
  if (!state.lastText) return;
  const form = document.createElement("form");
  form.method = "POST"; form.action = "/api/export"; form.hidden = true;
  for (const [name, value] of Object.entries({content: state.lastText, format: state.lastTask === "kie" ? "json" : "md"})) {
    const input = document.createElement("input"); input.type = "hidden";
    input.name = name; input.value = value; form.appendChild(input);
  }
  document.body.appendChild(form); form.submit(); form.remove();
};
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !$("drawer").hidden) closeDrawer();
  if (e.key === "Tab" && !$("drawer").hidden) {
    const nodes = [...$("drawer").querySelectorAll("button,input")];
    const first = nodes[0], last = nodes[nodes.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }
});
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("服务检查失败");
    const health = await res.json();
    $("connDot").classList.toggle("off", !health.ready);
    $("statusText").textContent = health.ready ? "模型已连接" : "模型未连接";
    $("statusText").title = health.ready ? health.model : health.error;
  } catch (_) {
    $("connDot").classList.add("off");
    $("statusText").textContent = "服务未连接";
  }
}
