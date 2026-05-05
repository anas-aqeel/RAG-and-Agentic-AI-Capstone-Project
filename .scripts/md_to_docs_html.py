"""Convert module preparation markdown files into standalone docs-page HTML.

Each output is a single self-contained HTML file with:
  - Left sidebar: H2 sections of the doc as nav (sticky)
  - Article body: full converted markdown content (all headings, code, tables)
  - Right rail: "On this page" with H3 anchors (sticky)
  - Prism.js (CDN) for syntax highlighting

No content is summarized — the entire markdown is preserved verbatim.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import markdown

PROJECT = Path(r"c:\Users\LENOVO\OneDrive\Desktop\Capstone Project")

MODULES = [
    {
        "label": "Module 1",
        "subtitle": "Structured GenAI",
        "title": "Module 1 — Interview Preparation & Deep Dive",
        "lede": "Using LLMs to transform unstructured data into structured JSON, validating outputs, building a CLI application, and testing LLM-dependent code.",
        "md": PROJECT / "module-1-build-structured-genai-app" / "MODULE_1_PREPARATION.md",
        "out": PROJECT / "module-1-build-structured-genai-app" / "MODULE_1_PREPARATION.html",
        "accent": "#c96442",        # warm amber
        "accent_soft": "#f9ebe4",
        "tag": "Structured Data · LLMs",
    },
    {
        "label": "Module 2",
        "subtitle": "Multimodal RAG",
        "title": "Module 2 — Interview Preparation & Deep Dive",
        "lede": "Building a multimodal RAG retrieval layer — embedding text and images into vector spaces, persisting them in ChromaDB, performing similarity search with metadata filtering, and fusing multi-modal results into a single ranked list.",
        "md": PROJECT / "module-2-design-multimodal-rag-system" / "MODULE_2_PREPARATION.md",
        "out": PROJECT / "module-2-design-multimodal-rag-system" / "MODULE_2_PREPARATION.html",
        "accent": "#0d7d7e",        # teal
        "accent_soft": "#dff1f1",
        "tag": "Vector Search · CLIP · ChromaDB",
    },
    {
        "label": "Module 3",
        "subtitle": "Multi-Agent Systems",
        "title": "Module 3 — Interview Preparation & Deep Dive",
        "lede": "Designing specialized AI agents, orchestrating them into multi-agent workflows (sequential + parallel hybrid), managing shared state, and exposing the system through a Gradio chatbot with intent classification and preference extraction.",
        "md": PROJECT / "module-3-combine-agents-multi-agent-system" / "MODULE_3_PREPARATION.md",
        "out": PROJECT / "module-3-combine-agents-multi-agent-system" / "MODULE_3_PREPARATION.html",
        "accent": "#6f42c1",        # purple
        "accent_soft": "#efe6f9",
        "tag": "Agents · LangGraph · Gradio",
    },
    {
        "label": "Module 5",
        "subtitle": "Model Context Protocol",
        "title": "Module 5 — Interview Preparation & Deep Dive (MCP)",
        "lede": "Building a complete Model Context Protocol (MCP) system — a FastMCP server exposing data and tools, a Python MCP client connecting over stdio with roots and sampling callbacks, and a full host application using a ReAct agent loop with a Gradio chat UI.",
        "md": PROJECT / "module-5-final-project" / "MODULE_5_PREPARATION.md",
        "out": PROJECT / "module-5-final-project" / "MODULE_5_PREPARATION.html",
        "accent": "#0f766e",        # emerald-deep
        "accent_soft": "#daf0ed",
        "tag": "MCP · FastMCP · ReAct",
    },
]


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    return text


def preprocess_markdown(md_body: str) -> str:
    """Insert blank lines before numbered/bulleted lists that immediately
    follow paragraph text, so python-markdown recognizes them as lists.
    Also leave fenced code blocks untouched.
    """
    out: list[str] = []
    in_code = False
    fence_re = re.compile(r"^```")
    list_start_re = re.compile(r"^(\d+\.|[-*+])\s+")
    for i, line in enumerate(md_body.splitlines()):
        if fence_re.match(line):
            in_code = not in_code
            out.append(line)
            continue
        if not in_code and list_start_re.match(line):
            # Need blank line before this list item if previous out-line is
            # non-blank text that's not already a list item or heading marker
            if out:
                prev = out[-1]
                prev_is_blank = prev.strip() == ""
                prev_is_list = bool(list_start_re.match(prev))
                prev_is_indent_continuation = prev.startswith("    ") or prev.startswith("\t")
                if not (prev_is_blank or prev_is_list or prev_is_indent_continuation):
                    out.append("")
        out.append(line)
    return "\n".join(out)


def md_to_html(md_text: str) -> tuple[str, list[dict], list[dict]]:
    """Convert markdown to HTML body and extract H2/H3 outline.

    Returns:
        (html_body, h2_sections, h3_subsections)
        h2_sections = [{id, text}], h3_subsections = [{id, text, parent_id}]
    """
    # Drop the leading H1 since we render it manually with hero styling
    lines = md_text.splitlines()
    while lines and not lines[0].startswith("# "):
        lines.pop(0)
    if lines and lines[0].startswith("# "):
        lines.pop(0)
    # Drop the immediate blockquote (subtitle) — we render as lede
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].startswith("> "):
        lines.pop(0)
    md_body = "\n".join(lines)
    md_body = preprocess_markdown(md_body)

    md = markdown.Markdown(
        extensions=[
            "fenced_code",
            "tables",
            "toc",
            "attr_list",
            "sane_lists",
            "md_in_html",
        ],
        extension_configs={
            "toc": {
                "anchorlink": False,
                "permalink": False,
                "toc_depth": "2-3",
                "slugify": lambda value, sep: slugify(value),
            }
        },
    )

    html_body = md.convert(md_body)

    # Build outline from the markdown TOC tokens
    h2_sections: list[dict] = []
    h3_subsections: list[dict] = []
    current_h2_id: str | None = None
    for token in md.toc_tokens:  # H2 entries (toc_depth starts at 2)
        h2_id = token["id"]
        h2_text = token["name"]
        h2_sections.append({"id": h2_id, "text": h2_text})
        for sub in token.get("children", []):
            h3_subsections.append({
                "id": sub["id"],
                "text": sub["name"],
                "parent_id": h2_id,
            })

    return html_body, h2_sections, h3_subsections


PRISM_LANG_MAP = {
    "python": "python",
    "py": "python",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "bash": "bash",
    "sh": "bash",
    "shell": "bash",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "sql": "sql",
    "html": "markup",
    "xml": "markup",
    "markdown": "markdown",
    "md": "markdown",
    "dockerfile": "docker",
    "ruby": "ruby",
    "rb": "ruby",
}


def post_process_html(body: str) -> str:
    """Add Prism language classes to code blocks and tweak structure."""

    # python-markdown emits <pre><code class="lang"> for fenced code blocks
    def fix_code(match: re.Match) -> str:
        attrs = match.group(1) or ""
        cls_match = re.search(r'class="([^"]*)"', attrs)
        lang = ""
        if cls_match:
            classes = cls_match.group(1).split()
            for c in classes:
                # python-markdown emits "language-xxx" with fenced_code
                if c.startswith("language-"):
                    lang = c[len("language-"):]
                    break
                # Some versions use bare lang or "highlight"
                if c.lower() in PRISM_LANG_MAP:
                    lang = c.lower()
                    break
        prism_lang = PRISM_LANG_MAP.get(lang.lower(), "")
        cls = f"language-{prism_lang}" if prism_lang else "language-none"
        return f'<pre class="code-block"><code class="{cls}">'

    body = re.sub(r"<pre><code([^>]*)>", fix_code, body)
    # Tables -> wrap for horizontal scroll
    body = re.sub(
        r"(<table>)",
        r'<div class="table-wrap">\1',
        body,
    )
    body = body.replace("</table>", "</table></div>")
    return body


def render_sidebar(sections: list[dict], accent: str) -> str:
    """Group sections into nav with subtle group label.

    The `text` from python-markdown's toc_tokens is already HTML-escaped, so
    do not escape again.
    """
    items = []
    for sec in sections:
        items.append(
            f'<a href="#{sec["id"]}" data-target="{sec["id"]}">{sec["text"]}</a>'
        )
    nav_links = "\n".join(items)
    return f"""
    <nav class="sidebar" data-od-id="sidebar" aria-label="Module sections">
      <div class="sidebar-inner">
        <div class="group">
          <div class="group-label">Sections</div>
          {nav_links}
        </div>
      </div>
    </nav>
    """


def render_toc(h2: list[dict], h3: list[dict]) -> str:
    """Right rail — full outline H2 (level 1) + H3 (level 2 indented).

    Names are already HTML-escaped by python-markdown.
    """
    rows: list[str] = []
    for sec in h2:
        rows.append(
            f'<a class="toc-h2" href="#{sec["id"]}" data-target="{sec["id"]}">{sec["text"]}</a>'
        )
        children = [c for c in h3 if c["parent_id"] == sec["id"]]
        for c in children:
            rows.append(
                f'<a class="toc-h3" href="#{c["id"]}" data-target="{c["id"]}">{c["text"]}</a>'
            )
    body = "\n".join(rows)
    return f"""
    <aside class="toc" data-od-id="toc" aria-label="On this page">
      <div class="toc-inner">
        <div class="toc-label">On this page</div>
        {body}
      </div>
    </aside>
    """


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet" />
  <style>
    :root {{
      --bg: #fafaf8;
      --fg: #1a1a1a;
      --fg-2: #2c2c2c;
      --muted: #6b7280;
      --muted-2: #9ca3af;
      --border: #e5e7eb;
      --border-soft: #eceef1;
      --surface: #ffffff;
      --surface-2: #f5f5f3;
      --accent: {accent};
      --accent-soft: {accent_soft};
      --code-bg: #1c1f24;
      --code-fg: #e6e6e6;
      --kbd-bg: #f3f4f6;
      --max-prose: 760px;
      --font-body: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; scroll-padding-top: 80px; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--fg);
      font-family: var(--font-body);
      font-size: 15.5px;
      line-height: 1.7;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }}

    /* Top bar */
    .topbar {{
      position: sticky; top: 0; z-index: 50;
      background: rgba(255,255,255,0.9);
      backdrop-filter: saturate(180%) blur(12px);
      -webkit-backdrop-filter: saturate(180%) blur(12px);
      border-bottom: 1px solid var(--border);
      padding: 13px 32px;
      display: flex; justify-content: space-between; align-items: center;
      gap: 16px;
    }}
    .topbar .brand {{
      display: flex; align-items: center; gap: 10px;
      font-weight: 700; font-size: 15px; color: var(--fg);
      text-decoration: none;
    }}
    .topbar .brand .mark {{
      display: inline-flex; align-items: center; justify-content: center;
      width: 26px; height: 26px; border-radius: 7px;
      background: var(--accent); color: #fff;
      font-size: 12px; font-weight: 700;
      letter-spacing: 0;
    }}
    .topbar .meta {{ display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 13px; }}
    .topbar .pill {{
      display: inline-flex; align-items: center; gap: 6px;
      padding: 4px 10px; border-radius: 999px;
      background: var(--accent-soft); color: var(--accent);
      font-size: 12px; font-weight: 600; letter-spacing: 0.01em;
    }}

    /* Layout */
    .layout {{
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr) 240px;
      min-height: calc(100vh - 53px);
    }}
    @media (max-width: 1180px) {{
      .layout {{ grid-template-columns: 240px minmax(0, 1fr); }}
      .toc {{ display: none; }}
    }}
    @media (max-width: 820px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ display: none; }}
      article {{ padding: 32px 22px 80px !important; }}
    }}

    /* Sidebar */
    .sidebar {{
      border-right: 1px solid var(--border);
      background: var(--surface);
      position: sticky;
      top: 53px;
      align-self: start;
      height: calc(100vh - 53px);
      overflow-y: auto;
    }}
    .sidebar-inner {{ padding: 28px 18px 40px; font-size: 13.5px; }}
    .sidebar .group-label {{
      font-size: 11px; color: var(--muted-2);
      text-transform: uppercase; letter-spacing: 0.08em;
      padding: 0 10px 12px; font-weight: 600;
    }}
    .sidebar a {{
      display: block;
      color: var(--fg-2);
      text-decoration: none;
      padding: 7px 12px;
      border-radius: 6px;
      margin-bottom: 1px;
      font-weight: 500;
      line-height: 1.4;
      border-left: 2px solid transparent;
      transition: background 80ms, color 80ms, border-color 80ms;
    }}
    .sidebar a:hover {{
      background: var(--surface-2);
      color: var(--fg);
    }}
    .sidebar a.active {{
      background: var(--accent-soft);
      color: var(--accent);
      border-left-color: var(--accent);
      font-weight: 600;
    }}

    /* Article body */
    article {{
      padding: 56px 64px 96px;
      max-width: calc(var(--max-prose) + 128px);
      width: 100%;
      margin: 0 auto;
    }}
    .crumbs {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 18px;
      letter-spacing: 0.01em;
    }}
    .crumbs .sep {{ margin: 0 6px; opacity: 0.5; }}

    .hero h1 {{
      font-size: 42px;
      letter-spacing: -0.025em;
      line-height: 1.1;
      margin: 0 0 18px;
      font-weight: 800;
    }}
    .hero .lede {{
      color: var(--muted);
      font-size: 18px;
      line-height: 1.55;
      margin: 0 0 8px;
      max-width: 680px;
    }}
    .hero .module-tag {{
      display: inline-flex; align-items: center; gap: 8px;
      padding: 5px 12px; border-radius: 999px;
      background: var(--accent-soft); color: var(--accent);
      font-size: 12px; font-weight: 600; letter-spacing: 0.02em;
      margin-bottom: 18px;
    }}
    .hero {{ padding-bottom: 28px; margin-bottom: 28px; border-bottom: 1px solid var(--border); }}

    /* Headings */
    article h1 {{ font-size: 36px; }}
    article h2 {{
      font-size: 26px;
      letter-spacing: -0.015em;
      margin: 56px 0 14px;
      padding-top: 8px;
      font-weight: 700;
      line-height: 1.25;
    }}
    article h3 {{
      font-size: 19px;
      margin: 36px 0 10px;
      font-weight: 700;
      letter-spacing: -0.005em;
      line-height: 1.3;
    }}
    article h4 {{
      font-size: 15.5px;
      margin: 28px 0 8px;
      font-weight: 700;
      letter-spacing: 0.01em;
      text-transform: none;
      color: var(--fg);
    }}
    article h5, article h6 {{ font-size: 14.5px; margin: 22px 0 6px; font-weight: 600; }}
    article h2 + p, article h3 + p {{ margin-top: 0; }}
    article p {{ margin: 14px 0; }}
    article p, article li {{ overflow-wrap: anywhere; }}

    /* Body links */
    article a {{
      color: var(--accent);
      text-decoration: none;
      border-bottom: 1px solid color-mix(in oklab, var(--accent) 30%, transparent);
      transition: border-color 80ms;
    }}
    article a:hover {{ border-bottom-color: var(--accent); }}

    /* Inline code */
    article code {{
      font-family: var(--font-mono);
      background: var(--surface-2);
      color: #b91c1c;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.88em;
      font-weight: 500;
      border: 1px solid var(--border-soft);
    }}

    /* Block code */
    article pre, article pre.code-block {{
      background: var(--code-bg);
      color: var(--code-fg);
      border-radius: 10px;
      padding: 16px 18px;
      overflow-x: auto;
      font-size: 13.5px;
      line-height: 1.6;
      margin: 18px 0 22px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
      border: 1px solid #2a2e35;
    }}
    article pre code, article pre.code-block code {{
      background: transparent;
      color: inherit;
      padding: 0;
      border: 0;
      font-size: inherit;
      font-weight: 400;
      font-family: var(--font-mono);
    }}

    /* Lists */
    article ul, article ol {{ margin: 12px 0; padding-left: 24px; }}
    article li {{ margin: 6px 0; }}
    article li > p {{ margin: 4px 0; }}
    article li::marker {{ color: var(--muted); }}

    /* Blockquote */
    article blockquote {{
      margin: 18px 0;
      padding: 14px 18px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-left: 3px solid var(--accent);
      border-radius: 8px;
      color: var(--fg-2);
    }}
    article blockquote p {{ margin: 6px 0; }}

    /* Tables */
    article .table-wrap {{
      overflow-x: auto;
      margin: 22px 0;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--surface);
    }}
    article table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    article th, article td {{
      text-align: left;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border-soft);
      vertical-align: top;
    }}
    article tr:last-child td {{ border-bottom: 0; }}
    article th {{
      background: var(--surface-2);
      font-weight: 700;
      color: var(--fg);
      font-size: 13px;
      letter-spacing: 0.01em;
      border-bottom: 1px solid var(--border);
    }}
    article tbody tr:nth-child(even) {{ background: var(--bg); }}

    /* HR */
    article hr {{ border: 0; border-top: 1px solid var(--border); margin: 48px 0; }}

    /* Strong / emphasis */
    article strong {{ color: var(--fg); font-weight: 700; }}
    article em {{ color: var(--fg-2); }}

    /* TOC right rail */
    .toc {{
      border-left: 1px solid var(--border);
      position: sticky;
      top: 53px;
      align-self: start;
      height: calc(100vh - 53px);
      overflow-y: auto;
    }}
    .toc-inner {{ padding: 56px 22px 40px; font-size: 13px; }}
    .toc-label {{
      font-size: 11px; color: var(--muted-2);
      text-transform: uppercase; letter-spacing: 0.08em;
      margin-bottom: 14px; font-weight: 600;
    }}
    .toc a {{
      display: block;
      color: var(--muted);
      text-decoration: none;
      padding: 5px 0 5px 12px;
      border-left: 2px solid transparent;
      line-height: 1.4;
      transition: color 80ms, border-color 80ms;
    }}
    .toc a:hover {{ color: var(--fg-2); }}
    .toc a.toc-h2 {{ font-weight: 600; color: var(--fg-2); margin-top: 4px; }}
    .toc a.toc-h3 {{ padding-left: 24px; font-size: 12.5px; }}
    .toc a.active {{ color: var(--accent); border-left-color: var(--accent); }}
    .toc a.active.toc-h2 {{ color: var(--accent); }}

    /* Footer */
    .footer {{
      padding: 32px 0 0;
      margin-top: 48px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 13px;
      display: flex; justify-content: space-between; gap: 12px;
      flex-wrap: wrap;
    }}

    /* Code block scrollbar */
    pre::-webkit-scrollbar {{ height: 8px; width: 8px; }}
    pre::-webkit-scrollbar-thumb {{ background: #3a3f48; border-radius: 4px; }}
    pre::-webkit-scrollbar-track {{ background: transparent; }}

    /* Selection */
    ::selection {{ background: color-mix(in oklab, var(--accent) 26%, transparent); }}

    /* Focus styles */
    a:focus-visible, button:focus-visible {{
      outline: 2px solid var(--accent);
      outline-offset: 3px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <header class="topbar" data-od-id="topbar">
    <a class="brand" href="#top">
      <span class="mark">{module_initial}</span>
      <span>Capstone Prep · {module_short}</span>
    </a>
    <span class="meta">
      <span class="pill">{tag}</span>
    </span>
  </header>

  <div class="layout">
    {sidebar}

    <article data-od-id="article" id="top">
      <div class="crumbs">
        Interview Prep <span class="sep">›</span> {module_short} <span class="sep">›</span> Deep Dive
      </div>

      <header class="hero">
        <span class="module-tag">{module_label} · {module_subtitle}</span>
        <h1>{hero_title}</h1>
        <p class="lede">{lede}</p>
      </header>

      {body}

      <div class="footer">
        <span>RAG &amp; Agentic AI Capstone — {module_label} preparation</span>
        <span>Read alongside the code in <code>{module_label_lower}</code></span>
      </div>
    </article>

    {toc}
  </div>

  <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
  <script>
    // Active section highlighting via IntersectionObserver
    (function() {{
      const headings = Array.from(document.querySelectorAll('article h2, article h3'));
      const links = Array.from(document.querySelectorAll('.sidebar a, .toc a'));
      if (!headings.length) return;

      const linkById = new Map();
      links.forEach(a => {{
        const id = a.getAttribute('data-target');
        if (id) {{
          if (!linkById.has(id)) linkById.set(id, []);
          linkById.get(id).push(a);
        }}
      }});

      const setActive = (id) => {{
        links.forEach(a => a.classList.remove('active'));
        if (linkById.has(id)) {{
          linkById.get(id).forEach(a => a.classList.add('active'));
        }}
      }};

      const visible = new Map();
      const observer = new IntersectionObserver((entries) => {{
        entries.forEach(e => {{
          if (e.isIntersecting) visible.set(e.target.id, e.intersectionRatio);
          else visible.delete(e.target.id);
        }});
        // Pick the heading closest to the top that is visible
        let bestId = null, bestTop = Infinity;
        visible.forEach((_, id) => {{
          const el = document.getElementById(id);
          if (!el) return;
          const top = el.getBoundingClientRect().top;
          if (top >= 0 && top < bestTop) {{ bestTop = top; bestId = id; }}
        }});
        if (!bestId) {{
          // Fall back to last heading above viewport
          let lastId = null;
          headings.forEach(h => {{
            if (h.getBoundingClientRect().top < 80) lastId = h.id;
          }});
          bestId = lastId;
        }}
        if (bestId) setActive(bestId);
      }}, {{
        rootMargin: '-70px 0px -70% 0px',
        threshold: [0, 0.5, 1]
      }});

      headings.forEach(h => observer.observe(h));
    }})();
  </script>
</body>
</html>
"""


def render_module(cfg: dict) -> None:
    md_text = cfg["md"].read_text(encoding="utf-8")
    body, h2, h3 = md_to_html(md_text)
    body = post_process_html(body)

    sidebar = render_sidebar(h2, cfg["accent"])
    toc = render_toc(h2, h3)

    page = PAGE_TEMPLATE.format(
        title=cfg["title"],
        accent=cfg["accent"],
        accent_soft=cfg["accent_soft"],
        module_label=cfg["label"],
        module_label_lower=cfg["label"].lower().replace(" ", "-"),
        module_short=cfg["label"],
        module_subtitle=cfg["subtitle"],
        module_initial=cfg["label"].split()[-1][0].upper(),
        hero_title=html.escape(cfg["title"]),
        lede=html.escape(cfg["lede"]),
        tag=html.escape(cfg["tag"]),
        sidebar=sidebar,
        toc=toc,
        body=body,
    )

    cfg["out"].write_text(page, encoding="utf-8")
    print(f"  wrote: {cfg['out'].relative_to(PROJECT)}")


def main() -> None:
    print("Converting markdown prep files to HTML...")
    for cfg in MODULES:
        if not cfg["md"].exists():
            print(f"  SKIP: {cfg['md']} not found")
            continue
        render_module(cfg)
    print("Done.")


if __name__ == "__main__":
    main()
