import asyncio
import base64
from html import unescape
from html.parser import HTMLParser
import os
import re
import sys
import textwrap
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
from playwright.async_api import async_playwright, Browser, Page

SYSTEM_PROMPT = """You are a browser automation agent that controls a web browser to complete tasks.

Rules:
- For searches, use navigate() with direct URLs — never click search boxes:
    Google search:    https://www.google.com/search?q=your+query
    Google Shopping:  https://www.google.com/search?q=your+query&tbm=shop
- If a tool fails, try a DIFFERENT approach immediately — never repeat the same failing action.
- Prefer extract_content() to read page text rather than clicking through menus.
- Call finish() as soon as you have enough information — do not keep browsing unnecessarily.
"""

TOOLS = [
    {
        "name": "navigate",
        "description": "Navigate the browser to a URL",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to navigate to"}
            },
            "required": ["url"],
        },
    },
    {
        "name": "click_element",
        "description": "Click on an element using a CSS selector or visible text",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector or visible text to click"}
            },
            "required": ["selector"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text into an input field",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector for the input field"},
                "text": {"type": "string", "description": "Text to type"},
            },
            "required": ["selector", "text"],
        },
    },
    {
        "name": "press_key",
        "description": "Press a keyboard key (e.g. Enter, Tab, Escape)",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key name to press"}
            },
            "required": ["key"],
        },
    },
    {
        "name": "extract_content",
        "description": "Extract text content from the page or a specific element",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector (optional, defaults to full page body)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "scroll_page",
        "description": "Scroll the page up or down",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Scroll direction",
                },
                "amount": {
                    "type": "integer",
                    "description": "Pixels to scroll (default 500)",
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "fill_form",
        "description": "Fill a form field with a value",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector for the form field"},
                "value": {"type": "string", "description": "Value to fill in"},
            },
            "required": ["selector", "value"],
        },
    },
    {
        "name": "search_web",
        "description": "Search Google for information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_page_info",
        "description": "Get the current page title and URL",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "wait_and_extract",
        "description": "Wait for an element to appear and extract its text content",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector to wait for"},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default 5000)",
                },
            },
            "required": ["selector"],
        },
    },
    {
        "name": "finish",
        "description": "Call this when you have collected enough information to answer the task. Ends the session immediately with your final answer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Complete final answer to the task"}
            },
            "required": ["answer"],
        },
    },
]

# Tools that don't need a screenshot after execution
_NO_SCREENSHOT = {"extract_content", "get_page_info", "wait_and_extract"}


def _make_gif(screenshots: list[str], width: int = 720) -> str:
    """Stitch base64 JPEG/PNG screenshots into a base64 GIF."""
    try:
        import io as _io
        from PIL import Image

        resample = getattr(Image, "Resampling", Image).LANCZOS
        frames = []
        for b64 in screenshots:
            raw = base64.b64decode(b64)
            img = Image.open(_io.BytesIO(raw)).convert("RGB")
            h = int(img.height * width / img.width)
            frames.append(img.resize((width, h), resample).quantize(colors=256))

        if not frames:
            return ""

        buf = _io.BytesIO()
        frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=600,
            loop=0,
            optimize=False,
        )
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        print(f"GIF generation failed: {e}")
        return ""


def _trim_screenshots(messages: list, keep_last: int = 3) -> list:
    """Return a copy of messages with old screenshot images replaced by a text placeholder.

    Keeps only the `keep_last` most recent screenshots, cutting input token cost
    dramatically on long runs (each screenshot costs ~1 000–2 000 input tokens).
    """
    import copy
    msgs = copy.deepcopy(messages)
    seen = 0
    for msg in reversed(msgs):
        if not isinstance(msg.get("content"), list):
            continue
        for block in reversed(msg["content"]):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                seen += 1
                if seen > keep_last:
                    block.clear()
                    block.update({"type": "text", "text": "[screenshot]"})
            elif block.get("type") == "tool_result":
                inner = block.get("content", [])
                if not isinstance(inner, list):
                    continue
                for ib in reversed(inner):
                    if isinstance(ib, dict) and ib.get("type") == "image":
                        seen += 1
                        if seen > keep_last:
                            ib.clear()
                            ib.update({"type": "text", "text": "[screenshot]"})
    return msgs


_CHROMIUM_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-software-rasterizer",
    "--disable-extensions",
    "--disable-background-networking",
    "--disable-default-apps",
    "--disable-sync",
    "--no-first-run",
    "--mute-audio",
    "--hide-scrollbars",
    "--disable-site-isolation-trials",
    "--disable-features=IsolateOrigins,site-per-process",
    "--renderer-process-limit=1",
    "--js-flags=--max-old-space-size=256",
]


class _DOMExtractor(HTMLParser):
    """Lightweight HTML parser for serverless fallback mode."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg", "path", "meta", "link", "iframe"}
    _BLOCK_TAGS = {"h1", "h2", "h3", "h4", "p", "li", "td", "th", "pre", "blockquote", "article", "section", "div"}

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title = ""
        self.blocks: list[tuple[str, str, str]] = []  # (kind, text, href)
        self.links: list[tuple[str, str, str]] = []   # (text, abs_url, attrs_str)
        self.tag_texts: dict[str, list[str]] = {}
        self._skip_depth = 0
        self._in_title = False
        self._cur_tag = ""
        self._cur_attrs: dict[str, str] = {}
        self._cur_href = ""
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        attr_dict = {k.lower(): (v or "") for k, v in attrs}
        if tag == "title":
            self._in_title = True
            self._buf = []
        elif tag in ("h1", "h2", "h3", "h4", "p", "li", "pre"):
            self._flush()
            self._cur_tag = tag
            self._cur_attrs = attr_dict
        elif tag == "a":
            href = attr_dict.get("href", "").strip()
            if href and not href.startswith(("javascript:", "#", "mailto:")):
                self._cur_href = urljoin(self.base_url, href)
                self._cur_attrs = attr_dict
        elif tag in ("br", "hr", "tr", "div", "section", "article", "header", "footer", "nav", "main"):
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth > 0:
            return
        if tag == "title":
            self._in_title = False
            self.title = " ".join("".join(self._buf).split())
            self._buf = []
        elif tag == "a":
            text = " ".join("".join(self._buf).split())
            if text and self._cur_href:
                attrs_s = f"{self._cur_attrs.get('id', '')} {self._cur_attrs.get('class', '')}".strip()
                self.links.append((text, self._cur_href, attrs_s))
                self.blocks.append(("link", text, self._cur_href))
                self.tag_texts.setdefault("a", []).append(text)
                self._buf = []
            self._cur_href = ""
        elif tag in ("h1", "h2", "h3", "h4", "p", "li", "pre"):
            self._flush()
            self._cur_tag = ""

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title or self._cur_href or self._cur_tag:
            self._buf.append(data)
        else:
            cleaned = " ".join(data.split())
            if len(cleaned) > 2:
                self.blocks.append(("text", cleaned, ""))

    def _flush(self) -> None:
        if not self._buf:
            return
        text = " ".join("".join(self._buf).split())
        self._buf = []
        if not text:
            return
        tag = self._cur_tag or "text"
        kind = tag if tag in ("h1", "h2", "h3") else "text"
        self.blocks.append((kind, text, self._cur_href))
        self.tag_texts.setdefault(tag, []).append(text)
        for cls in self._cur_attrs.get("class", "").split():
            self.tag_texts.setdefault(f".{cls}", []).append(text)
        if self._cur_attrs.get("id"):
            self.tag_texts.setdefault(f"#{self._cur_attrs['id']}", []).append(text)


class _DummyLocator:
    def __init__(self, page: "_HttpBrowserPage", selector: str):
        self._page = page
        self._selector = selector

    @property
    def first(self) -> "_DummyLocator":
        return self

    async def wait_for(self, state: str = "attached", timeout: int = 3000) -> None:
        return None

    async def all_inner_texts(self) -> list[str]:
        return self._page._query_selector_texts(self._selector)

    async def inner_text(self, timeout: int = 3000) -> str:
        texts = self._page._query_selector_texts(self._selector)
        return "\n".join(texts) if texts else await self._page.inner_text("body")

    async def click(self, timeout: int = 3000) -> None:
        await self._page.click(self._selector, timeout=timeout)

    async def fill(self, value: str, timeout: int = 3000) -> None:
        await self._page.fill(self._selector, value, timeout=timeout)


class _DummyKeyboard:
    def __init__(self, page: "_HttpBrowserPage"):
        self._page = page

    async def press(self, key: str) -> None:
        if key.lower() == "enter" and self._page._last_input:
            q = self._page._last_input
            await self._page.goto("https://www.google.com/search?q=" + quote_plus(q))


class _HttpBrowserPage:
    """Serverless-compatible browser page engine (httpx + HTML parser + Pillow renderer).

    Used automatically when running on platforms like Vercel where Playwright's native
    Node/Chromium binaries cannot execute due to missing OS shared libraries.
    """

    def __init__(self) -> None:
        import httpx
        self._client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=15.0,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        self.url: str = "about:blank"
        self._title: str = "New Tab"
        self._blocks: list[tuple[str, str, str]] = []
        self._links: list[tuple[str, str, str]] = []
        self._tag_texts: dict[str, list[str]] = {}
        self._body_text: str = ""
        self._scroll_y: int = 0
        self._last_input: str = ""
        self.keyboard = _DummyKeyboard(self)

    def set_default_timeout(self, timeout: int) -> None:
        pass

    async def wait_for_timeout(self, ms: int) -> None:
        await asyncio.sleep(min(ms, 200) / 1000.0)

    async def title(self) -> str:
        return self._title

    async def close(self) -> None:
        await self._client.aclose()

    def _extract_search_query(self, url: str) -> str | None:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        qs = parse_qs(parsed.query)
        if "google." in host and (
            parsed.path.startswith("/search") or "/flights" in parsed.path or "/travel" in parsed.path
        ):
            q = (qs.get("q") or [""])[0]
            if not q and "/flights" in parsed.path:
                q = "flights"
            if qs.get("tbm") == ["shop"] and q and "price" not in q.lower():
                q = f"{q} price buy online"
            return q or None
        if "duckduckgo.com" in host and qs.get("q"):
            return qs["q"][0]
        if "bing.com" in host and parsed.path.startswith("/search") and qs.get("q"):
            return qs["q"][0]
        return None

    async def _search_fallback(self, query: str, display_url: str) -> None:
        self.url = display_url
        self._title = f"{query} - Search Results"
        self._scroll_y = 0
        self._blocks = [("h1", f"Search results for: {query}", "")]
        self._links = []
        self._tag_texts = {"h1": [f"Search results for: {query}"]}
        results: list[tuple[str, str, str]] = []

        # 1. Try Yahoo Search HTML (reliable from cloud/datacenter IPs)
        try:
            resp = await self._client.get(
                "https://search.yahoo.com/search",
                params={"p": query, "n": "10"},
            )
            if resp.status_code == 200:
                html = resp.text
                # Extract result blocks from Yahoo Search
                for m in re.finditer(
                    r'<h3[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?</h3>(.*?)(?=<h3|<div class="dd pagination"|$)',
                    html,
                    re.S | re.I,
                ):
                    raw_href, raw_title, rest = unescape(m.group(1)), unescape(m.group(2)), unescape(m.group(3))
                    title_clean = re.sub(r"<[^>]+>", " ", raw_title)
                    title_clean = " ".join(title_clean.split())
                    if not title_clean or "yahoo.com/preferences" in raw_href or "bing.com/aclick" in raw_href:
                        continue
                    # Decode Yahoo RU= redirect if present
                    ru_match = re.search(r"/RU=([^/]+)/", raw_href)
                    real_url = unquote(ru_match.group(1)) if ru_match else raw_href
                    if any(
                        ad in real_url.lower()
                        for ad in ("yahoo.com", "bing.com/aclick", "doubleclick.net", "googleadservices")
                    ):
                        continue
                    snip_match = re.search(
                        r'<div[^>]+class="[^"]*(?:compText|fc-falcon)[^"]*"[^>]*>(.*?)</div>',
                        rest,
                        re.S | re.I,
                    )
                    snip_clean = ""
                    if snip_match:
                        snip_clean = " ".join(re.sub(r"<[^>]+>", " ", snip_match.group(1)).split())
                    if real_url.startswith("http"):
                        results.append((title_clean, real_url, snip_clean))
        except Exception:
            pass

        # 2. Try DuckDuckGo HTML if needed
        if len(results) < 5:
            try:
                resp = await self._client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if resp.status_code == 200:
                    for m in re.finditer(
                        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
                        r'(?:<a[^>]+class="result__snippet"[^>]*>(.*?)</a>)?',
                        resp.text,
                        re.S | re.I,
                    ):
                        raw_href = unescape(m.group(1))
                        raw_title = unescape(m.group(2))
                        raw_snip = unescape(m.group(3) or "")
                        if "y.js" in raw_href or "ad_domain=" in raw_href:
                            continue
                        if raw_href.startswith("//"):
                            raw_href = "https:" + raw_href
                        if "uddg=" in raw_href:
                            qs = parse_qs(urlparse(raw_href).query)
                            raw_href = unquote(qs.get("uddg", [raw_href])[0])
                        t_clean = " ".join(re.sub(r"<[^>]+>", " ", raw_title).split())
                        s_clean = " ".join(re.sub(r"<[^>]+>", " ", raw_snip).split())
                        if t_clean and raw_href.startswith("http"):
                            results.append((t_clean, raw_href, s_clean))
            except Exception:
                pass

        # 3. Try Wikipedia OpenSearch API if still empty
        if not results:
            try:
                resp = await self._client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "format": "json",
                    },
                )
                if resp.status_code == 200:
                    for item in resp.json().get("query", {}).get("search", [])[:6]:
                        t_clean = item.get("title", "")
                        s_clean = re.sub(r"<[^>]+>", " ", item.get("snippet", ""))
                        u = "https://en.wikipedia.org/wiki/" + quote_plus(t_clean.replace(" ", "_"))
                        results.append((t_clean, u, " ".join(s_clean.split())))
            except Exception:
                pass

        lines = [f"Search Results for: {query}\n"]
        for idx, (t_str, u_str, s_str) in enumerate(results[:10], 1):
            self._blocks.append(("result", f"{idx}. {t_str}", u_str))
            if s_str:
                self._blocks.append(("text", s_str, ""))
            self._links.append((t_str, u_str, "result"))
            self._tag_texts.setdefault("h2", []).append(t_str)
            self._tag_texts.setdefault("h3", []).append(t_str)
            self._tag_texts.setdefault("a", []).append(t_str)
            lines.append(f"[{idx}] {t_str}\n    URL: {u_str}\n    {s_str}\n")

        self._body_text = "\n".join(lines)[:12000]

    async def goto(self, url: str, wait_until: str = "domcontentloaded", timeout: int = 15000) -> None:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        search_q = self._extract_search_query(url)
        if search_q:
            await self._search_fallback(search_q, url)
            return

        self.url = url
        self._scroll_y = 0
        try:
            resp = await self._client.get(url, timeout=timeout / 1000.0)
            self.url = str(resp.url)
            ctype = resp.headers.get("content-type", "").lower()
            if "json" in ctype:
                self._title = self.url
                self._body_text = resp.text[:10000]
                self._blocks = [("text", line, "") for line in self._body_text.splitlines() if line.strip()]
                self._links = []
                self._tag_texts = {"body": [self._body_text]}
                return

            parser = _DOMExtractor(self.url)
            parser.feed(resp.text)
            self._title = parser.title or self.url
            self._blocks = parser.blocks
            self._links = parser.links
            self._tag_texts = parser.tag_texts

            body_lines = []
            if self._title:
                body_lines.append(f"Title: {self._title}\n")
            for kind, text, href in self._blocks:
                if kind in ("h1", "h2", "h3"):
                    body_lines.append(f"\n## {text}")
                elif kind == "link":
                    body_lines.append(f"- {text} ({href})")
                else:
                    body_lines.append(text)
            self._body_text = "\n".join(body_lines)[:12000]

            # If a JS-heavy site returned almost no text, supplement with search context
            if len(self._body_text.strip()) < 120:
                domain_q = f"{urlparse(self.url).netloc} {urlparse(self.url).path.replace('/', ' ')}"
                await self._search_fallback(domain_q.strip(), self.url)
        except Exception as e:
            self._title = f"Error loading {url}"
            self._body_text = f"Failed to load {url}: {e}"
            self._blocks = [("h1", self._title, ""), ("text", self._body_text, "")]

    def _query_selector_texts(self, selector: str) -> list[str]:
        sel = selector.strip()
        if sel in ("body", "html", "main", "article", "#content", "#main"):
            return [self._body_text] if self._body_text else []
        if sel in self._tag_texts:
            return self._tag_texts[sel]
        # Try individual parts of comma-separated or descendant selectors
        for part in re.split(r"[, >+~]+", sel):
            part = part.strip()
            if part in self._tag_texts:
                return self._tag_texts[part]
        return [self._body_text] if self._body_text else []

    async def inner_text(self, selector: str = "body") -> str:
        texts = self._query_selector_texts(selector)
        return "\n".join(texts) if texts else self._body_text

    def locator(self, selector: str) -> _DummyLocator:
        return _DummyLocator(self, selector)

    def get_by_text(self, text: str, exact: bool = False) -> _DummyLocator:
        return _DummyLocator(self, text)

    def get_by_placeholder(self, text: str, exact: bool = False) -> _DummyLocator:
        return _DummyLocator(self, text)

    def get_by_label(self, text: str, exact: bool = False) -> _DummyLocator:
        return _DummyLocator(self, text)

    def get_by_role(self, role: str, name: str = "") -> _DummyLocator:
        return _DummyLocator(self, name or role)

    async def click(self, selector: str, timeout: int = 3000) -> None:
        target = selector.strip().lower()
        for text, href, attrs in self._links:
            if (
                target in text.lower()
                or target in href.lower()
                or (attrs and target.lstrip(".#") in attrs.lower())
            ):
                await self.goto(href)
                return
        raise ValueError(f"Element not found: {selector}")

    async def fill(self, selector: str, value: str, timeout: int = 3000) -> None:
        self._last_input = value

    async def evaluate(self, expression: str) -> Any:
        m = re.search(r"scrollBy\(\s*[^,]+,\s*(-?\d+)\s*\)", expression)
        if m:
            self._scroll_y = max(0, self._scroll_y + int(m.group(1)))
        return None

    async def screenshot(self, type: str = "jpeg", quality: int = 70, timeout: int = 5000) -> bytes:
        import io as _io
        from PIL import Image, ImageDraw

        W, H = 1024, 640
        img = Image.new("RGB", (W, H), (248, 250, 252))
        draw = ImageDraw.Draw(img)

        # Browser top chrome bar
        draw.rectangle([0, 0, W, 68], fill=(24, 24, 28))
        draw.ellipse([16, 14, 28, 26], fill=(239, 68, 68))
        draw.ellipse([36, 14, 48, 26], fill=(245, 158, 11))
        draw.ellipse([56, 14, 68, 26], fill=(34, 197, 94))
        draw.text((84, 13), (self._title or "Browser")[:85], fill=(226, 232, 240))

        # URL bar
        draw.rounded_rectangle([16, 36, W - 16, 60], radius=6, fill=(39, 39, 44))
        draw.text((28, 42), self.url[:125], fill=(161, 161, 170))

        if self.url == "about:blank" and not self._blocks:
            draw.text((W // 2 - 60, H // 2), "Ready to browse...", fill=(148, 163, 184))
        else:
            y = 84 - self._scroll_y
            for kind, text, href in self._blocks:
                if y > H - 20:
                    break
                if kind in ("h1", "h2", "h3"):
                    if y >= 72:
                        draw.text((32, y), text[:105], fill=(15, 23, 42))
                    y += 26
                elif kind == "result":
                    if y >= 72:
                        draw.text((32, y), text[:105], fill=(37, 99, 235))
                        if href:
                            draw.text((32, y + 16), href[:115], fill=(22, 163, 74))
                    y += 36
                elif kind == "link":
                    if y >= 72:
                        draw.text((40, y), f"• {text[:100]}", fill=(79, 70, 229))
                    y += 20
                else:
                    for line in textwrap.wrap(text, width=115)[:4]:
                        if 72 <= y <= H - 18:
                            draw.text((32, y), line, fill=(51, 65, 85))
                        y += 18
                    y += 6

        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


class BrowserAgent:
    def __init__(self, headless: bool = True, model: str = "claude-sonnet-4-6"):
        # Force headless on Linux servers/containers without an X display
        if sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY"):
            headless = True
        self.headless = headless
        self.model = model
        self.client = Anthropic()
        self._playwright = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    async def _start_browser(self) -> None:
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=_CHROMIUM_ARGS,
            )
            self._page = await self._browser.new_page(viewport={"width": 1024, "height": 640})
            self._page.set_default_timeout(10_000)  # 10s instead of Playwright's 30s default
        except Exception as e:
            print(f"Playwright Chromium unavailable ({e}); using serverless HTTP browser fallback.")
            await self._close_browser()
            self._page = _HttpBrowserPage()  # type: ignore[assignment]

    async def _close_browser(self) -> None:
        if self._page:
            try:
                await self._page.close()
            except Exception:
                pass
            self._page = None
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    async def _restart_browser(self) -> None:
        await self._close_browser()
        await self._start_browser()

    async def __aenter__(self):
        await self._start_browser()
        return self

    async def __aexit__(self, *args):
        await self._close_browser()

    async def _screenshot(self) -> str:
        for attempt in range(2):
            try:
                if not self._page:
                    await self._restart_browser()
                data = await self._page.screenshot(type="jpeg", quality=70, timeout=5_000)
                return base64.standard_b64encode(data).decode()
            except Exception as e:
                err = str(e)
                if attempt == 0 and any(
                    k in err for k in ("Connection closed", "Target closed", "has been closed", "crashed")
                ):
                    try:
                        await self._restart_browser()
                    except Exception:
                        return ""
                else:
                    return ""
        return ""

    # ── tool implementations ──────────────────────────────────────────────────

    async def _navigate(self, url: str) -> str:
        await self._page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        await self._page.wait_for_timeout(500)
        return f"Navigated to {url}"

    async def _click_element(self, selector: str) -> str:
        t = 3_000  # 3s per strategy
        for attempt, fn in [
            ("css",         lambda: self._page.click(selector, timeout=t)),
            ("text",        lambda: self._page.get_by_text(selector, exact=False).first.click(timeout=t)),
            ("placeholder", lambda: self._page.get_by_placeholder(selector, exact=False).first.click(timeout=t)),
            ("label",       lambda: self._page.get_by_label(selector, exact=False).first.click(timeout=t)),
            ("role-btn",    lambda: self._page.get_by_role("button", name=selector).first.click(timeout=t)),
            ("role-link",   lambda: self._page.get_by_role("link", name=selector).first.click(timeout=t)),
        ]:
            try:
                await fn()
                return f"Clicked ({attempt}): {selector}"
            except Exception:
                continue
        return f"Could not click: {selector}"

    async def _type_text(self, selector: str, text: str) -> str:
        t = 3_000
        for attempt, fn in [
            ("css",         lambda: self._page.fill(selector, text, timeout=t)),
            ("placeholder", lambda: self._page.get_by_placeholder(selector, exact=False).first.fill(text, timeout=t)),
            ("label",       lambda: self._page.get_by_label(selector, exact=False).first.fill(text, timeout=t)),
            ("role-box",    lambda: self._page.get_by_role("textbox", name=selector).first.fill(text, timeout=t)),
        ]:
            try:
                await fn()
                return f"Typed ({attempt}) '{text}' into {selector}"
            except Exception:
                continue
        return f"Could not type into: {selector}"

    async def _press_key(self, key: str) -> str:
        await self._page.keyboard.press(key)
        return f"Pressed {key}"

    async def _extract_content(self, selector: str | None = None) -> str:
        if selector:
            try:
                loc = self._page.locator(selector)
                await loc.first.wait_for(state="attached", timeout=3_000)
                texts = await loc.all_inner_texts()
                combined = "\n".join(t.strip() for t in texts if t.strip())
                if combined:
                    return combined[:4000]
            except Exception:
                pass
        text = await self._page.inner_text("body")
        return text[:4000]

    async def _scroll_page(self, direction: str, amount: int = 500) -> str:
        delta = amount if direction == "down" else -amount
        await self._page.evaluate(f"window.scrollBy(0, {delta})")
        return f"Scrolled {direction} {amount}px"

    async def _fill_form(self, selector: str, value: str) -> str:
        t = 3_000
        for attempt, fn in [
            ("css",         lambda: self._page.fill(selector, value, timeout=t)),
            ("placeholder", lambda: self._page.get_by_placeholder(selector, exact=False).first.fill(value, timeout=t)),
            ("label",       lambda: self._page.get_by_label(selector, exact=False).first.fill(value, timeout=t)),
            ("role-box",    lambda: self._page.get_by_role("textbox", name=selector).first.fill(value, timeout=t)),
        ]:
            try:
                await fn()
                return f"Filled ({attempt}) {selector} with '{value}'"
            except Exception:
                continue
        return f"Could not fill: {selector}"

    async def _search_web(self, query: str) -> str:
        url = "https://www.google.com/search?q=" + quote_plus(query)
        await self._page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        await self._page.wait_for_timeout(500)
        return f"Searched: {query}"

    async def _get_page_info(self) -> str:
        title = await self._page.title()
        return f"Title: {title}\nURL: {self._page.url}"

    async def _wait_and_extract(self, selector: str, timeout: int = 5000) -> str:
        try:
            loc = self._page.locator(selector)
            await loc.first.wait_for(state="attached", timeout=timeout)
            texts = await loc.all_inner_texts()
            combined = "\n".join(t.strip() for t in texts if t.strip())
            if combined:
                return combined[:4000]
            text = await loc.first.inner_text(timeout=timeout)
            return text[:4000]
        except Exception as e:
            return f"Element not found: {e}"

    # ── dispatch ──────────────────────────────────────────────────────────────

    async def _dispatch_tool(self, name: str, inp: dict[str, Any]) -> tuple[str, bool]:
        """Return (result_text, take_screenshot)."""
        try:
            match name:
                case "navigate":
                    result = await self._navigate(inp["url"])
                case "click_element":
                    result = await self._click_element(inp["selector"])
                case "type_text":
                    result = await self._type_text(inp["selector"], inp["text"])
                case "press_key":
                    result = await self._press_key(inp["key"])
                case "extract_content":
                    result = await self._extract_content(inp.get("selector"))
                case "scroll_page":
                    result = await self._scroll_page(inp["direction"], inp.get("amount", 500))
                case "fill_form":
                    result = await self._fill_form(inp["selector"], inp["value"])
                case "search_web":
                    result = await self._search_web(inp["query"])
                case "get_page_info":
                    result = await self._get_page_info()
                case "wait_and_extract":
                    result = await self._wait_and_extract(inp["selector"], inp.get("timeout", 5000))
                case _:
                    result = f"Unknown tool: {name}"
        except Exception as e:
            err = str(e)
            if any(
                k in err
                for k in ("Connection closed", "Target closed", "has been closed", "crashed")
            ):
                try:
                    await self._restart_browser()
                    return (
                        "Tool failed because the target page crashed the browser. "
                        "The browser has been restarted; try a lighter page or search query.",
                        False,
                    )
                except Exception as restart_err:
                    return f"Browser crashed and restart failed: {str(restart_err)[:200]}", False
            # Return the error as a result so Claude can try a different approach
            return f"Tool failed ({type(e).__name__}): {err[:300]}", True

        return result, name not in _NO_SCREENSHOT

    # ── main loop ─────────────────────────────────────────────────────────────

    async def run(self, task: str, max_steps: int = 25) -> dict[str, Any]:
        total_input_tokens = 0
        total_output_tokens = 0
        steps: list[dict] = []
        output = ""
        success = False

        screenshot_b64 = await self._screenshot()
        init_content: list[dict] = []
        if screenshot_b64:
            init_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": screenshot_b64,
                    },
                }
            )
        init_content.append(
            {
                "type": "text",
                "text": f"Task: {task}\n\nComplete this task using the available tools.",
            }
        )
        messages = [{"role": "user", "content": init_content}]

        for step_num in range(max_steps):
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=_trim_screenshots(messages),
            )

            step_in = response.usage.input_tokens
            step_out = response.usage.output_tokens
            total_input_tokens += step_in
            total_output_tokens += step_out
            print(f"  Step {step_num + 1}: {step_in} in / {step_out} out tokens")

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        output = block.text
                        break
                success = True
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                finished = False

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    steps.append({"tool": block.name, "input": block.input})

                    if block.name == "finish":
                        output = block.input.get("answer", "")
                        success = True
                        finished = True
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": [{"type": "text", "text": "Done."}],
                        })
                        break

                    result_text, take_screenshot = await self._dispatch_tool(
                        block.name, block.input
                    )

                    content: list[dict] = [{"type": "text", "text": result_text}]
                    if take_screenshot:
                        screenshot_b64 = await self._screenshot()
                        if screenshot_b64:
                            content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": screenshot_b64,
                                    },
                                }
                            )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content,
                        }
                    )

                messages.append({"role": "user", "content": tool_results})
                if finished:
                    break
            else:
                break

        estimated_cost_usd = (
            total_input_tokens / 1_000_000 * 3.0
            + total_output_tokens / 1_000_000 * 15.0
        )
        print(f"\nTotal: {total_input_tokens} in / {total_output_tokens} out tokens")
        print(f"Estimated cost: ${estimated_cost_usd:.4f}")

        return {
            "task": task,
            "steps": steps,
            "output": output,
            "success": success,
            "step_count": len(steps),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "estimated_cost_usd": estimated_cost_usd,
        }

    async def run_streaming(self, task: str, max_steps: int = 25):
        """Async generator — yields step events for live UI streaming."""
        total_input_tokens = 0
        total_output_tokens = 0
        screenshots: list[str] = []
        steps_taken: list[dict] = []
        output = ""
        success = False

        screenshot_b64 = await self._screenshot()
        init_content: list[dict] = []
        if screenshot_b64:
            screenshots.append(screenshot_b64)
            yield {"type": "screenshot", "data": screenshot_b64}
            init_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": screenshot_b64,
                    },
                }
            )
        init_content.append(
            {
                "type": "text",
                "text": f"Task: {task}\n\nComplete this task using the available tools.",
            }
        )
        messages = [{"role": "user", "content": init_content}]

        for step_num in range(max_steps):
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=_trim_screenshots(messages),
            )

            step_in = response.usage.input_tokens
            step_out = response.usage.output_tokens
            total_input_tokens += step_in
            total_output_tokens += step_out
            current_cost = (
                total_input_tokens / 1_000_000 * 3.0
                + total_output_tokens / 1_000_000 * 15.0
            )

            yield {
                "type": "tokens",
                "step_num": step_num + 1,
                "step_in": step_in,
                "step_out": step_out,
                "total_in": total_input_tokens,
                "total_out": total_output_tokens,
                "cost_usd": current_cost,
            }

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        output = block.text
                        break
                success = True
                break

            if response.stop_reason == "tool_use":
                tool_results = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    steps_taken.append({"tool": block.name, "input": block.input})

                    if block.name == "finish":
                        output = block.input.get("answer", "")
                        success = True
                        yield {
                            "type": "step",
                            "step_num": step_num + 1,
                            "tool": "finish",
                            "input": block.input,
                            "result": output,
                            "screenshot": None,
                        }
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": [{"type": "text", "text": "Done."}],
                        })
                        break

                    result_text, take_screenshot = await self._dispatch_tool(
                        block.name, block.input
                    )

                    new_screenshot = None
                    content: list[dict] = [{"type": "text", "text": result_text}]

                    if take_screenshot:
                        screenshot_b64 = await self._screenshot()
                        if screenshot_b64:
                            screenshots.append(screenshot_b64)
                            new_screenshot = screenshot_b64
                            content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": screenshot_b64,
                                    },
                                }
                            )

                    yield {
                        "type": "step",
                        "step_num": step_num + 1,
                        "tool": block.name,
                        "input": block.input,
                        "result": result_text,
                        "screenshot": new_screenshot,
                    }

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content,
                        }
                    )

                messages.append({"role": "user", "content": tool_results})
                if success:
                    break
            else:
                break

        estimated_cost_usd = (
            total_input_tokens / 1_000_000 * 3.0
            + total_output_tokens / 1_000_000 * 15.0
        )
        gif_b64 = _make_gif(screenshots)

        yield {
            "type": "done",
            "output": output,
            "success": success,
            "step_count": len(steps_taken),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "gif_b64": gif_b64,
        }
