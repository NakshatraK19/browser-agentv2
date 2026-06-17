import asyncio
import base64
from typing import Any

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
    """Stitch base64 PNG screenshots into a base64 GIF."""
    try:
        import io as _io
        from PIL import Image

        frames = []
        for b64 in screenshots:
            raw = base64.b64decode(b64)
            img = Image.open(_io.BytesIO(raw)).convert("RGB")
            h = int(img.height * width / img.width)
            frames.append(img.resize((width, h), Image.LANCZOS).quantize(colors=256))

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


class BrowserAgent:
    def __init__(self, headless: bool = True, model: str = "claude-sonnet-4-6"):
        self.headless = headless
        self.model = model
        self.client = Anthropic()
        self._playwright = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        self._page = await self._browser.new_page(viewport={"width": 1024, "height": 640})
        self._page.set_default_timeout(10_000)  # 10s instead of Playwright's 30s default
        return self

    async def __aexit__(self, *args):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def _screenshot(self) -> str:
        data = await self._page.screenshot(type="jpeg", quality=70)
        return base64.standard_b64encode(data).decode()

    # ── tool implementations ──────────────────────────────────────────────────

    async def _navigate(self, url: str) -> str:
        await self._page.goto(url)
        return f"Navigated to {url}"

    async def _click_element(self, selector: str) -> str:
        t = 3_000  # 3s per strategy
        for attempt, fn in [
            ("css",         lambda: self._page.click(selector, timeout=t)),
            ("text",        lambda: self._page.get_by_text(selector, exact=False).first.click(timeout=t)),
            ("placeholder", lambda: self._page.get_by_placeholder(selector, exact=False).first.click(timeout=t)),
            ("label",       lambda: self._page.get_by_label(selector, exact=False).first.click(timeout=t)),
            ("role-btn",    lambda: self._page.get_by_role("button", name=selector).first.click(timeout=t)),
        ]:
            try:
                await fn()
                return f"Clicked ({attempt}): {selector}"
            except Exception:
                continue
        return f"Could not click: {selector}"

    async def _type_text(self, selector: str, text: str) -> str:
        await self._page.fill(selector, text)
        return f"Typed '{text}' into {selector}"

    async def _press_key(self, key: str) -> str:
        await self._page.keyboard.press(key)
        return f"Pressed {key}"

    async def _extract_content(self, selector: str | None = None) -> str:
        if selector:
            try:
                text = await self._page.locator(selector).inner_text()
                return text[:4000]
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
        ]:
            try:
                await fn()
                return f"Filled ({attempt}) {selector} with '{value}'"
            except Exception:
                continue
        return f"Could not fill: {selector}"

    async def _search_web(self, query: str) -> str:
        url = "https://www.google.com/search?q=" + query.replace(" ", "+")
        await self._page.goto(url)
        return f"Searched: {query}"

    async def _get_page_info(self) -> str:
        title = await self._page.title()
        return f"Title: {title}\nURL: {self._page.url}"

    async def _wait_and_extract(self, selector: str, timeout: int = 5000) -> str:
        try:
            await self._page.wait_for_selector(selector, timeout=timeout)
            text = await self._page.locator(selector).inner_text()
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
            # Return the error as a result so Claude can try a different approach
            return f"Tool failed ({type(e).__name__}): {str(e)[:300]}", True

        return result, name not in _NO_SCREENSHOT

    # ── main loop ─────────────────────────────────────────────────────────────

    async def run(self, task: str, max_steps: int = 25) -> dict[str, Any]:
        total_input_tokens = 0
        total_output_tokens = 0
        steps: list[dict] = []
        output = ""
        success = False

        screenshot_b64 = await self._screenshot()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": screenshot_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Task: {task}\n\nComplete this task using the available tools.",
                    },
                ],
            }
        ]

        for step_num in range(max_steps):
            response = self.client.messages.create(
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
        screenshots.append(screenshot_b64)
        yield {"type": "screenshot", "data": screenshot_b64}

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": screenshot_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Task: {task}\n\nComplete this task using the available tools.",
                    },
                ],
            }
        ]

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
                        messages.append({"role": "user", "content": tool_results})
                        break

                    result_text, take_screenshot = await self._dispatch_tool(
                        block.name, block.input
                    )

                    new_screenshot = None
                    content: list[dict] = [{"type": "text", "text": result_text}]

                    if take_screenshot:
                        screenshot_b64 = await self._screenshot()
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
