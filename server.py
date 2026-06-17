import asyncio
import json
import queue
import sys
from pathlib import Path
from threading import Thread

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import BrowserAgent

load_dotenv(override=True)

app = FastAPI(title="Browser Agent")

STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


class RunRequest(BaseModel):
    task: str
    headless: bool = True
    max_steps: int = 25


def _agent_thread(q: queue.Queue, task: str, headless: bool, max_steps: int) -> None:
    """Run BrowserAgent in a dedicated OS thread with its own event loop.

    On Windows, uvicorn's SelectorEventLoop doesn't support subprocess creation
    (which Playwright needs). Running in a separate thread with a ProactorEventLoop
    bypasses that restriction cleanly.
    """
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run():
        try:
            async with BrowserAgent(headless=headless) as agent:
                async for event in agent.run_streaming(task, max_steps):
                    q.put(event)
        except Exception as e:
            import traceback
            traceback.print_exc()
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(None)  # sentinel: tells the SSE stream we're done

    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/run")
async def run_task(req: RunRequest):
    q: queue.Queue = queue.Queue()
    Thread(
        target=_agent_thread,
        args=(q, req.task, req.headless, req.max_steps),
        daemon=True,
    ).start()

    async def stream():
        main_loop = asyncio.get_event_loop()
        while True:
            try:
                event = await main_loop.run_in_executor(
                    None, lambda: q.get(block=True, timeout=0.5)
                )
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                continue

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
