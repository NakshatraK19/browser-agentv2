import asyncio
import json
from agent import BrowserAgent


async def _run(task: str, headless: bool = True, max_steps: int = 25) -> dict:
    async with BrowserAgent(headless=headless) as agent:
        return await agent.run(task, max_steps=max_steps)


def scrape(url: str, what_to_extract: str, headless: bool = True) -> dict:
    task = f"Go to {url} and extract: {what_to_extract}"
    return asyncio.run(_run(task, headless=headless))


def research(topic: str, num_sources: int = 3, headless: bool = True) -> dict:
    task = (
        f"Research '{topic}' by searching the web and gathering information from "
        f"{num_sources} different sources. Summarize the key findings."
    )
    return asyncio.run(_run(task, headless=headless))


def fill_form_data(
    url: str, form_data: dict, submit: bool = False, headless: bool = True
) -> dict:
    fields = ", ".join(f"{k}={v}" for k, v in form_data.items())
    task = f"Go to {url}, fill in the form with these values: {fields}"
    task += (
        ", then submit the form." if submit else ", but do NOT submit the form."
    )
    return asyncio.run(_run(task, headless=headless))


def summarize_page(url: str, headless: bool = True) -> dict:
    task = f"Go to {url} and provide a comprehensive summary of the page content."
    return asyncio.run(_run(task, headless=headless))


async def _demo():
    demo_tasks = [
        ("scrape", "Go to https://example.com and extract: the main heading and description"),
        ("research", "Research 'Python async programming best practices' by searching the web and gathering information from 2 different sources. Summarize the key findings."),
        ("summarize", "Go to https://httpbin.org/html and provide a comprehensive summary of the page content."),
    ]

    results = []
    for task_type, task in demo_tasks:
        print(f"\n{'=' * 60}")
        print(f"Running: {task_type}")
        print("=" * 60)

        result = await _run(task)

        results.append({"type": task_type, "task": task, "result": result})
        print(f"Success: {result['success']}  steps: {result['step_count']}  cost: ${result['estimated_cost_usd']:.4f}")

    with open("demo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to demo_results.json")


if __name__ == "__main__":
    asyncio.run(_demo())
