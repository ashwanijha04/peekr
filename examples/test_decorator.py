import sys
sys.path.insert(0, ".")

import peekr
peekr.instrument(jsonl_path=None)  # console only for this test

from peekr import trace


@trace
def search_web(query: str) -> str:
    return f"Results for: {query}"


@trace(name="tool.calculator")
def calculate(expression: str) -> float:
    return eval(expression)


@trace(name="agent.run")
def run_agent(task: str):
    result1 = search_web(task)
    result2 = calculate("2 + 2")
    return result1, result2


# Async example
import asyncio

@trace
async def async_tool(query: str) -> str:
    await asyncio.sleep(0.01)
    return f"async result for {query}"

@trace(name="agent.async_run")
async def async_agent():
    return await async_tool("hello")


if __name__ == "__main__":
    print("=== Sync agent run ===")
    run_agent("climate change")

    print("\n=== Async agent run ===")
    asyncio.run(async_agent())
