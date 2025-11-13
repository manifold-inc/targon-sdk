"""Async helpers for calling remote LLM APIs."""

from __future__ import annotations

import httpx
import openai


async def call_openai_chat(
    *,
    prompt: str,
    model: str,
    temperature: float,
    timeout: float,
    base_url: str,
    api_key: str,
    seed: int | None,
) -> str:
    """Call an OpenAI-compatible chat completion endpoint."""
    client = openai.AsyncOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        timeout=httpx.Timeout(timeout),
        max_retries=0,
    )

    params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    }

    if seed is not None:
        params["seed"] = seed

    response = await client.chat.completions.create(**params)

    if not response.choices:
        raise ValueError("LLM API returned empty choices list")

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM API returned None content (possible content filtering or API error)")

    return content.strip()

