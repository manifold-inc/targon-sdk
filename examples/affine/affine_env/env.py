"""Affine environment actor for bundled evaluations."""

from __future__ import annotations

import gc
import os
import random
import time
from typing import Callable, Optional

from .abd import ABDTask
from .dataset import R2Dataset
from .ded import DEDTask
from .sat import SATTask

# Global R2Dataset instance - created on module import to trigger background download
_global_dataset = R2Dataset(dataset_name="satpalsr/rl-python")


class Actor:
    """Multi-task evaluation actor."""

    # Task registry - map task_type to task class
    TASKS = {
        "sat": SATTask,
        "abd": ABDTask,
        "ded": DEDTask,
    }

    def __init__(
        self,
        *,
        llm_fn: Optional[Callable[[str, dict], str]] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Actor with API key and optional LLM callable.

        Args:
            llm_fn: Optional callable used for LLM inference. Signature:
                ``llm_fn(prompt: str, *, temperature: float, seed: int) -> str``.
            api_key: API key for LLM service (when llm_fn is not provided).
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self._llm_fn = llm_fn

    async def _llm_chat(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        timeout: float,
        base_url: str,
        seed: Optional[int],
        api_key: Optional[str],
    ) -> str:
        """Call LLM implementation."""
        if self._llm_fn is None:
            from .llm_proxy import call_openai_chat

            return await call_openai_chat(
                prompt=prompt,
                model=model,
                temperature=temperature,
                timeout=timeout,
                base_url=base_url,
                api_key=api_key or self.api_key,
                seed=seed,
            )

        return await self._llm_fn(
            prompt=prompt,
            model=model,
            temperature=temperature,
            timeout=timeout,
            base_url=base_url,
            seed=seed,
            api_key=api_key or self.api_key,
        )

    async def evaluate(
        self,
        task_type: str = "sat",
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        timeout: float = 600,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Run evaluation on a single task.

        Args:
            task_type: Type of task to evaluate (sat, abd, ded)
            model: Model name to use for evaluation
            base_url: Base URL for LLM API
            timeout: Timeout for LLM API calls
            temperature: Temperature for LLM generation
            api_key: Override API key for this evaluation.
            seed: Random seed for LLM generation. If not provided, a random seed will be generated.
        """
        # Generate random seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Allow per-call api_key override
        current_api_key = api_key or self.api_key
        # Get task class from registry
        task_cls = self.TASKS.get(task_type)
        if not task_cls:
            raise ValueError(f"Unknown task: {task_type}. Available: {list(self.TASKS.keys())}")

        # Initialize task instance, passing global dataset if task supports it
        if task_type in ("abd", "ded"):
            task_instance = task_cls(dataset=_global_dataset)
        else:
            task_instance = task_cls()

        start = time.time()

        # Generate challenge (unified async interface)
        challenge = await task_instance.generate()

        # Call LLM
        try:
            resp = await self._llm_chat(
                prompt=challenge.prompt,
                model=model,
                temperature=temperature,
                timeout=timeout,
                base_url=base_url,
                api_key=current_api_key,
                seed=seed,
            )
            error = None
        except Exception as exc:
            import traceback

            resp = None
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

        # Evaluate (unified async interface)
        score = 0.0
        if resp:
            score = await task_instance.evaluate(resp, challenge)

        conversation = [
            {"role": "user", "content": challenge.prompt},
            {"role": "assistant", "content": resp},
        ]

        result = {
            "task_name": f"affine:{task_type}",
            "score": score,
            "success": score > 0,
            "time_taken": time.time() - start,
            "extra": {"conversation": conversation, "seed": seed},
        }

        # Add error info if present
        if error:
            result["error"] = error
            result["error_type"] = "llm_failure"

        # Force garbage collection to free memory immediately
        del task_instance
        gc.collect()

        return result

