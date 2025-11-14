"""Serverless Affine validator using offline vLLM inference."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import targon  # type: ignore[import-not-found]

AFFINE_DIR = Path(__file__).resolve().parent
AFFINE_ENV_SOURCE = AFFINE_DIR / "affine_env"
AFFINE_REQUIREMENTS = AFFINE_DIR / "requirements.txt"

affine_image = (
    targon.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install("vllm==0.10.2",
        "torch==2.8.0",
        "huggingface_hub==0.35.0",
        "hf_transfer")
    .pip_install_from_requirements(str(AFFINE_REQUIREMENTS))
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(AFFINE_ENV_SOURCE), "/workspace/affine_env")
)

app = targon.App("affine", image=affine_image)

ENV_ALIASES = {
    "SAT": "sat",
    "ABD": "abd",
    "DED": "ded",
}


def _normalize_env(env: str) -> str:
    key = env.strip().upper()
    if key not in ENV_ALIASES:
        choices = ", ".join(sorted(ENV_ALIASES))
        raise ValueError(f"Unsupported env '{env}'. Available values: {choices}")
    return ENV_ALIASES[key]


def _coerce_json(value: Any) -> Any:
    """Ensure returned data is JSON serialisable."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): _coerce_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_coerce_json(v) for v in value]
        return str(value)


@app.function(resource=targon.Compute.H200_MEDIUM, timeout=900, min_replicas=0, max_concurrency=0)
def run_env(
    model_name: str,
    env: str,
    n: int,
    *,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Run `n` Affine evaluations using offline vLLM inference."""
    import asyncio
    import sys
    from statistics import mean

    sys.path.insert(0, "/workspace")

    task_type = _normalize_env(env)

    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    llm = LLM(
        model=model_name,
        revision="main",
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        download_dir="/root/.cache/huggingface",
    )

    async def _run_evaluations() -> Dict[str, Any]:
        # Import inside async function so event loop is available
        import affine_env  # noqa: F401
        from affine_env.env import Actor

        async def offline_llm(
            *,
            prompt: str,
            model: str,
            temperature: float,
            timeout: float,
            base_url: str,
            api_key: str | None,
            seed: int | None,
            **_: Any,
        ) -> str:
            params = SamplingParams(
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_new_tokens,
                seed=seed,
            )
            loop = asyncio.get_running_loop()

            def _generate() -> str:
                outputs = llm.generate([prompt], sampling_params=params, use_tqdm=True)
                if not outputs:
                    raise RuntimeError("vLLM returned no outputs")
                generations = outputs[0].outputs
                if not generations:
                    raise RuntimeError("vLLM returned empty generations")
                return generations[0].text.strip()

            return await loop.run_in_executor(None, _generate)

        actor = Actor(llm_fn=offline_llm)

        rollouts: list[Dict[str, Any]] = []
        scores: list[float] = []

        for _ in range(n):
            result = await actor.evaluate(
                task_type=task_type,
                model=model_name,
                base_url="offline://vllm",
                timeout=600,
                temperature=temperature,
            )
            rollouts.append(_coerce_json(result))
            scores.append(float(result.get("score", 0.0)))

        successes = sum(1 for score in scores if score > 0)
        avg_reward = mean(scores) if scores else 0.0
        success_rate = successes / len(scores) if scores else 0.0

        return _coerce_json(
            {
            "model_name": model_name,
            "env": task_type,
            "total_samples": len(rollouts),
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "timestamp": time.time(),
            "rollouts": rollouts,
            }
        )

    return asyncio.run(_run_evaluations())


@app.local_entrypoint()
async def main(
    model_name: str,
    env: str,
    n: int,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Run the Affine validator remotely and save rollouts locally."""
    normalized_env = _normalize_env(env)

    result = await run_env.remote(
        model_name=model_name,
        env=normalized_env,
        n=n,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(f"rollouts_{normalized_env}_{timestamp}.json")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {
        "model_name": result["model_name"],
        "env": result["env"],
        "total_samples": result["total_samples"],
        "avg_reward": result["avg_reward"],
        "success_rate": result["success_rate"],
        "output_path": str(output_path),
    }

    print(json.dumps(summary, indent=2))
    return result
