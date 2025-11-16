"""Serverless Affine validator using offline vLLM inference."""

from __future__ import annotations

import asyncio
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
    .pip_install("vllm==0.11.0",
        "flashinfer-python==0.5.2")
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


@app.function(resource=targon.Compute.H200_SMALL, timeout=900, min_replicas=0, max_replicas=10, max_concurrency=1)
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
        tensor_parallel_size=1,
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
    # model_name: str | list[str],
    env: str,
    n: int,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    num_models: int = 3,
) -> Dict[str, Any]:
    """Run the Affine validator remotely and save rollouts locally."""
    normalized_env = _normalize_env(env)

    import urllib.request
    import json

    # Fetch weights from affine.io API
    url = "https://www.affine.io/api/weights"
    with urllib.request.urlopen(url) as response:
        resp_json = json.loads(response.read().decode())

    # Example: extract the list of weights from the third column in rows
    # The API returns something like: resp_json['data']['rows']
    affine_model_names = [row[2] for row in resp_json['data']['rows']][:num_models]

    # Execute all models in parallel instead of using sequential map()
    tasks = [
        run_env.remote(
            model_name,
            normalized_env,
            n,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        for model_name in affine_model_names
    ]
    results = await asyncio.gather(*tasks)

    if not results:
        raise RuntimeError("No results returned from remote evaluations")

    single_model = len(results) == 1
    payload = results[0] if single_model else results

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(f"rollouts_{normalized_env}_{timestamp}.json")
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if single_model:
        summary: Dict[str, Any] = {
            "model_name": results[0]["model_name"],
            "env": results[0]["env"],
            "total_samples": results[0]["total_samples"],
            "avg_reward": results[0]["avg_reward"],
            "success_rate": results[0]["success_rate"],
            "output_path": str(output_path),
        }
    else:
        summary = {
            "env": normalized_env,
            "total_models": len(results),
            "models": [
                {
                    "model_name": res["model_name"],
                    "total_samples": res["total_samples"],
                    "avg_reward": res["avg_reward"],
                    "success_rate": res["success_rate"],
                }
                for res in results
            ],
            "output_path": str(output_path),
        }

    print(json.dumps(summary, indent=2))
    return payload
