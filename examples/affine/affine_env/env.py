import asyncio
import time
from typing import Any, Dict, List, Sequence

import sglang as sgl  # type: ignore
import sglang.test.doc_patch  # type: ignore  # noqa: F401

from .sat import Challenge, SATTask


class Actor:
    """Adapter that runs Affine tasks using an offline sglang engine."""

    _TASK_REGISTRY = {
        "sat": SATTask,
    }

    @staticmethod
    def create_sglang_instance(model_path: str, **engine_kwargs: Any) -> "sgl.Engine":
        """Instantiate the offline sglang engine."""
        config = {"model_path": model_path}
        config.update(engine_kwargs)
        return sgl.Engine(**config)

    @staticmethod
    def create_task_instance(env: str) -> SATTask:
        """Return the task implementation for the requested environment."""
        task_cls = Actor._TASK_REGISTRY.get(env.lower())
        if task_cls is None:
            supported = ", ".join(sorted(Actor._TASK_REGISTRY.keys()))
            raise ValueError(f"Unsupported env '{env}'. Supported envs: {supported}")
        return task_cls()

    @staticmethod
    async def _async_generate_batches(
        llm: "sgl.Engine",
        prompts: Sequence[str],
        sampling_params: Dict[str, Any],
        batch_size: int,
    ) -> List[str]:
        """Generate texts for the provided prompts in fixed-size batches."""
        responses: List[str] = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            outputs = await llm.async_generate(batch_prompts, sampling_params)
            responses.extend(output.get("text", "") for output in outputs)
        return responses

    def evaluate(
        self,
        llm: "sgl.Engine",
        task: SATTask,
        sampling_params: Dict[str, Any],
        *,
        env: str,
        samples: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        """Run batched offline inference and evaluate task rewards."""
        started_at = time.time()
        challenges: List[Challenge] = task.generate(samples=samples)
        prompts = [challenge.prompt for challenge in challenges]

        responses = asyncio.run(
            self._async_generate_batches(llm, prompts, sampling_params, batch_size)
        )

        rewards = task.evaluate(responses, challenges)
        total_samples = len(challenges)
        total_reward = sum(rewards)
        avg_reward = total_reward / total_samples if total_samples else 0.0
        success_rate = (
            sum(1 for reward in rewards if reward > 0) / total_samples
            if total_samples
            else 0.0
        )

        rollouts: List[Dict[str, Any]] = []
        for idx, (challenge, response, reward) in enumerate(
            zip(challenges, responses, rewards)
        ):
            conversation = [
                {"role": "user", "content": challenge.prompt},
                {"role": "assistant", "content": response},
            ]
            rollouts.append(
                {
                    "task_name": f"affine:{(challenge.env or env).lower()}",
                    "score": reward,
                    "success": reward > 0,
                    "extra": {
                        "conversation": conversation,
                        "sample_idx": idx,
                        "metrics": {},
                        "details": challenge.extra,
                        "timestamp": challenge.timestamp,
                    },
                }
            )

        return {
            "env": challenges[0].env if challenges else env,
            "total_samples": total_samples,
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "timestamp": started_at,
            "total_evaluation_time": time.time() - started_at,
            "rollouts": rollouts,
        }

