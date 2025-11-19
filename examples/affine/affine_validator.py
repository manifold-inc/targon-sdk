import json
import sys
import time
import targon
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

AFFINE_DIR = Path(__file__).resolve().parent
AFFINE_ENV_SOURCE = AFFINE_DIR / "affine_env"
AFFINE_REQUIREMENTS = AFFINE_DIR / "requirements.txt"

image = (
    targon.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("libnuma1")
    .pip_install_from_requirements(str(AFFINE_REQUIREMENTS))
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(AFFINE_ENV_SOURCE), "/workspace/affine_env")
)

app = targon.App("affine", image=image)


@app.function(resource=targon.Compute.H200_MEDIUM, timeout=1500, min_replicas=0, container_concurrency=1, max_replicas=3)
def run(
    model_name: str,
    env: str,
    n: int,
    *,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Run `n` Affine evaluations using batched offline sglang inference."""
    sys.path.insert(0, "/workspace")
    import gc
    import torch  # type: ignore
    from affine_env.env import Actor

    gc.collect()
    torch.cuda.empty_cache()
    
    # Log GPU memory status
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        total_mem = torch.cuda.mem_get_info()[1] / 1024**3
        print(f"GPU Memory: {free_mem:.2f} GiB free / {total_mem:.2f} GiB total")

    if batch_size <= 0:
        raise ValueError("`batch_size` must be a positive integer.")
    if n % batch_size != 0:
        raise ValueError(
            f"`n` ({n}) must be divisible by `batch_size` ({batch_size}) for batched inference."
        )

    sampling_params = {
        "temperature": temperature,
        "top_p": 0.8,
        "max_tokens": max_new_tokens,
    }

    print(
        f"Evaluating {n} samples for model '{model_name}' on environment '{env}' "
        f"with batch size {batch_size}."
    )
    time.sleep(1)

    actor = Actor()
    llm = Actor.create_sglang_instance(model_name)
    task = Actor.create_task_instance(env=env)
    
    result = actor.evaluate(
        llm,
        task,
        sampling_params,
        env=env,
        samples=n,
        batch_size=batch_size,
    )
    result["model_name"] = model_name
    result["batch_size"] = batch_size
    
    return result


ENV_ALIASES = ["SAT"]


@app.local_entrypoint()
async def main(
    model_name: str,
    env: str,
    n: int,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Run the Affine validator remotely and save rollouts locally.
    targon run examples/affine/affine_validator.py --model-name "Qwen/Qwen2.5-7B-Instruct" --env "sat" --n 512 --batch-size 32
    """
    if env.upper() not in ENV_ALIASES: 
        raise ValueError(f"Unsupported env '{env}', available envs {', '.join(ENV_ALIASES)}")

    result = await run.remote(
        model_name=model_name,
        env=env,
        n=n,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        timeout=15000
    )

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(f"rollouts_{env}_{timestamp}.json")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {
        "model_name": result["model_name"],
        "env": result["env"],
        "total_samples": result["total_samples"],
        "avg_reward": result["avg_reward"],
        "success_rate": result["success_rate"],
        "batch_size": batch_size,
        "output_path": str(output_path),
    }

    print(json.dumps(summary, indent=2))
    return result