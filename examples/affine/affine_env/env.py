import gc
import time
from typing import Any, Dict
from enum import Enum
from statistics import mean

from affine_env.sat import SATTask, Challenge


class ENVS(Enum):
    SAT = "SAT"
    ABD = "ABD"
    DED = "DED"
    
    
class TASKS:
    SAT: SATTask


class Task:
    def __init__(self, env: str, evaluator: Any, challenges: list[Challenge]):
        self.env = env
        self.evaluate = evaluator
        self.challenges = challenges

class Actor:
    
    def __init__(self) -> None:
        pass

    @staticmethod
    def create_vllm_instance(model_name: str, revision: str, tp: int, gpu_memory_utilization: float = 0.9, download_dir: str = "/root/.cache/huggingface"):
        """Lazily construct and cache a vLLM `LLM` instance for the given model"""
        from vllm import LLM  # type: ignore
        return LLM(
            model=model_name,
            revision=revision,
            trust_remote_code=True,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_memory_utilization,
            download_dir=download_dir,
        )

    @staticmethod
    def create_task_instance(env: str, samples: int, **kwargs) -> Task:
        """Create a task instance with generated challenges."""
        env_enum = ENVS[env_key] if (env_key := env.strip().upper()) in ENVS.__members__ else (_ for _ in ()).throw(ValueError(f"Unsupported env '{env}'."))
        
        # task_class = SATTask.get(env_enum.value)
        # if not task_class:
        #     raise ValueError(f"No task implementation for env '{env_enum.value}'")
        
        task_instance = SATTask()
        challenges = task_instance.generate(samples=samples, **kwargs)
        
        return Task(env=env_enum.value, evaluator=task_instance.evaluate, challenges=challenges)

    def evaluate(
        self,
        llm: Any,
        task: Task,
        sampling_params: Any
    ) -> Dict[str, Any]:
        """Run evaluation on challenges and return aggregated results."""
        start = time.time()
        
        prompts = [challenge.prompt for challenge in task.challenges]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        
        response_texts = [
            output.outputs[0].text.strip() if output.outputs else ""
            for output in outputs
        ]
        
        scores = task.evaluate(response_texts, task.challenges)
        
        rollouts: list[Dict[str, Any]] = []
        
        for idx, (challenge, output, response_text, score) in enumerate(zip(task.challenges, outputs, response_texts, scores)):
            conversation = [
                {"role": "user", "content": challenge.prompt},
                {"role": "assistant", "content": response_text}
            ]
            
            metrics = {}
            if hasattr(output, 'metrics') and output.metrics:
                m = output.metrics
                metrics['time_in_queue'] = m.time_in_queue if hasattr(m, 'time_in_queue') else None
                
                if hasattr(m, 'first_token_time') and hasattr(m, 'first_scheduled_time'):
                    if m.first_token_time is not None and m.first_scheduled_time is not None:
                        metrics['time_to_first_token'] = m.first_token_time - m.first_scheduled_time
                    else:
                        metrics['time_to_first_token'] = None
                else:
                    metrics['time_to_first_token'] = None
                
                if hasattr(m, 'finished_time') and hasattr(m, 'arrival_time'):
                    if m.finished_time is not None and m.arrival_time is not None:
                        metrics['total_time'] = m.finished_time - m.arrival_time
                    else:
                        metrics['total_time'] = None
                else:
                    metrics['total_time'] = None
            
            rollout = {
                "task_name": f"affine:{task.env.lower()}",
                "score": score,
                "success": score > 0,
                "extra": {
                    "conversation": conversation,
                    "sample_idx": idx,
                    "metrics": metrics
                }
            }
            
            rollouts.append(rollout)
        
        successes = sum(1 for score in scores if score > 0)
        avg_reward = mean(scores) if scores else 0.0
        success_rate = successes / len(scores) if scores else 0.0
        
        result = {
            "env": task.env,
            "total_samples": len(rollouts),
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "timestamp": time.time(),
            "total_evaluation_time": time.time() - start,
            "rollouts": rollouts
        }
        
        del task
        del llm
        gc.collect()
        
        return result

