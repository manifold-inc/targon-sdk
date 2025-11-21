import random
import re
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class Challenge:
    env: str
    prompt: str
    extra: Dict[str, Any]
    timestamp: Optional[float]


class SATTask:
    async def _generate(self, n: int = 15, k: int = 10) -> Challenge:
        m = int(4.26 * n)
        sol = {i: random.choice([True, False]) for i in range(1, n + 1)}

        cls = []
        for _ in range(m):
            vs = random.sample(list(sol), k)
            sv = random.choice(vs)
            cls.append(
                [
                    (v if sol[v] else -v)
                    if v == sv
                    else (v if random.choice([True, False]) else -v)
                    for v in vs
                ]
            )

        formula = " ∧ ".join(
            "(" + " ∨ ".join(f"{'¬' if l < 0 else ''}x{abs(l)}" for l in c) + ")"
            for c in cls
        )

        prompt = (
            f"Find a satisfying assignment for the following {k}-SAT formula over variables x1..x{n}:\n"
            f"{formula}\n"
            "Provide your answer as comma-separated assignments like `x1=True, x2=False, ...`, "
            "or respond `UNSAT` if it has no solution."
        )

        return Challenge(
            env="sat",
            prompt=prompt,
            extra={"solution": sol, "clauses": cls},
            timestamp=1,
        )

    async def _generate_async(self, samples: int, n: int, k: int) -> list[Challenge]:
        """Generate multiple SAT problems in parallel."""
        tasks = [self._generate(n, k) for _ in range(samples)]
        return await asyncio.gather(*tasks)

    def generate(self, samples: int = 10, n: int = 15, k: int = 10) -> list[Challenge]:
        """Synchronous entry point for generating multiple SAT problems."""
        return asyncio.run(self._generate_async(samples, n, k))

        
    async def _evaluate(self, response: str, challenge: Challenge) -> float:
        """Evaluate SAT response."""
        cls = challenge.extra.get("clauses", [])

        got = {
            int(v): val.lower() in ("true", "1")
            for v, val in re.findall(r"x(\d+)=(True|False|1|0)", response or "")
        }

        ok = all(any((lit > 0) == got.get(abs(lit), None) for lit in c) for c in cls)
        return float(ok)
    
    async def _evaluate_async(self, responses: list[str], challenges: list[Challenge]) -> list[float]:
        """Evaluate multiple SAT responses in parallel."""
        tasks = [self._evaluate(response, challenge) for response, challenge in zip(responses, challenges)]
        return await asyncio.gather(*tasks)
    
    def evaluate(self, responses: list[str], challenges: list[Challenge]) -> list[float]:
        """Synchronous entry point for generating multiple SAT problems."""
        return asyncio.run(self._evaluate_async(responses, challenges))
