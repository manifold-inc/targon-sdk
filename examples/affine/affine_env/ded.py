# from __future__ import annotations

# import ast
# import asyncio
# import json
# import logging
# from typing import Any, Dict

# from .dataset import R2Dataset
# from affine_env.executor import ProgramExecutor
# from .models import Challenge

# # Logger
# logger = logging.getLogger("affine")


# # -------------------------------- Helpers -------------------------------- #
# def _to_str(payload) -> str:
#     """
#     Canonicalise any JSON-serialisable test-case payload to a single
#     newline-delimited string suitable for feeding to ``stdin``.
#     """
#     if isinstance(payload, str):
#         return payload
#     if isinstance(payload, (bytes, bytearray)):
#         return payload.decode()
#     if isinstance(payload, list):
#         # Recursively stringify nested lists and join with newlines
#         return "\n".join(_to_str(element) for element in payload)
#     # Dicts / numbers / other scalars → JSON text
#     return json.dumps(payload, ensure_ascii=False)


# def _normalize(text: str) -> str:
#     """Trim trailing blank lines and per-line trailing spaces."""
#     return "\n".join(line.rstrip() for line in text.rstrip().splitlines())


# class DEDTask:
#     """DED (Direct Execution Debug) task - Python program generation from requirements."""

#     def __init__(self, dataset=None, dataset_name: str = "satpalsr/rl-python"):
#         """
#         Initialize DED task.

#         Args:
#             dataset: Optional pre-initialized R2Dataset instance to use
#             dataset_name: Name of the R2 dataset to use (only if dataset not provided)
#         """
#         self._executor = ProgramExecutor()
#         self._dataset = dataset if dataset is not None else R2Dataset(dataset_name=dataset_name)

#     async def generate(self) -> Challenge:
#         """Generate a coding challenge from R2 dataset."""
#         logger.debug("Generating DED challenge")
#         sample = await self._dataset.get()

#         if sample is None:
#             raise RuntimeError("Failed to fetch dataset row")

#         # Add extra instructions to ensure proper formatting
#         extra_hint = (
#             "\n\n---\n"
#             "⚠️ **Instructions** ⚠️\n"
#             "Write a complete **Python 3** program that\n"
#             "• reads *all* input from **STDIN** (using `input()` / `sys.stdin`),\n"
#             "• writes *only* the required answer(s) to **STDOUT** using `print`,\n"
#             "• contains no additional prompts or debug text, and\n"
#             "• is returned as a single ```python … ``` fenced block.\n"
#         )

#         prompt = sample["prompt"].rstrip() + extra_hint

#         return Challenge(env="affine:ded", prompt=prompt, extra={"sample": sample})

#     async def evaluate(self, response: str, challenge: Challenge) -> float:
#         """Evaluate program against test cases."""
#         logger.debug("Evaluating DED response")

#         sample = challenge.extra.get("sample", {})

#         raw_reply = response
#         program = self._executor._strip_fences(raw_reply)
#         logger.debug("Stripped program: %s...", program[:50])

#         # Get verification info
#         ver_raw = sample.get("verification_info") or sample.get("test_cases")
#         logger.debug("Verification raw: %s...", str(ver_raw)[:50])

#         # Parse verification info (try JSON first, then Python literal)
#         try:
#             if isinstance(ver_raw, str):
#                 try:
#                     ver_json = json.loads(ver_raw)
#                     logger.debug("Parsed via json.loads")
#                 except json.JSONDecodeError:
#                     ver_json = ast.literal_eval(ver_raw)
#                     logger.debug("Parsed via ast.literal_eval")
#             else:
#                 ver_json = ver_raw
#         except Exception as err:
#             logger.warning("Failed to parse verification info: %s", err)
#             return 0.0

#         # Extract test cases
#         cases = ver_json.get("test_cases")
#         if not cases:
#             logger.debug("No test_cases found")
#             return 0.0

#         logger.debug("Found %d test cases", len(cases))

#         loop = asyncio.get_running_loop()
#         passed, total = 0, len(cases)

#         for idx, case in enumerate(cases, start=1):
#             case_type = case.get("type")
#             raw_inp = case.get("input")
#             raw_exp = case.get("output")

#             if case_type == "stdin_stdout":
#                 inp = _to_str(raw_inp)
#                 if not inp.endswith("\n"):
#                     inp += "\n"
#                 exec_prog = program
#                 exp = _to_str(raw_exp)
#             elif case_type == "function_call":
#                 fn_name = case.get("fn_name")
#                 args = case.get("input", [])
#                 # Wrap program with function call
#                 exec_prog = (
#                     program
#                     + "\n"
#                     + "if __name__ == '__main__':\n"
#                     + f"    result = {fn_name}(*{args!r})\n"
#                     + "    print(result)"
#                 )
#                 inp = ""
#                 exp = _to_str(raw_exp[0]) if isinstance(raw_exp, list) and raw_exp else _to_str(raw_exp)
#             else:
#                 logger.debug("Unknown test case type '%s', skipping", case_type)
#                 total -= 1
#                 continue

#             try:
#                 out, err = await loop.run_in_executor(None, self._executor.execute, exec_prog, inp)
#             except Exception as exc:
#                 out, err = "", str(exc)

#             ok_run = not err.strip()
#             out_norm = _normalize(out)
#             exp_norm = _normalize(exp) if exp is not None else None
#             correct = ok_run and (exp_norm is None or out_norm == exp_norm)

#             if correct:
#                 passed += 1
#                 logger.debug("Test case %d passed", idx)
#             else:
#                 logger.debug(
#                     "Test case %d failed. Got: %r, Expected: %r",
#                     idx,
#                     out_norm,
#                     exp_norm,
#                 )

#         score = 1.0 if passed == total else 0.0
#         logger.debug("DED evaluation completed with score: %s (%d/%d)", score, passed, total)

#         return score

