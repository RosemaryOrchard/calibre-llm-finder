"""Evaluation runner: scores the agent's final answer against ground truth.

Deliberately lightweight (string containment on the expected title, not an
LLM-as-judge) so the eval suite has no external dependency of its own and
runs deterministically — the thing under test (the agent's model calls) is
already non-deterministic enough.
"""

from __future__ import annotations

from dataclasses import dataclass

from calibre_llm_finder.eval.cases import EVAL_CASES, EvalCase
from calibre_llm_finder.llm.agent import AgentSession, BookFindingAgent


@dataclass
class EvalResult:
    case: EvalCase
    answer: str
    passed: bool


async def run_eval(agent: BookFindingAgent) -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in EVAL_CASES:
        session = AgentSession()
        answer = ""
        async for event in agent.ask(session, case.query):
            if event.type == "done":
                answer = event.text
        passed = case.expected_title.lower() in answer.lower()
        results.append(EvalResult(case=case, answer=answer, passed=passed))
    return results
