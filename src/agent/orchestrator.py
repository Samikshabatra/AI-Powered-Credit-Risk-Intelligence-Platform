"""The orchestrator agent: one chat surface over the whole platform.

A hand-rolled AI tool-use loop - no agent framework. LangChain or similar
would add ~80 MB to the image and an abstraction layer over what is, honestly,
a while-loop around `messages.create`. The loop is 40 lines and every step of it
is inspectable, which matters more here than the convenience.

    user question
        -> the model picks a tool (query_data / predict_risk / explain_prediction /
           get_policy_rules / list_applicants / get_model_performance)
        -> the tool runs against the real module
        -> the result goes back into the conversation
        -> repeat until the model answers in prose (bounded by MAX_TOOL_TURNS)

Every turn records which tools ran, so the UI can show the routing rather than
asking the user to trust it. And every tool the agent can call is also wired to
a button elsewhere in the UI: if the router misfires during a demo, the
capability is still one click away.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.agent.tools import TOOL_SCHEMAS, dispatch
from src.utils.config import settings
from src.utils.llm import LLMUnavailable, get_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_TOOL_TURNS = 5
MAX_HISTORY_TURNS = 12

SYSTEM_PROMPT = """You are the credit-risk analyst assistant for a bank. You sit
in front of a deployed decision system: a LightGBM default-risk model with
isotonic calibration, SHAP explanations, a cost-optimised decision threshold, a
derived rule set, and a SQL warehouse of 307,511 loan applications.

ROUTING
- Population questions - rates, averages, counts, breakdowns, rankings, "how
  many", "which group" -> query_data.
- One named applicant, "score", "risk", "would we approve" -> predict_risk.
- "Why", "explain", "what drove it", "reasons" for an applicant -> explain_prediction.
- Policy, criteria, "what rules" -> get_policy_rules.
- "How accurate", "how good", "AUC", "calibration", "threshold" -> get_model_performance.
- No applicant id given but one is needed -> list_applicants first, then proceed.
- Combine tools when the question needs it: score an applicant, then explain it.

ANSWERING
- Answer from tool results only. Never invent a number, a rule or a reason.
- If a tool returns an error, say plainly what failed and what would fix it.
- Quote figures as returned: rates as percentages, money with separators.
- Be brief - a credit analyst wants the number and the reason, not an essay.
- When you report a decision, give the probability, the band and the action.
- Never present a protected attribute (gender, age, marital status) as a reason
  for declining an applicant, even if the underlying data shows a correlation.
  You may discuss such correlations as portfolio statistics when asked directly.

You have no knowledge of this portfolio beyond what the tools return. If a
question cannot be answered with them, say so."""


@dataclass
class AgentTurn:
    """One question-and-answer exchange, with the routing it took."""

    question: str
    answer: str
    tools_used: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    error: str | None = None

    @property
    def tool_names(self) -> list[str]:
        return [call["name"] for call in self.tools_used]


class CreditRiskAgent:
    """The AI model with the platform's six tools bound to it, plus memory."""

    def __init__(self, max_tool_turns: int = MAX_TOOL_TURNS) -> None:
        self.client = get_client()  # raises LLMUnavailable without an API key
        self.max_tool_turns = max_tool_turns
        self.messages: list[dict[str, Any]] = []
        self.turns: list[AgentTurn] = []
        logger.info("Orchestrator ready with %d tools", len(TOOL_SCHEMAS))

    def reset(self) -> None:
        self.messages.clear()
        self.turns.clear()

    def _trim_history(self) -> None:
        """Keep the conversation bounded; tool results are the bulky part."""
        if len(self.messages) > MAX_HISTORY_TURNS * 2:
            self.messages = self.messages[-MAX_HISTORY_TURNS * 2:]

    # ------------------------------------------------------------------ #
    def chat(self, question: str) -> AgentTurn:
        """Run one exchange to completion, executing tools as the model asks."""
        started = time.perf_counter()
        turn = AgentTurn(question=question, answer="")
        self.messages.append({"role": "user", "content": question})

        try:
            for _ in range(self.max_tool_turns):
                response = self.client.complete(
                    system=SYSTEM_PROMPT,
                    messages=self.messages,
                    model=settings.sql_model,
                    tools=TOOL_SCHEMAS,
                    max_tokens=1500,
                    cache_system=True,
                    label="orchestrator",
                )
                turn.input_tokens += response.usage.input_tokens
                turn.output_tokens += response.usage.output_tokens
                turn.cache_read_tokens += response.usage.cache_read_tokens

                if not response.tool_calls:
                    turn.answer = response.text or "(no answer produced)"
                    self.messages.append({"role": "assistant", "content": turn.answer})
                    break

                # Record the assistant's tool-use turn verbatim, then answer every
                # tool call it made in a single user message, as the API requires.
                self.messages.append({"role": "assistant", "content": response.raw.content})
                tool_results = []
                for call in response.tool_calls:
                    result = dispatch(call["name"], call["input"])
                    turn.tools_used.append({
                        "name": call["name"],
                        "input": call["input"],
                        "ok": "error" not in result,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": _stringify(result),
                        "is_error": "error" in result,
                    })
                self.messages.append({"role": "user", "content": tool_results})
            else:
                turn.answer = (
                    "I could not finish that request within the tool-call budget. "
                    "Try asking for one thing at a time."
                )
                turn.error = "max_tool_turns_exceeded"

        except Exception as exc:
            logger.exception("Agent turn failed")
            turn.answer = f"Something went wrong handling that request: {exc}"
            turn.error = str(exc)

        turn.elapsed_ms = (time.perf_counter() - started) * 1000
        self._trim_history()
        self.turns.append(turn)
        logger.info("Agent turn: tools=%s | %.0fms",
                    turn.tool_names or ["none"], turn.elapsed_ms)
        return turn


def _stringify(payload: dict[str, Any]) -> str:
    """Tool results go back to the model as compact JSON."""
    import json

    return json.dumps(payload, default=str, separators=(",", ":"))


# --------------------------------------------------------------------------- #
_AGENT: CreditRiskAgent | None = None


def get_orchestrator(reset: bool = False) -> CreditRiskAgent:
    """Shared agent, so conversation memory survives Streamlit re-runs."""
    global _AGENT
    if _AGENT is None or reset:
        _AGENT = CreditRiskAgent()
    return _AGENT


def chat(question: str) -> AgentTurn:
    return get_orchestrator().chat(question)


def chat_safely(question: str) -> AgentTurn:
    """Degrades to an explanatory turn when no API key is configured."""
    try:
        return chat(question)
    except LLMUnavailable as exc:
        return AgentTurn(
            question=question,
            answer=(
                f"The AI assistant needs an API key to run. {exc}\n\n"
                "Every capability it fronts is still available from the other tabs: "
                "Risk Prediction, Explainability, Business Rules and EDA."
            ),
            error="llm_unavailable",
        )
