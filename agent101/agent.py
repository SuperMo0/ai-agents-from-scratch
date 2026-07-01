import datetime
import time
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam, ChatCompletionToolUnionParam, ChatCompletionMessageToolCallUnion
from pydantic import BaseModel, Field

from clients import RESULTS_DIR, SUB_CALL_MODEL, client
from tools.tool import Tool

DEFAULT_SYSTEM_PROMPT = """
You are a research agent. Today's real date is {current_date}. Use tools to
accomplish the task, then provide a final answer.

Before your answer is accepted, it will be automatically evaluated against a
strict rubric: full coverage of every sub-question in the request, claims
backed by multiple independent authoritative sources (not a single blog),
verified recency for any time-sensitive facts, and explicit acknowledgment of
conflicting information if the search results disagree. If the evaluation
finds gaps, you will be told exactly what's missing and what to search next —
keep researching until it passes.

Your training data has a knowledge cutoff and may be stale. For any question
involving "current", "latest", or otherwise time-sensitive facts (software
versions, office-holders, prices, etc.), do not inject a specific date, version,
or name from your own memory into a search_web query — search neutrally (e.g.
"latest stable Python version" not "Python version 2023") and trust the search
results over your prior knowledge. Call get_current_date first if you need
today's date to interpret "current" or to compute an age/duration.
"""

EVALUATOR_SYSTEM_PROMPT = """
You are a strict research evaluator. Today's date is {current_date}. Judge the
agent's draft answer against the original request and the research gathered,
using these five pillars:
- Prompt Coverage: did it answer every sub-question, or latch onto the easy
  part and ignore the rest?
- Source Triangulation: is each claim backed by multiple independent,
  authoritative sources, or just one?
- Temporal Validity: for facts that change (prices, versions, officeholders),
  was recency verified against today's date?
- Conflict Detection: did the search results disagree, and if so, did the
  draft answer explicitly acknowledge it?
- Actionability: if the criteria aren't met, name the exact next search query
  needed to close the gap.

is_ready_to_finish must be strictly True only if prompt_coverage is 1.0 AND
source_quality is 'triangulated' or 'authoritative' AND temporal_validity is
not 'outdated'.
"""


class ResearchEvaluator(BaseModel):
    prompt_coverage: float = Field(
        ...,
        description="Percentage of the original user prompt addressed by the current findings (0.0 to 1.0).",
    )
    unanswered_aspects: list[str] = Field(
        ...,
        description="Specific requirements from the user prompt that still lack sufficient evidence.",
    )
    source_quality: Literal["unverified", "single_source", "triangulated", "authoritative"] = Field(
        ...,
        description="The strongest level of verification achieved for the core claims.",
    )
    conflicting_information_found: bool = Field(
        ...,
        description="True if different search results contradict each other.",
    )
    temporal_validity: Literal["outdated", "current", "not_applicable"] = Field(
        ...,
        description="Whether time-sensitive claims have been verified as current.",
    )
    is_ready_to_finish: bool = Field(
        ...,
        description="STRICTLY True ONLY IF: prompt_coverage is 1.0 AND source_quality is 'triangulated' or 'authoritative' AND temporal_validity is not 'outdated'.",
    )
    next_search_query: str | None = Field(
        None,
        description="If is_ready_to_finish is False, the exact search query needed to fill the gaps. If True, null.",
    )


@dataclass
class AgentState:
    """Mutable state for one agent run."""
    messages: list[ChatCompletionMessageParam] = field(default_factory=list)
    iterations: int = 0
    total_tokens: int = 0
    elapsed_s: float = 0.0
    finished: bool = False
    error: str | None = None


class Agent:
    """
    Instantiate once; call run() for each new task.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: list[Tool] | None = None,
        max_iterations: int = 10,
        max_tokens_per_run: int = 50_000,
        timeout_s: float = 200,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens_per_run
        self.timeout_s = timeout_s

        self.tools = list(tools or [])
        self._tool_map = {t.name: t for t in self.tools}
        self._openai_schemas: list[ChatCompletionToolUnionParam] = [t.to_openai_schema() for t in self.tools]

    def _render_system_prompt(self) -> str:
        return self.system_prompt.format(current_date=datetime.date.today().isoformat())

    def _complete(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        tools: list[ChatCompletionToolUnionParam] | None = None,
        response_format: type[BaseModel] | None = None,
    ):
        kwargs: dict[str, Any] = {"model": model or self.model, "messages": messages}
        if tools is not None:
            kwargs["tools"] = tools
        if response_format is not None:
            return client.chat.completions.parse(response_format=response_format, **kwargs)
        return client.chat.completions.create(**kwargs)

    def _evaluate_research(self, state: AgentState, user_message: str, draft_answer: str) -> ResearchEvaluator | None:
        transcript = "\n\n".join(
            f"{m.get('role')}: {m.get('content')}" for m in state.messages[1:] if m.get("content")
        )
        response = self._complete(
            model=SUB_CALL_MODEL,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT.format(current_date=datetime.date.today().isoformat())},
                {"role": "user", "content": (
                    f"Original request: {user_message}\n\n"
                    f"Draft answer: {draft_answer}\n\n"
                    f"Research gathered so far:\n{transcript}"
                )},
            ],
            response_format=ResearchEvaluator,
        )
        return cast(ResearchEvaluator | None, response.choices[0].message.parsed)

    def run(self, user_message: str, history: list[ChatCompletionMessageParam] | None = None, verbose: bool = False) -> str:
        state = AgentState()
        state.messages = list(history) if history else []
        system_message: ChatCompletionMessageParam = {"role": "system", "content": self._render_system_prompt()}
        if state.messages and state.messages[0].get("role") == "system":
            state.messages[0] = system_message
        else:
            state.messages.insert(0, system_message)
        state.messages.append({"role": "user", "content": user_message})

        start = time.monotonic()
        while True:
            abort_reason = self._budget_exceeded(state, start)
            if abort_reason:
                return self._abort(state, abort_reason, verbose)

            msg = self._call_model(state)

            if not msg.tool_calls:
                evaluation = self._evaluate_research(state, user_message, msg.content or "")

                if verbose and evaluation:
                    print(
                        f"[Iter {state.iterations}] Evaluation: coverage={evaluation.prompt_coverage:.0%} "
                        f"source_quality={evaluation.source_quality} temporal={evaluation.temporal_validity} "
                        f"ready={evaluation.is_ready_to_finish}"
                    )

                if evaluation is None or evaluation.is_ready_to_finish:
                    return self._finish(state, msg, history, verbose, user_message)

                state.messages.append({
                    "role": "user",
                    "content": (
                        "Evaluation: research is not yet sufficient to finish.\n"
                        f"Unanswered aspects: {', '.join(evaluation.unanswered_aspects) or 'none listed'}\n"
                        f"Source quality: {evaluation.source_quality}\n"
                        f"Temporal validity: {evaluation.temporal_validity}\n"
                        f"Conflicting information found: {evaluation.conflicting_information_found}\n"
                        f"Run this search next: {evaluation.next_search_query}"
                    ),
                })
                continue

            self._handle_tool_calls(state, msg.tool_calls, verbose)

    def _budget_exceeded(self, state: AgentState, start: float) -> str | None:
        if state.iterations >= self.max_iterations:
            return "max iterations reached"
        if state.total_tokens >= self.max_tokens:
            return "token budget exceeded"
        if time.monotonic() - start > self.timeout_s:
            return "timeout"
        return None

    def _call_model(self, state: AgentState) -> ChatCompletionMessage:
        response = self._complete(state.messages, tools=self._openai_schemas or [])
        state.iterations += 1
        if response.usage is not None:
            state.total_tokens += response.usage.total_tokens

        msg = response.choices[0].message
        state.messages.append(cast(ChatCompletionMessageParam, msg.model_dump(exclude_none=True)))
        return msg

    def _slugify_question(self, user_message: str) -> str:
        response = self._complete(
            model=SUB_CALL_MODEL,
            messages=[
                {"role": "system", "content": "Summarize the question as a filename slug: lowercase words joined by hyphens, max 6 words, no punctuation. Reply with only the slug."},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or "research"

    def _finish(self, state: AgentState, msg: ChatCompletionMessage, history: list[ChatCompletionMessageParam] | None, verbose: bool, user_message: str) -> str:
        state.finished = True
        answer = msg.content or ""

        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / f"{self._slugify_question(user_message)}.md").write_text(answer)

        if verbose:
            print(f"[Iter {state.iterations}] Final answer (no tool call)")
            print(f"Answer: {answer}")
            print(f"Tokens: {state.total_tokens:,} | Iterations: {state.iterations}")

        if history is not None:
            history.clear()
            history.extend(state.messages)

        return answer

    def _handle_tool_calls(self, state: AgentState, tool_calls:list[ChatCompletionMessageToolCallUnion], verbose: bool) -> None:
        for tool_call in tool_calls:
            if tool_call.type != "function":
                continue

            tool_name = tool_call.function.name
            tool = self._tool_map.get(tool_name)
            result = tool.execute(tool_call.function.arguments) if tool else f"Unknown tool: {tool_name}"

            if verbose:
                args = tool_call.function.arguments
                args_preview = args if len(args) <= 50 else args[:50] + "..."
                print(f"[Iter {state.iterations}] Tool: {tool_name}({args_preview}) → {len(result)} chars")

            state.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    def _abort(self, state: AgentState, reason: str, verbose: bool) -> str:
        state.error = reason
        try:
            r = self._complete(state.messages + [{
                "role": "user",
                "content": f"Stopping early: {reason}. Summarise what you found so far.",
            }])
            partial = r.choices[0].message.content or ""
        except Exception:
            partial = "No partial result available."

        if verbose:
            print(f"[WARNING] {reason} — returning partial answer.")
            print(f"Answer: {partial}")
            print(f"Tokens: {state.total_tokens:,} | Iterations: {state.iterations}")

        return f"[{reason}] {partial}"


class ConversationalAgent(Agent):
    """
    Extends Agent to maintain message history across sequential calls to ask(),
    so the model sees the full prior conversation on each new turn.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history: list[ChatCompletionMessageParam] = []

    def ask(self, user_message: str, verbose: bool = False) -> str:
        return self.run(user_message, history=self.history, verbose=verbose)
