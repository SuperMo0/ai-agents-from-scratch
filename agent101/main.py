import datetime
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any, Literal, cast

import requests
from openai import OpenAI
from openai.types.chat import ChatCompletionToolUnionParam, ChatCompletionMessageParam, ChatCompletionMessage
from dotenv import load_dotenv
from pydantic import BaseModel
from simpleeval import simple_eval


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

RESULTS_DIR = Path(__file__).parent / "results"
SUB_CALL_MODEL = "gpt-4o-mini"

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """
You are a research agent. Today's real date is {current_date}. Use tools to
accomplish the task, then provide a final answer.
<scratchpad>
{scratchpad_content}
</scratchpad>
RESEARCH NOTES PROTOCOL: The scratchpad above holds working notes for this run only
(e.g. a fact you found via search_web or summarise_content), not permanent memory.
- `remember_fact(fact)` appends a note so you don't lose track of a finding or
  re-search for it later in this same run. Never overwrites existing notes, so
  call it once per finding, not with a big rewritten blob.
- `forget_fact(fact_id)` removes a note by the id shown in the scratchpad once
  it's no longer needed.
Don't note transient status ("searching now") or anything already visible above.
If you're unsure whether a claim you plan to state is accurate, use `check_fact`
to verify it against the context you've gathered before including it in your answer.
Your training data has a knowledge cutoff and may be stale. For any question
involving "current", "latest", or otherwise time-sensitive facts (software
versions, office-holders, prices, etc.), do not inject a specific date, version,
or name from your own memory into a search_web query — search neutrally (e.g.
"latest stable Python version" not "Python version 2023") and trust the search
results over your prior knowledge. Call get_current_date first if you need
today's date to interpret "current" or to compute an age/duration.
"""

# --------------------------------------------------------------------------
# Tool schemas (name, description, JSON-schema parameters)
# --------------------------------------------------------------------------

REMEMBER_FACT_SCHEMA: dict[str, Any] = dict(
    name="remember_fact",
    description="Append a working research note to the scratchpad for this run. Never overwrites prior notes.",
    parameters={
        "type": "object",
        "properties": {
            "fact": {"type": "string", "description": "One concise, self-contained note to remember."}
        },
        "required": ["fact"],
    },
)

FORGET_FACT_SCHEMA: dict[str, Any] = dict(
    name="forget_fact",
    description="Remove a previously saved research note by its id, e.g. once it's no longer needed.",
    parameters={
        "type": "object",
        "properties": {
            "fact_id": {"type": "integer", "description": "The id shown in brackets before the note in the scratchpad."}
        },
        "required": ["fact_id"],
    },
)

SEARCH_WEB_SCHEMA: dict[str, Any] = dict(
    name="search_web",
    description="Search the web for a query and return a summary of top results with their sources.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."}
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)

GET_CURRENT_DATE_SCHEMA: dict[str, Any] = dict(
    name="get_current_date",
    description="Returns today's date in ISO format (YYYY-MM-DD).",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
)

CALCULATE_SCHEMA: dict[str, Any] = dict(
    name="calculate",
    description="Evaluates a safe arithmetic expression, e.g. '5000 * (1.08 ** 15)', and returns the result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "The arithmetic expression to evaluate."}
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
)

SUMMARISE_CONTENT_SCHEMA: dict[str, Any] = dict(
    name="summarise_content",
    description=(
        "Distils a long piece of content down to the key facts relevant to a given "
        "focus, via a second LLM call. Use this on raw search_web results before "
        "reasoning over them."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The content to summarise."},
            "focus": {"type": "string", "description": "What to focus the summary on."},
        },
        "required": ["content", "focus"],
        "additionalProperties": False,
    },
)

CHECK_FACT_SCHEMA: dict[str, Any] = dict(
    name="check_fact",
    description=(
        "Checks whether a specific claim is supported by the given context, via a "
        "second LLM call. Returns 'supported', 'unsupported', or 'uncertain' with a "
        "one-sentence explanation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "claim": {"type": "string", "description": "The factual claim to check."},
            "context": {"type": "string", "description": "The context to check the claim against."},
        },
        "required": ["claim", "context"],
        "additionalProperties": False,
    },
)


class CheckFactResult(BaseModel):
    verdict: Literal["supported", "unsupported", "uncertain"]
    explanation: str


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------

def search_web(query: str) -> str:
    if not TAVILY_API_KEY:
        return (
            f"[MOCK — no TAVILY_API_KEY configured] Plausible static result for "
            f"'{query}': no real search was performed."
        )
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
        },
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        return f"No results found for '{query}'."
    return "\n".join(
        f"- {r.get('title', '')}: {r.get('content', '')} (source: {r.get('url', '')})"
        for r in results
    )


def get_current_date() -> str:
    return datetime.date.today().isoformat()


def calculate(expression: str) -> str:
    return str(simple_eval(expression))


def summarise_content(content: str, focus: str) -> str:
    response = client.chat.completions.create(
        model=SUB_CALL_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Distil the given content down to the key facts relevant to the "
                    "requested focus. Be concise: a few sentences, facts only, no preamble."
                ),
            },
            {"role": "user", "content": f"Focus: {focus}\n\nContent:\n{content}"},
        ],
    )
    return response.choices[0].message.content or ""


def check_fact(claim: str, context: str) -> str:
    response = client.chat.completions.parse(
        model=SUB_CALL_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict fact-checker. Decide whether the claim is "
                    "supported by the given context alone, and give a one-sentence "
                    "explanation."
                ),
            },
            {"role": "user", "content": f"Claim: {claim}\n\nContext:\n{context}"},
        ],
        response_format=CheckFactResult,
    )
    result = response.choices[0].message.parsed
    return result.model_dump_json() if result else "{}"


@dataclass
class Tool:
    """Wraps a Python function as an agent tool."""
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def to_openai_schema(self) -> ChatCompletionToolUnionParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def execute(self, arguments_json: str) -> str:
        args = json.loads(arguments_json)
        try:
            result = self.fn(**args)
            return str(result)
        except Exception as e:
            return f"Tool error: {e}"


RESEARCH_TOOLS = [
    Tool(fn=search_web, **SEARCH_WEB_SCHEMA),
    Tool(fn=get_current_date, **GET_CURRENT_DATE_SCHEMA),
    Tool(fn=calculate, **CALCULATE_SCHEMA),
    Tool(fn=summarise_content, **SUMMARISE_CONTENT_SCHEMA),
    Tool(fn=check_fact, **CHECK_FACT_SCHEMA),
]


# --------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------

_ARG_TRUNCATE_LEN = 50


def _format_call_args(arguments_json: str) -> str:
    try:
        args = json.loads(arguments_json)
    except json.JSONDecodeError:
        return arguments_json
    parts = []
    for key, value in args.items():
        text = str(value)
        if len(text) > _ARG_TRUNCATE_LEN:
            parts.append(f"{key}=...")
        elif isinstance(value, str):
            parts.append(f'{key}="{value}"')
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


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
        max_tokens_per_run: int = 30_000,
        timeout_s: float = 60.0,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens_per_run
        self.timeout_s = timeout_s

        self.scratchpad: dict[int, str] = {}
        self._next_fact_id = 1
        self.last_state: AgentState | None = None

        self.tools = [
            *(tools or []),
            self._remember_fact_tool(),
            self._forget_fact_tool(),
        ]
        self._tool_map = {t.name: t for t in self.tools}
        self._openai_schemas: list[ChatCompletionToolUnionParam] = [t.to_openai_schema() for t in self.tools]

    # ----------------------------------------------------------------
    # Tool factories
    # ----------------------------------------------------------------

    def _remember_fact_tool(self) -> Tool:
        def remember_fact(fact: str) -> str:
            fid = self._next_fact_id
            self.scratchpad[fid] = fact
            self._next_fact_id += 1
            return f"Saved as note #{fid}."
        return Tool(fn=remember_fact, **REMEMBER_FACT_SCHEMA)

    def _forget_fact_tool(self) -> Tool:
        def forget_fact(fact_id: int) -> str:
            if fact_id in self.scratchpad:
                del self.scratchpad[fact_id]
                return f"Removed note #{fact_id}."
            return f"No note with id #{fact_id}."
        return Tool(fn=forget_fact, **FORGET_FACT_SCHEMA)

    def _render_system_prompt(self) -> str:
        content = "\n".join(f"[{fid}] {fact}" for fid, fact in self.scratchpad.items()) or "No notes recorded yet."
        return self.system_prompt.format(
            scratchpad_content=content,
            current_date=datetime.date.today().isoformat(),
        )

    # ----------------------------------------------------------------
    # Run loop
    # ----------------------------------------------------------------

    def run(self, user_message: str, history: list[ChatCompletionMessageParam] | None = None, verbose: bool = False) -> str:
        state = AgentState()
        self.last_state = state
        state.messages = list(history) if history else [{"role": "system", "content": self._render_system_prompt()}]
        state.messages.append({"role": "user", "content": user_message})

        start = time.monotonic()
        while True:
            abort_reason = self._budget_exceeded(state, start)
            if abort_reason:
                return self._abort(state, abort_reason, verbose)

            msg = self._call_model(state)

            if not msg.tool_calls:
                return self._finish(state, msg, history, verbose)

            self._handle_tool_calls(state, msg.tool_calls, verbose)
            state.messages[0]["content"] = self._render_system_prompt()

    def _budget_exceeded(self, state: AgentState, start: float) -> str | None:
        if state.iterations >= self.max_iterations:
            return "max iterations reached"
        if state.total_tokens >= self.max_tokens:
            return "token budget exceeded"
        if time.monotonic() - start > self.timeout_s:
            return "timeout"
        return None

    def _call_model(self, state: AgentState) -> ChatCompletionMessage:
        response = client.chat.completions.create(
            model=self.model,
            messages=state.messages,
            tools=self._openai_schemas or [],
        )
        state.iterations += 1
        if response.usage is not None:
            state.total_tokens += response.usage.total_tokens

        msg = response.choices[0].message
        state.messages.append(cast(ChatCompletionMessageParam, msg.model_dump(exclude_none=True)))
        return msg

    def _finish(self, state: AgentState, msg: ChatCompletionMessage, history: list[ChatCompletionMessageParam] | None, verbose: bool) -> str:
        state.finished = True
        answer = msg.content or ""

        if verbose:
            print(f"[Iter {state.iterations}] Final answer (no tool call)")
            print(f"Answer: {answer}")
            print(f"Tokens: {state.total_tokens:,} | Iterations: {state.iterations}")

        if history is not None:
            history.clear()
            history.extend(state.messages)

        return answer

    def _handle_tool_calls(self, state: AgentState, tool_calls, verbose: bool) -> None:
        for tool_call in tool_calls:
            if tool_call.type != "function":
                continue

            tool_name = tool_call.function.name
            tool = self._tool_map.get(tool_name)
            result = tool.execute(tool_call.function.arguments) if tool else f"Unknown tool: {tool_name}"

            if verbose:
                formatted_args = _format_call_args(tool_call.function.arguments)
                print(f"[Iter {state.iterations}] Tool: {tool_name}({formatted_args}) → {len(result)} chars")

            state.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    def _abort(self, state: AgentState, reason: str, verbose: bool) -> str:
        state.error = reason
        try:
            r = client.chat.completions.create(
                model=self.model,
                messages=state.messages + [{
                    "role": "user",
                    "content": f"Stopping early: {reason}. Summarise what you found so far.",
                }],
            )
            partial = r.choices[0].message.content or ""
        except Exception:
            partial = "No partial result available."

        if verbose:
            print(f"[WARNING] {reason} — returning partial answer.")
            print(f"Answer: {partial}")
            print(f"Tokens: {state.total_tokens:,} | Iterations: {state.iterations}")

        return f"[{reason}] {partial}"


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

def save_result(question_number: int, question: str, answer: str, iterations: int, tokens: int) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"question_{question_number}.md"
    path.write_text(
        f"# Question {question_number}\n\n"
        f"**Question:** {question}\n\n"
        f"**Answer:**\n\n{answer}\n\n"
        f"---\nIterations: {iterations} | Tokens: {tokens}\n"
    )
    return path


# --------------------------------------------------------------------------
# Test questions
# --------------------------------------------------------------------------

TEST_QUESTIONS = [
    "What is today's date, and what day of the week is it?",
    "If a $5,000 investment grows at 8% per year, how much will it be worth in 15 years? Show the calculation.",
    "Who founded OpenAI, and in what year? What is the current CEO's name?",
    "What is the current version of Python, and when was it released?",
]


if __name__ == "__main__":
    agent = Agent(tools=RESEARCH_TOOLS)

    for i, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n--- Question {i} ---")
        answer = agent.run(question, verbose=True)
        state = agent.last_state
        assert state is not None
        path = save_result(i, question, answer, state.iterations, state.total_tokens)
        print(f"[Saved] {path}")

    print("\n--- Verifying abort path (max_iterations=2 on Question 3) ---")
    limited_agent = Agent(tools=RESEARCH_TOOLS, max_iterations=2)
    limited_agent.run(TEST_QUESTIONS[2], verbose=True)
