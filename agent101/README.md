# agent101

A from-scratch research agent (no LangChain/LangGraph/PydanticAI) built
directly on the OpenAI API. It searches the web, then runs its draft answer
through an automatic evaluator (prompt coverage, source triangulation,
temporal validity, conflict detection) before it's allowed to finish —
otherwise it's told what's missing and keeps researching.

## Structure

- `main.py` — entrypoint: `--chat` for interactive mode, or pass a question as an argument
- `agent.py` — `Agent` / `ConversationalAgent`, system prompts, the research evaluator
- `clients.py` — OpenAI/Tavily clients and shared config
- `tools/` — one file per tool (`search_web`, `get_current_date`, `calculate`)

## Running

```sh
uv run main.py "What are the top 5 promising technologies in 2026?"
uv run main.py --chat
```

Requires a `.env` with `OPENAI_API_KEY` and `TAVILY_API_KEY`.
