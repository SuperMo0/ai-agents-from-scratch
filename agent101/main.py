import sys
from agent import Agent, ConversationalAgent
from tools import RESEARCH_TOOLS


def run_conversational_repl() -> None:
    """Interactive command-line chat loop backed by ConversationalAgent."""
    agent = ConversationalAgent(tools=RESEARCH_TOOLS)
    print("Conversational research agent. Type 'exit' or 'quit' to stop (Ctrl-D also works).\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        agent.ask(user_message, verbose=True)
        print()


if __name__ == "__main__":
    if "--chat" in sys.argv:
        run_conversational_repl()
    else:
        question = sys.argv[1]
        if not question:
            print('Usage: uv run main.py "<question>"  (or --chat for interactive mode)')
            sys.exit(1)
        agent = Agent(tools=RESEARCH_TOOLS)
        agent.run(question, verbose=True)
