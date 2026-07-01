import json
from dataclasses import dataclass
from typing import Any, Callable

from openai.types.chat import ChatCompletionToolUnionParam


@dataclass
class Tool:
    """Wraps a Python function as an agent tool."""
    name: str
    description: str
    parameters: dict[str, Any]
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
