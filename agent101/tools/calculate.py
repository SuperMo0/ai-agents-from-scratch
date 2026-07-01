from typing import Any

from simpleeval import simple_eval

from tools.tool import Tool

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


def calculate(expression: str) -> str:
    return str(simple_eval(expression))


calculate_tool = Tool(fn=calculate, **CALCULATE_SCHEMA)
