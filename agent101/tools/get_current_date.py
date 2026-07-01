import datetime
from typing import Any

from tools.tool import Tool

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


def get_current_date() -> str:
    return datetime.date.today().isoformat()


get_current_date_tool = Tool(fn=get_current_date, **GET_CURRENT_DATE_SCHEMA)
