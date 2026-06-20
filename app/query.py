import os
import json
import logging
from datetime import date
from dotenv import load_dotenv
from litellm import completion

from app.db import query_tasks, semantic_search_entries
from app.classify import generate_query_embedding

load_dotenv()
logger = logging.getLogger(__name__)

# Swap this single string to change provider/model — LiteLLM handles the rest.
# Examples: "gemini/gemini-2.5-flash", "claude-sonnet-4-6", "gpt-4o"
QUERY_MODEL = os.getenv("QUERY_MODEL", "gemini/gemini-2.5-flash")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_tasks_structured",
            "description": (
                "Query the user's tasks table with exact filters. Use this for questions "
                "about due dates, task status (open/done), or 'what's due today/this week/overdue'. "
                "Dates must be in YYYY-MM-DD format."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["open", "done"],
                        "description": "Filter by task status. Omit to get all statuses.",
                    },
                    "due_before": {
                        "type": "string",
                        "description": "YYYY-MM-DD — return tasks due on or before this date",
                    },
                    "due_after": {
                        "type": "string",
                        "description": "YYYY-MM-DD — return tasks due on or after this date",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes_semantic",
            "description": (
                "Semantic search over all saved notes/entries (tasks, ideas, work notes, etc.) "
                "using meaning-based similarity, not exact keywords. Use this for fuzzy/topical "
                "questions like 'find my notes about X' or 'what did I think about Y'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The topic or concept to search for",
                    }
                },
                "required": ["search_query"],
            },
        },
    },
]


def execute_tool(tool_name: str, tool_input: dict) -> list:
    """Dispatches a tool call to the actual DB function."""
    if tool_name == "query_tasks_structured":
        return query_tasks(
            status=tool_input.get("status"),
            due_before=tool_input.get("due_before"),
            due_after=tool_input.get("due_after"),
        )
    elif tool_name == "search_notes_semantic":
        embedding = generate_query_embedding(tool_input["search_query"])
        return semantic_search_entries(embedding)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


def answer_query(user_question: str) -> str:
    """Main entry point: takes a natural language question, routes to the right
    data source(s) via LLM tool use (provider-agnostic via LiteLLM), and returns
    a synthesized natural language answer."""

    system_prompt = (
        f"You are the query interface for the user's personal second-brain system. "
        f"Today's date is {date.today().isoformat()}. "
        f"Use the available tools to fetch real data before answering — never guess or "
        f"make up tasks/notes. If a tool returns no results, say so honestly. "
        f"Keep answers concise and conversational, suitable for a Telegram message."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]

    # Allow up to a few tool-use rounds in case the model needs multiple lookups
    for _ in range(3):
        response = completion(
            model=QUERY_MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=1000,
        )

        choice = response.choices[0]
        message = choice.message

        if not message.tool_calls:
            return (message.content or "").strip() or "I couldn't find an answer to that."

        # Append assistant turn (with tool calls) to history
        messages.append(message.model_dump())

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_input = json.loads(tool_call.function.arguments)
                result_data = execute_tool(tool_name, tool_input)
                content = json.dumps(result_data, default=str)
            except Exception as e:
                logger.error(f"Tool execution failed: {tool_name}: {e}")
                content = f"Error: {e}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": content,
                }
            )

    return "I had trouble finding a complete answer — try rephrasing the question."