"""The conversation loop the harness drives: model -> tools -> model, with a simulated user.

It follows mcp-logistica's ``run_agent`` (same message format, same tool
conversion, same ``ChatModel`` protocol) and adds three things an eval needs:

* an ``ask_user`` tool that the harness answers itself. Calling it is the
  structured clarification signal; a question in plain text is the fallback.
* a deterministic simulated user. A task can list replies; each question gets
  the next one, so "ask, then act on the answer" can be graded.
* the trajectory survives an error. Steps are appended to the caller's trace
  as they happen, so a provider failure mid-loop keeps what already ran.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.types import CallToolResult, TextContent
from wms_mcp.agent.loop import to_openai_tool
from wms_mcp.agent.models import ChatModel, Message, ToolSpec

ASK_USER = "ask_user"
ASK_USER_TOOL: ToolSpec = {
    "type": "function",
    "function": {
        "name": ASK_USER,
        "description": (
            "Ask the warehouse user a clarifying question and wait for the answer. Use it when "
            "a request matches several orders, names no order id, or is otherwise ambiguous. "
            "Include the candidates you found in the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
}
NO_REPLY = "(the user has not answered yet)"


def result_text(result: CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


def outcome_of(result: CallToolResult, text: str) -> str:
    """The server's ``outcome`` field (applied, dry_run, needs_clarification...), or ok/error."""
    if result.isError:
        return "error"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return "ok"
    value = payload.get("outcome") if isinstance(payload, dict) else None
    return value if isinstance(value, str) else "ok"


@dataclass
class AgentStep:
    tool: str
    arguments: dict[str, Any]
    is_error: bool
    outcome: str
    text: str
    latency_ms: float


@dataclass
class Question:
    text: str
    via_tool: bool
    reply: str | None


@dataclass
class Conversation:
    model: str
    prompt_version: str
    user: str
    steps: list[AgentStep] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    final_text: str = ""
    stopped: str = "running"  # answered | asked | max_steps | error

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


async def run_conversation(
    model: ChatModel,
    session: ClientSession,
    conv: Conversation,
    *,
    system_prompt: str,
    user_replies: Sequence[str] = (),
    max_steps: int = 6,
    is_question: Callable[[str], bool],
) -> None:
    """Run until the model answers, asks with no reply left, or ``max_steps`` model calls."""
    listed = await session.list_tools()
    tools = [to_openai_tool(t) for t in listed.tools] + [ASK_USER_TOOL]
    messages: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conv.user},
    ]
    replies = list(user_replies)
    for _ in range(max_steps):
        turn = model.complete(messages, tools)
        messages.append(turn.as_message())
        if not turn.tool_calls:
            text = turn.text or ""
            if replies and is_question(text):
                reply = replies.pop(0)
                conv.questions.append(Question(text, via_tool=False, reply=reply))
                messages.append({"role": "user", "content": reply})
                continue
            conv.final_text = text
            conv.stopped = "answered"
            return
        unanswered: str | None = None
        for call in turn.tool_calls:
            if call.name == ASK_USER:
                question = str(call.arguments.get("question", "")).strip()
                answer: str | None = replies.pop(0) if replies else None
                conv.questions.append(Question(question, via_tool=True, reply=answer))
                if answer is None and unanswered is None:
                    unanswered = question
                content = answer if answer is not None else NO_REPLY
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                continue
            started = time.perf_counter()
            result = await session.call_tool(call.name, call.arguments)
            text = result_text(result)
            conv.steps.append(
                AgentStep(
                    tool=call.name,
                    arguments=call.arguments,
                    is_error=bool(result.isError),
                    outcome=outcome_of(result, text),
                    text=text,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
        if unanswered is not None:
            # The question is what the user sees; the turn ends here.
            conv.final_text = unanswered
            conv.stopped = "asked"
            return
    conv.stopped = "max_steps"
