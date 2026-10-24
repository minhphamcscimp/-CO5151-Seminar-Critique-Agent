"""Small helper shared by all agents. Owner: Van."""
from __future__ import annotations

from ..llm import LLMClient
from ..schemas import Config, TraceStep


def call_agent(
    llm: LLMClient,
    cfg: Config,
    agent: str,
    system: str,
    user: str,
    tool_calls: list[dict] | None = None,
) -> tuple[str, TraceStep]:
    """Run one agent step and return (text, trace step with usage)."""
    resp = llm.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=cfg.models[agent],
        temperature=cfg.temperature,
    )
    step = TraceStep(
        agent=agent,
        input=user,
        output=resp.text,
        tool_calls=tool_calls or [],
        usage=resp.usage,
    )
    return resp.text, step
