"""Tool registry with an allowlist. Owner: Tin.

Every tool is registered with a permission level. Only allowlisted tools can be called,
and only READ_ONLY tools are allowed to run. Other permissions are rejected.
This is the code-level guarantee the report (Part 3) points to.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable

from .benchmark_lookup import lookup_benchmark


class Permission(str, Enum):
    READ_ONLY = "read_only"
    WRITE_REVERSIBLE = "write_reversible"
    WRITE_IRREVERSIBLE = "write_irreversible"
    CODE_EXECUTION = "code_execution"


# name -> (function, permission level). Add new tools here and nowhere else.
TOOL_REGISTRY: dict[str, tuple[Callable[..., Any], Permission]] = {
    "lookup_benchmark": (lookup_benchmark, Permission.READ_ONLY),
}

# Only these may be called by agents. Keep it minimal.
ALLOWED_PERMISSIONS = {Permission.READ_ONLY}


def call_tool(tool_name: str, /, **kwargs: Any) -> dict:
    """Call a registered tool, or raise PermissionError.

    tool_name is positional-only so a tool argument may also be called `name`.
    """
    if tool_name not in TOOL_REGISTRY:
        raise PermissionError(f"tool not registered: {tool_name}")
    fn, level = TOOL_REGISTRY[tool_name]
    if level not in ALLOWED_PERMISSIONS:
        raise PermissionError(f"tool {tool_name} needs permission {level.value}, which is blocked")
    return {"tool": tool_name, "args": kwargs, "result": fn(**kwargs)}
