"""Shared service-binary whitelist loading for rhl guard/helpers."""

from __future__ import annotations

import importlib.resources
import os
import shlex
from typing import Any

import yaml

_TOOLS_YAML_ENV = "RHL_TOOLS_YAML"


def _default_tools_text() -> str:
    try:
        return importlib.resources.files("ssh_guard").joinpath("tools.yaml").read_text()
    except Exception:
        path = os.path.join(os.path.dirname(__file__), "tools.yaml")
        with open(path, encoding="utf-8") as handle:
            return handle.read()


def load_service_binaries() -> frozenset[str]:
    tools_path = os.environ.get(_TOOLS_YAML_ENV)
    try:
        if tools_path:
            with open(tools_path, encoding="utf-8") as handle:
                tools: Any = yaml.safe_load(handle)
        else:
            tools = yaml.safe_load(_default_tools_text())
    except Exception as exc:
        raise SystemExit(f"rhl: could not load tools.yaml: {exc}") from exc

    if not isinstance(tools, dict) or not tools:
        raise SystemExit("rhl: tools.yaml must contain a non-empty mapping")

    binaries: set[str] = set()
    for name, tool_def in tools.items():
        if not isinstance(tool_def, dict):
            raise SystemExit(f"rhl: malformed tool definition for {name!r}")
        cmd_template = tool_def.get("command_template", "")
        if not isinstance(cmd_template, str):
            raise SystemExit(f"rhl: malformed command_template for {name!r}")
        try:
            parts = shlex.split(cmd_template)
        except ValueError as exc:
            raise SystemExit(f"rhl: malformed command_template for {name!r}: {exc}") from exc
        if not parts:
            raise SystemExit(f"rhl: empty command_template for {name!r}")
        first_token = parts[0].split("{", 1)[0]
        if not first_token:
            raise SystemExit(f"rhl: malformed command_template for {name!r}")
        binaries.add(first_token)
    if not binaries:
        raise SystemExit("rhl: no service binaries found in tools.yaml")
    return frozenset(binaries)
