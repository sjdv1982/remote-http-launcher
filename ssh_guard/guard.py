"""SSH guard for remote-http-launcher.

Install on the remote server by adding to ~/.ssh/authorized_keys:
    command="rhl-guard" ssh-rsa AAAA... key-comment

Every SSH session arriving on that key is validated against a whitelist
before the command is exec'd. Interactive sessions (empty
SSH_ORIGINAL_COMMAND) are rejected outright.

The guard reads tools.yaml (vendored copy bundled with this package,
overridable via RHL_TOOLS_YAML env var) to derive the set of service
binary names that are allowed inside Python heredoc launch scripts.
"""

import importlib.resources
import os
import pathlib
import re
import shlex
import sys
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Tools-yaml service name extraction
# ---------------------------------------------------------------------------

_DEFAULT_TOOLS_YAML = importlib.resources.files(__package__).joinpath("tools.yaml")

# Fallback in case importlib.resources fails (e.g., direct script execution)
_FALLBACK_TOOLS_YAML = os.path.join(os.path.dirname(__file__), "tools.yaml")

_TOOLS_YAML_ENV = "RHL_TOOLS_YAML"


def _load_service_binaries() -> frozenset:
    tools_path = os.environ.get(_TOOLS_YAML_ENV)
    if tools_path:
        with open(tools_path) as fh:
            tools = yaml.safe_load(fh)
    else:
        try:
            text = _DEFAULT_TOOLS_YAML.read_text()
        except Exception:
            with open(_FALLBACK_TOOLS_YAML) as fh:
                text = fh.read()
        tools = yaml.safe_load(text)

    binaries = set()
    for tool_def in tools.values():
        cmd_template = tool_def.get("command_template", "")
        first_token = cmd_template.split()[0] if cmd_template.split() else ""
        if first_token:
            binaries.add(first_token)
    return frozenset(binaries)


# ---------------------------------------------------------------------------
# Whitelist patterns
# ---------------------------------------------------------------------------

# Optional conda activation prefix: source /path/conda.sh && conda activate env &&
_CONDA_PREFIX_RE = re.compile(
    r"^source \S+ && conda activate \S+ && "
)

# Inner-command patterns (after stripping optional conda prefix)
_INNER_PATTERNS = [
    # conda probe (fallback when no cache)
    re.compile(r"^command -v conda >/dev/null 2>&1$"),
    re.compile(r"^cat ~/\.bashrc$"),
    re.compile(r"^conda info --base$"),
    re.compile(r"^conda env list --json$"),
    # process management
    re.compile(r"^ps -p \d+ -o pid=$"),
    # Python heredoc (launcher scripts)
    re.compile(r"^python3 - <<'__RHL_REMOTE_SCRIPT__'"),
]

_HEREDOC_SENTINEL = "__RHL_REMOTE_SCRIPT__"
_POPEN_MARKER = "subprocess.Popen"
# The launcher writes: command = '<evaluated_command>'
_COMMAND_LINE_RE = re.compile(r"""command\s*=\s*['"]([^'"]+)['"]""")


def _check_python_heredoc(inner: str, service_binaries: frozenset) -> bool:
    """Validate a Python heredoc script body.

    Non-launch scripts (exists, remove, handshake, verify_port,
    conda-cache-read) contain no subprocess.Popen — allow unconditionally.
    Launch scripts embed a service command; validate its binary.
    """
    # Extract script body between sentinels
    start = inner.find(_HEREDOC_SENTINEL)
    if start == -1:
        return False
    body_start = inner.find("\n", start)
    if body_start == -1:
        return False
    end = inner.rfind(_HEREDOC_SENTINEL)
    if end <= body_start:
        # Only one sentinel found — treat as non-launch, allow
        body = inner[body_start:]
    else:
        body = inner[body_start:end]

    if _POPEN_MARKER not in body:
        return True  # non-launch script

    # Launch script: verify embedded service command
    match = _COMMAND_LINE_RE.search(body)
    if not match:
        return False
    command_value = match.group(1)
    binary = command_value.split()[0] if command_value.split() else ""
    return binary in service_binaries


def _strip_conda_prefix(inner: str) -> str:
    m = _CONDA_PREFIX_RE.match(inner)
    if m:
        return inner[m.end():]
    return inner


def _is_allowed_bash_lc(inner: str, service_binaries: frozenset) -> bool:
    stripped = _strip_conda_prefix(inner)
    for pattern in _INNER_PATTERNS:
        if pattern.match(stripped):
            if pattern.pattern.startswith("^python3"):
                return _check_python_heredoc(stripped, service_binaries)
            return True
    return False


def _is_allowed(command: str, service_binaries: frozenset) -> tuple[bool, str]:
    """Return (allowed, reason)."""
    if not command.strip():
        return False, "empty command (interactive session not allowed)"

    # Allow any rhl-* helper command (top-level, not bash-wrapped)
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return False, f"command parse error: {exc}"

    if not parts:
        return False, "empty command"

    if re.match(r"^rhl-[a-z][a-z-]*$", parts[0]):
        return True, "rhl helper"

    # All launcher commands arrive as: bash -lc '<inner>'
    if parts[:2] == ["bash", "-lc"] and len(parts) == 3:
        inner = parts[2]
        if _is_allowed_bash_lc(inner, service_binaries):
            return True, "whitelisted launcher command"
        return False, f"inner command not whitelisted: {inner[:120]!r}"

    return False, f"command not in whitelist: {command[:120]!r}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    command = os.environ.get("SSH_ORIGINAL_COMMAND", "")
    if not command:
        print(
            "rhl-guard: this program is an SSH guard for remote-http-launcher.\n"
            "It must be invoked via SSH, not run directly.\n\n"
            "To install, add to ~/.ssh/authorized_keys on the remote server:\n"
            '    command="rhl-guard" ssh-rsa AAAA... your-key-comment\n\n'
            "To test a specific command:\n"
            '    SSH_ORIGINAL_COMMAND="rhl-ps" rhl-guard',
            file=sys.stderr,
        )
        sys.exit(1)
    service_binaries = _load_service_binaries()
    allowed, reason = _is_allowed(command, service_binaries)

    if not allowed:
        print(f"rhl-guard: rejected: {reason}", file=sys.stderr)
        sys.exit(1)

    try:
        args = shlex.split(command)
    except ValueError as exc:
        print(f"rhl-guard: parse error: {exc}", file=sys.stderr)
        sys.exit(1)

    if re.match(r"^rhl-[a-z][a-z-]*$", args[0]):
        argv0 = pathlib.Path(sys.argv[0])
        if argv0.is_absolute():
            sibling = argv0.with_name(args[0])
            if sibling.exists():
                os.execv(sibling.as_posix(), args)

    os.execvp(args[0], args)
    # execvp only returns on error
    print(f"rhl-guard: exec failed: {args[0]}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
