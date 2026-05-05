"""SSH guard for remote-http-launcher.

Install on the remote server by adding to ~/.ssh/authorized_keys:
    command="rhl-guard" ssh-rsa AAAA... key-comment

Every SSH session arriving on that key is validated against a narrow
top-level helper whitelist before the command is exec'd. Interactive sessions
are rejected outright.
"""

from __future__ import annotations

import os
import pathlib
import re
import shlex
import sys

_HELPER_RE = re.compile(r"^rhl-[a-z][a-z-]*$")


def _is_allowed(command: str, _service_binaries=None) -> tuple[bool, str]:
    if not command.strip():
        return False, "empty command (interactive session not allowed)"
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return False, f"command parse error: {exc}"
    if not parts:
        return False, "empty command"
    if _HELPER_RE.match(parts[0]):
        return True, "rhl helper"
    return False, f"command not in whitelist: {command[:120]!r}"


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

    allowed, reason = _is_allowed(command)
    if not allowed:
        print(f"rhl-guard: rejected: {reason}", file=sys.stderr)
        sys.exit(1)

    try:
        args = shlex.split(command)
    except ValueError as exc:
        print(f"rhl-guard: parse error: {exc}", file=sys.stderr)
        sys.exit(1)

    if _HELPER_RE.match(args[0]):
        argv0 = pathlib.Path(sys.argv[0])
        if argv0.is_absolute():
            sibling = argv0.with_name(args[0])
            if sibling.exists():
                os.execv(sibling.as_posix(), args)

    os.execvp(args[0], args)
    print(f"rhl-guard: exec failed: {args[0]}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
