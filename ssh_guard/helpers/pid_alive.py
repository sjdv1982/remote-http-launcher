"""rhl-pid-alive -- test whether a PID is live."""

from __future__ import annotations

import sys

from ssh_guard._cli import die, handle_help
from ssh_guard._state import pid_is_live

USAGE = "rhl-pid-alive PID"
DESCRIPTION = "Exit 0 if PID is live, 1 if it is not live."


def _parse_pid(raw: str) -> int:
    if raw.lower() in {"true", "false"}:
        die("rhl-pid-alive", "PID must be a positive integer")
    try:
        pid = int(raw)
    except ValueError:
        die("rhl-pid-alive", "PID must be a positive integer")
    if pid <= 0:
        die("rhl-pid-alive", "PID must be a positive integer")
    return pid


def main() -> None:
    args = sys.argv[1:]
    handle_help(args, USAGE, DESCRIPTION)
    if len(args) != 1:
        die("rhl-pid-alive", "exactly one PID is required")
    raise SystemExit(0 if pid_is_live(_parse_pid(args[0])) else 1)


if __name__ == "__main__":
    main()
