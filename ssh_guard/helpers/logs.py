"""rhl-logs -- print the launcher server log for a service key."""

from __future__ import annotations

import sys

from ssh_guard._cli import die, handle_help, parse_int_flag
from ssh_guard._state import key_to_server_log, validate_key

USAGE = "rhl-logs <key> [--tail N]"
DESCRIPTION = "Print service logs. During the stale post-mortem window, read logs before rhl-rm removes the state handle."


def main() -> None:
    args = sys.argv[1:]
    handle_help(args, USAGE, DESCRIPTION)
    tail = parse_int_flag(args, "--tail", None)
    if len(args) != 1:
        die("rhl-logs", "exactly one key is required")
    key = args[0]
    validate_key(key)
    path = key_to_server_log(key)
    if not path.exists():
        die("rhl-logs", f"no log file for {key!r}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        die("rhl-logs", str(exc))
    if tail is None:
        sys.stdout.buffer.write(data)
    else:
        if tail < 0:
            die("rhl-logs", "--tail must be >= 0")
        lines = data.splitlines(keepends=True)
        sys.stdout.buffer.write(b"".join(lines[-tail:] if tail else []))


if __name__ == "__main__":
    main()
