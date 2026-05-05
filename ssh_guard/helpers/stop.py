"""rhl-stop -- stop one or more launcher-managed service processes."""

from __future__ import annotations

import os
import signal
import sys
import time
import json

from ssh_guard._cli import die, handle_help
from ssh_guard._state import key_to_server_json, pid_is_live, read_json_file, validate_key

USAGE = "rhl-stop <key> [<key>...]"
DESCRIPTION = """Stop services by key without deleting state JSON files.

Missing state is reported on stdout and exits 0. Invalid PID, malformed JSON,
and permission errors exit non-zero."""


def _pid_for_key(key: str) -> tuple[str, int | None]:
    data = read_json_file(key_to_server_json(key), prefix="rhl-stop")
    if data is None:
        return "missing state", None
    pid = data.get("pid")
    if isinstance(pid, bool):
        die("rhl-stop", f"{key}: invalid PID")
    try:
        return "", int(pid)
    except (TypeError, ValueError):
        die("rhl-stop", f"{key}: invalid PID")


def _send(pid: int, sig: signal.Signals) -> str | None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return "already gone"
    except PermissionError:
        return "permission denied"
    return None


def _poll(survivors: dict[str, int], seconds: float = 5.0) -> dict[str, int]:
    deadline = time.time() + seconds
    while time.time() < deadline and survivors:
        gone = [key for key, pid in survivors.items() if not pid_is_live(pid)]
        for key in gone:
            del survivors[key]
        if survivors:
            time.sleep(0.5)
    return survivors


def _mark_stale(key: str) -> None:
    path = key_to_server_json(key)
    data = read_json_file(path, prefix="rhl-stop")
    if data is None:
        return
    data["status"] = "stale"
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(json.dumps(data), encoding="utf-8")
        tmp_path.replace(path)
    except OSError as exc:
        die("rhl-stop", f"could not mark {key} stale: {exc}")


def main() -> None:
    args = sys.argv[1:]
    handle_help(args, USAGE, DESCRIPTION)
    if not args:
        die("rhl-stop", "at least one key is required")
    for key in args:
        validate_key(key)

    outcomes: dict[str, str] = {}
    survivors: dict[str, int] = {}
    for key in args:
        outcome, pid = _pid_for_key(key)
        if outcome:
            outcomes[key] = outcome
        elif pid is not None:
            survivors[key] = pid

    exit_code = 0
    for sig in (signal.SIGINT, signal.SIGTERM):
        current = dict(survivors)
        for key, pid in current.items():
            outcome = _send(pid, sig)
            if outcome == "already gone":
                outcomes[key] = outcome
                survivors.pop(key, None)
            elif outcome == "permission denied":
                outcomes[key] = outcome
                survivors.pop(key, None)
                exit_code = 1
        survivors = _poll(survivors)

    for key, pid in list(survivors.items()):
        outcome = _send(pid, signal.SIGKILL)
        if outcome == "permission denied":
            outcomes[key] = outcome
            exit_code = 1
        else:
            outcomes[key] = "kill required"
        survivors.pop(key, None)

    for key in args:
        outcome = outcomes.get(key, "stopped")
        if outcome in ("stopped", "already gone", "kill required"):
            _mark_stale(key)
        print(f"{key}: {outcomes.get(key, 'stopped')}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
