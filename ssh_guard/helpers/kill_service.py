"""rhl-kill-service <key> — send SIGHUP to a service and remove its state JSON."""

import json
import os
import signal
import sys

from ssh_guard._state import key_to_server_json, validate_key


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: rhl-kill-service <key>", file=sys.stderr)
        sys.exit(1)

    key = sys.argv[1]
    validate_key(key)

    json_path = key_to_server_json(key)
    if not json_path.exists():
        print(f"rhl-kill-service: no state file for {key!r}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"rhl-kill-service: could not read state: {exc}", file=sys.stderr)
        sys.exit(1)

    pid = data.get("pid")
    if pid is not None:
        try:
            os.kill(int(pid), signal.SIGHUP)
            print(f"rhl-kill-service: sent SIGHUP to PID {pid}")
        except ProcessLookupError:
            print(f"rhl-kill-service: PID {pid} not found (already gone)")
        except PermissionError as exc:
            print(f"rhl-kill-service: could not kill PID {pid}: {exc}", file=sys.stderr)
            sys.exit(1)

    try:
        json_path.unlink(missing_ok=True)
        print(f"rhl-kill-service: removed {json_path}")
    except OSError as exc:
        print(f"rhl-kill-service: could not remove state file: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
