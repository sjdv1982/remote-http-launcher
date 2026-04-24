"""rhl-restart-cluster <cluster> — SIGHUP all services for a cluster and clean up.

Globs ~/.remote-http-launcher/server/*-<cluster>-*.json, kills each PID,
and removes the state file.
"""

import json
import os
import signal
import sys

from ssh_guard._state import server_dir, validate_cluster


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: rhl-restart-cluster <cluster>", file=sys.stderr)
        sys.exit(1)

    cluster = sys.argv[1]
    validate_cluster(cluster)

    sdir = server_dir()
    if not sdir.exists():
        print(f"rhl-restart-cluster: no server state directory at {sdir}")
        return

    pattern = f"*-{cluster}-*.json"
    files = sorted(sdir.glob(pattern))
    if not files:
        print(f"rhl-restart-cluster: no services found for cluster {cluster!r}")
        return

    errors = 0
    for state_file in files:
        key = state_file.stem
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"rhl-restart-cluster: could not read {state_file}: {exc}", file=sys.stderr)
            errors += 1
            continue

        pid = data.get("pid")
        if pid is not None:
            try:
                os.kill(int(pid), signal.SIGHUP)
                print(f"rhl-restart-cluster: sent SIGHUP to {key} (PID {pid})")
            except ProcessLookupError:
                print(f"rhl-restart-cluster: {key} PID {pid} already gone")
            except PermissionError as exc:
                print(f"rhl-restart-cluster: could not kill {key}: {exc}", file=sys.stderr)
                errors += 1

        try:
            state_file.unlink(missing_ok=True)
            print(f"rhl-restart-cluster: removed state for {key}")
        except OSError as exc:
            print(f"rhl-restart-cluster: could not remove {state_file}: {exc}", file=sys.stderr)
            errors += 1

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
