"""rhl-cat-log <key> — print the service log for a given key."""

import sys

from ssh_guard._state import key_to_server_log, validate_key


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: rhl-cat-log <key>", file=sys.stderr)
        sys.exit(1)

    key = sys.argv[1]
    validate_key(key)

    log_path = key_to_server_log(key)
    if not log_path.exists():
        print(f"rhl-cat-log: no log file for {key!r}", file=sys.stderr)
        sys.exit(1)

    try:
        sys.stdout.buffer.write(log_path.read_bytes())
    except OSError as exc:
        print(f"rhl-cat-log: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
