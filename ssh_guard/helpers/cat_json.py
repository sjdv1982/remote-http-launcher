"""rhl-cat-json <key> — pretty-print the server state JSON for a given key."""

import json
import sys

from ssh_guard._state import key_to_server_json, validate_key


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: rhl-cat-json <key>", file=sys.stderr)
        sys.exit(1)

    key = sys.argv[1]
    validate_key(key)

    json_path = key_to_server_json(key)
    if not json_path.exists():
        print(f"rhl-cat-json: no state file for {key!r}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"rhl-cat-json: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
