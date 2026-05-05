"""rhl-inspect -- pretty-print server state JSON."""

from __future__ import annotations

import json
import sys

from ssh_guard._cli import die, handle_help
from ssh_guard._state import key_to_server_json, read_json_file, validate_key

USAGE = "rhl-inspect <key> [--with-mtime]"
DESCRIPTION = "Pretty-print the server-side launcher state JSON for one key."


def main() -> None:
    args = sys.argv[1:]
    handle_help(args, USAGE, DESCRIPTION)
    with_mtime = False
    if "--with-mtime" in args:
        args.remove("--with-mtime")
        with_mtime = True
    if len(args) != 1:
        die("rhl-inspect", "exactly one key is required")
    key = args[0]
    validate_key(key)
    path = key_to_server_json(key)
    data = read_json_file(path, prefix="rhl-inspect")
    if data is None:
        die("rhl-inspect", f"no state file for {key!r}")
    if with_mtime:
        print(json.dumps({"mtime": path.stat().st_mtime, "data": data}, separators=(",", ":")))
        return
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
