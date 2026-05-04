"""rhl-rm -- remove launcher state JSON files without touching logs."""

from __future__ import annotations

import sys

from ssh_guard._cli import die, handle_help
from ssh_guard._state import key_to_client_json, key_to_server_json, validate_key

USAGE = "rhl-rm <key> [<key>...] [--client] [--server]"
DESCRIPTION = """Remove client and/or server state JSON files.

For non-persistent services, removing server JSON discards the key-to-log handle
used by rhl-logs. Read logs before rhl-rm when using the stale post-mortem
window. Log files remain on disk and are normally overwritten on the next launch
with the same key."""


def main() -> None:
    args = sys.argv[1:]
    handle_help(args, USAGE, DESCRIPTION)
    flags = {arg for arg in args if arg.startswith("--")}
    unknown = flags - {"--client", "--server"}
    if unknown:
        die("rhl-rm", f"unknown flag(s): {' '.join(sorted(unknown))}")
    keys = [arg for arg in args if not arg.startswith("--")]
    if not keys:
        die("rhl-rm", "at least one key is required")
    for key in keys:
        validate_key(key)
    do_client = not flags or "--client" in flags
    do_server = not flags or "--server" in flags
    for key in keys:
        removed = False
        paths = []
        if do_client:
            paths.append(key_to_client_json(key))
        if do_server:
            paths.append(key_to_server_json(key))
        for path in paths:
            if not path.exists():
                continue
            try:
                path.unlink()
            except OSError as exc:
                die("rhl-rm", f"could not remove {path}: {exc}")
            print(f"removed {path}")
            removed = True
        if not removed:
            print(f"{key}: not found")


if __name__ == "__main__":
    main()
