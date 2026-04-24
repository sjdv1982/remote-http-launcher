"""rhl-rm-state <key> [--client] [--server] — remove launcher state JSON files.

Without flags, removes both server and client JSON files for the key.
"""

import sys

from ssh_guard._state import key_to_client_json, key_to_server_json, validate_key


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0].startswith("-") and args[0] not in ("--client", "--server"):
        print("usage: rhl-rm-state <key> [--client] [--server]", file=sys.stderr)
        sys.exit(1)

    key = args[0]
    validate_key(key)

    flags = set(args[1:])
    unknown = flags - {"--client", "--server"}
    if unknown:
        print(f"rhl-rm-state: unknown flags: {' '.join(unknown)}", file=sys.stderr)
        sys.exit(1)

    do_server = not flags or "--server" in flags
    do_client = not flags or "--client" in flags

    if do_server:
        p = key_to_server_json(key)
        try:
            p.unlink(missing_ok=True)
            print(f"rhl-rm-state: removed server state {p}")
        except OSError as exc:
            print(f"rhl-rm-state: {exc}", file=sys.stderr)
            sys.exit(1)

    if do_client:
        p = key_to_client_json(key)
        try:
            p.unlink(missing_ok=True)
            print(f"rhl-rm-state: removed client state {p}")
        except OSError as exc:
            print(f"rhl-rm-state: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
