"""rhl-ls-services [--client] — list known service keys.

Default: server-side keys (from ~/.remote-http-launcher/server/).
--client: client-side keys (from ~/.remote-http-launcher/client/).
"""

import sys

from ssh_guard._state import client_dir, server_dir


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] not in ("--client",):
        print("usage: rhl-ls-services [--client]", file=sys.stderr)
        sys.exit(1)

    use_client = "--client" in args
    directory = client_dir() if use_client else server_dir()

    if not directory.exists():
        return  # no services ever launched — silently empty

    keys = sorted(p.stem for p in directory.glob("*.json"))
    for key in keys:
        print(key)


if __name__ == "__main__":
    main()
