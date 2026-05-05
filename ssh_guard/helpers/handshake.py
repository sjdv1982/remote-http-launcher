"""rhl-handshake -- perform a launcher HTTP handshake request."""

from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request

from ssh_guard._cli import die, handle_help

USAGE = "rhl-handshake URL"
DESCRIPTION = "Fetch URL once and exit 0 for HTTP 2xx, 2 for non-2xx, 1 for errors."


def main() -> None:
    args = sys.argv[1:]
    handle_help(args, USAGE, DESCRIPTION)
    if len(args) != 1:
        die("rhl-handshake", "exactly one URL is required")
    url = args[0]
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        die("rhl-handshake", "URL must use http or https and include a host")
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            status = getattr(response, "status", 200)
            if 200 <= status < 300:
                raise SystemExit(0)
            print(f"HTTP status {status}", file=sys.stderr)
            raise SystemExit(2)
    except urllib.error.HTTPError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
