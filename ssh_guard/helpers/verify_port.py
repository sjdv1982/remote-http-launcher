"""rhl-verify-port -- verify that a TCP host:port accepts connections."""

from __future__ import annotations

import ipaddress
import re
import socket
import sys
import time

from ssh_guard._cli import die, handle_help

USAGE = "rhl-verify-port HOST PORT"
DESCRIPTION = "Connect to HOST PORT, retrying three times before failing."

_HOST_RE = re.compile(r"^[A-Za-z0-9.-]+$")


def _validate_host(host: str) -> str:
    if not host or len(host) > 253:
        die("rhl-verify-port", "invalid host")
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    if not _HOST_RE.fullmatch(host):
        die("rhl-verify-port", "invalid host")
    labels = host.rstrip(".").split(".")
    if any(not label or label.startswith("-") or label.endswith("-") for label in labels):
        die("rhl-verify-port", "invalid host")
    return host


def _validate_port(raw: str) -> int:
    try:
        port = int(raw)
    except ValueError:
        die("rhl-verify-port", "port must be an integer")
    if not 1 <= port <= 65535:
        die("rhl-verify-port", "port must be in [1, 65535]")
    return port


def main() -> None:
    args = sys.argv[1:]
    handle_help(args, USAGE, DESCRIPTION)
    if len(args) != 2:
        die("rhl-verify-port", "HOST and PORT are required")
    host = _validate_host(args[0])
    port = _validate_port(args[1])
    last_exc: BaseException | None = None
    for trial in range(3):
        sock = socket.socket()
        sock.settimeout(2.0)
        try:
            sock.connect((host, port))
            raise SystemExit(0)
        except Exception as exc:
            last_exc = exc
            if trial < 2:
                time.sleep(2)
        finally:
            sock.close()
    print(str(last_exc), file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
