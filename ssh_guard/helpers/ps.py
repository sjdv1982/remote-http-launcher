"""rhl-ps -- list remote-http-launcher process state."""

from __future__ import annotations

import sys

from ssh_guard._cli import die, emit_ndjson, handle_help, print_table
from ssh_guard._state import client_dir, client_row, iter_state_files, server_dir, server_row

USAGE = "rhl-ps [--client] [--host SSH_HOST] [--status STATE] [--key] [--no-status] [--json]"
DESCRIPTION = """List known remote-http-launcher state rows.

Default mode reads server state. Use --client for client connection files.
--no-status skips server PID liveness probing."""


def main() -> None:
    args = sys.argv[1:]
    handle_help(args, USAGE, DESCRIPTION)
    use_client = False
    json_mode = False
    key_mode = False
    no_status = False
    host_filter = None
    status_filter = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--client":
            use_client = True
        elif arg == "--json":
            json_mode = True
        elif arg == "--key":
            key_mode = True
        elif arg == "--no-status":
            no_status = True
        elif arg == "--host":
            i += 1
            if i >= len(args):
                die("rhl-ps", "--host requires a value")
            host_filter = args[i]
        elif arg == "--status":
            i += 1
            if i >= len(args):
                die("rhl-ps", "--status requires a value")
            status_filter = args[i]
        else:
            die("rhl-ps", f"unknown argument: {arg}")
        i += 1

    if host_filter and not use_client:
        die("rhl-ps", "--host requires --client")
    if key_mode and json_mode:
        die("rhl-ps", "--key and --json are mutually exclusive")

    directory = client_dir() if use_client else server_dir()
    rows = [
        client_row(path) if use_client else server_row(path, check_pid=not no_status)
        for path in iter_state_files(directory)
    ]
    if host_filter:
        rows = [row for row in rows if row.get("ssh_host") == host_filter]
    if status_filter:
        rows = [row for row in rows if row.get("status") == status_filter]

    if key_mode:
        for row in rows:
            print(row["key"])
    elif json_mode:
        emit_ndjson(rows)
    elif use_client:
        print_table(
            ("KEY", "SSH-HOST", "HOST", "PORT"),
            ((row.get("key"), row.get("ssh_host"), row.get("host"), row.get("port")) for row in rows),
        )
    else:
        print_table(
            ("KEY", "STATUS", "PORT"),
            ((row.get("key"), row.get("status"), row.get("port")) for row in rows),
        )


if __name__ == "__main__":
    main()
