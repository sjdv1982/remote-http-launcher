"""Small dependency-free CLI helpers for rhl-* commands."""

from __future__ import annotations

import json
import sys
from typing import Iterable, Sequence


def handle_help(args: Sequence[str], usage: str, description: str) -> None:
    if "-h" in args or "--help" in args:
        print(f"usage: {usage}")
        print()
        print(description)
        raise SystemExit(0)


def die(prefix: str, message: str, code: int = 1) -> None:
    print(f"{prefix}: {message}", file=sys.stderr)
    raise SystemExit(code)


def parse_int_flag(args: list[str], flag: str, default: int | None = None) -> int | None:
    if flag not in args:
        return default
    index = args.index(flag)
    try:
        raw = args[index + 1]
    except IndexError:
        die("rhl", f"{flag} requires an integer")
    try:
        value = int(raw)
    except ValueError:
        die("rhl", f"{flag} requires an integer")
    del args[index:index + 2]
    return value


def print_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    rendered = [[_display(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print(" ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    for row in rendered:
        print(" ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def emit_ndjson(rows: Iterable[dict]) -> None:
    for row in rows:
        print(json.dumps(row, sort_keys=True))


def _display(value: object) -> str:
    if value is None or value == "":
        return "-"
    return str(value)
