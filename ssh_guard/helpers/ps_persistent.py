"""rhl-ps-persistent -- report filesystem-backed persistent service state."""

from __future__ import annotations

import datetime as _dt
import pathlib
import sys

from ssh_guard._cli import die, emit_ndjson, handle_help, parse_int_flag, print_table
from ssh_guard._state import validate_clearable_path

USAGE = "rhl-ps-persistent <path> [<path>...] [--level N] [--file FILENAME] [--json]"
DESCRIPTION = "Walk absolute, validated paths and report absent, empty, or populated state."


def _parse(args: list[str]) -> tuple[list[pathlib.Path], int, str | None, bool]:
    handle_help(args, USAGE, DESCRIPTION)
    json_mode = False
    filename = None
    if "--json" in args:
        args.remove("--json")
        json_mode = True
    level = parse_int_flag(args, "--level", 0)
    if level is None:
        level = 0
    if level < 0:
        die("rhl-ps-persistent", "--level must be >= 0")
    if "--file" in args:
        index = args.index("--file")
        try:
            filename = args[index + 1]
        except IndexError:
            die("rhl-ps-persistent", "--file requires a filename")
        if "/" in filename or filename in ("", ".", ".."):
            die("rhl-ps-persistent", "--file requires a simple filename")
        del args[index:index + 2]
    if not args:
        die("rhl-ps-persistent", "at least one path is required")
    return [validate_clearable_path(arg) for arg in args], level, filename, json_mode


def _mtime(path: pathlib.Path) -> str | None:
    try:
        return _dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return None


def _row(path: pathlib.Path, filename: str | None) -> dict:
    if not path.exists():
        return {"path": path.as_posix(), "state": "absent", "size": None, "modified": None}
    if not path.is_dir():
        die("rhl-ps-persistent", f"not a directory: {path}")
    target = path / filename if filename else path
    if filename:
        populated = target.exists()
    else:
        try:
            populated = any(path.iterdir())
        except OSError as exc:
            die("rhl-ps-persistent", f"could not read {path}: {exc}")
    if not populated:
        return {"path": path.as_posix(), "state": "empty", "size": None, "modified": None}
    if filename:
        try:
            stat = target.stat()
        except OSError as exc:
            die("rhl-ps-persistent", f"could not stat {target}: {exc}")
        return {
            "path": path.as_posix(),
            "state": "populated",
            "size": stat.st_size,
            "modified": _mtime(target),
        }
    return {"path": path.as_posix(), "state": "populated", "size": None, "modified": _mtime(path)}


def _walk(root: pathlib.Path, level: int, filename: str | None) -> list[dict]:
    rows = [_row(root, filename)]
    if level == 0 or not root.exists():
        return rows
    if not root.is_dir():
        die("rhl-ps-persistent", f"not a directory: {root}")
    try:
        children = sorted(child for child in root.iterdir() if child.is_dir())
    except OSError as exc:
        die("rhl-ps-persistent", f"could not read {root}: {exc}")
    for child in children:
        rows.extend(_walk(child, level - 1, filename))
    return rows


def main() -> None:
    roots, level, filename, json_mode = _parse(sys.argv[1:])
    rows: list[dict] = []
    for root in roots:
        rows.extend(_walk(root, level, filename))
    if json_mode:
        emit_ndjson(rows)
    else:
        print_table(("PATH", "STATE", "SIZE", "MODIFIED"), ((r["path"], r["state"], r["size"], r["modified"]) for r in rows))


if __name__ == "__main__":
    main()
