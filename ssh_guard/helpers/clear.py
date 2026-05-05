"""rhl-clear -- remove direct children of a validated directory."""

from __future__ import annotations

import shutil
import sys

from ssh_guard._cli import die, handle_help
from ssh_guard._state import validate_clearable_path

USAGE = "rhl-clear <path>"
DESCRIPTION = "Remove all direct files, symlinks, and directories below an absolute validated directory."


def main() -> None:
    args = sys.argv[1:]
    handle_help(args, USAGE, DESCRIPTION)
    if len(args) != 1:
        die("rhl-clear", "exactly one path is required")
    path = validate_clearable_path(args[0])
    if not path.exists():
        die("rhl-clear", f"path does not exist: {path}")
    if not path.is_dir():
        die("rhl-clear", f"not a directory: {path}")
    removed = 0
    errors = 0
    for child in path.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed += 1
        except OSError as exc:
            print(f"rhl-clear: could not remove {child}: {exc}", file=sys.stderr)
            errors += 1
    print(f"removed {removed} item(s) from {path}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
