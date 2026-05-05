"""rhl-clear -- remove direct children of a validated directory."""

from __future__ import annotations

import sys
from pathlib import Path

from ssh_guard._cli import die, handle_help
from ssh_guard._state import validate_clearable_path

USAGE = "rhl-clear <path>"
DESCRIPTION = "Remove all direct files, symlinks, and directories below an absolute validated directory."


def _remove_tree_skipping_dot_dirs(path: Path) -> bool:
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            if child.name.startswith("."):
                continue
            _remove_tree_skipping_dot_dirs(child)
            try:
                child.rmdir()
            except OSError:
                if not _only_dot_dirs_remain(child):
                    raise
        else:
            child.unlink()
    try:
        path.rmdir()
    except OSError:
        if not _only_dot_dirs_remain(path):
            raise
        return False
    return True


def _only_dot_dirs_remain(path: Path) -> bool:
    return all(child.is_dir() and not child.is_symlink() and child.name.startswith(".") for child in path.iterdir())


def main() -> None:
    args = sys.argv[1:]
    handle_help(args, USAGE, DESCRIPTION)
    if len(args) != 1:
        die("rhl-clear", "exactly one path is required")
    path = validate_clearable_path(str(Path(args[0]).expanduser()))
    if not path.exists():
        die("rhl-clear", f"path does not exist: {path}")
    if not path.is_dir():
        die("rhl-clear", f"not a directory: {path}")
    removed = 0
    errors = 0
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink() and child.name.startswith("."):
            continue
        try:
            if child.is_dir() and not child.is_symlink():
                fully_removed = _remove_tree_skipping_dot_dirs(child)
                if not fully_removed:
                    continue
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
