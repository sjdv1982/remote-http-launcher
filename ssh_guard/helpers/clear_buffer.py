"""rhl-clear-buffer <path> — remove all files directly inside a buffer directory.

Validates that path is absolute and not a system root before touching anything.
Removes files only (not subdirectories) directly under the given path, to
avoid accidentally wiping nested project/stage structure.
"""

import sys

from ssh_guard._state import validate_clearable_path


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: rhl-clear-buffer <path>", file=sys.stderr)
        sys.exit(1)

    path = validate_clearable_path(sys.argv[1])

    if not path.exists():
        print(f"rhl-clear-buffer: path does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    if not path.is_dir():
        print(f"rhl-clear-buffer: not a directory: {path}", file=sys.stderr)
        sys.exit(1)

    errors = 0
    removed = 0
    for entry in path.iterdir():
        if entry.is_file() or entry.is_symlink():
            try:
                entry.unlink()
                removed += 1
            except OSError as exc:
                print(f"rhl-clear-buffer: could not remove {entry}: {exc}", file=sys.stderr)
                errors += 1

    print(f"rhl-clear-buffer: removed {removed} file(s) from {path}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
