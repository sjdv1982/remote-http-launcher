"""rhl-clear-db <path> — remove seamless.db from the given database directory."""

import sys

from ssh_guard._state import validate_clearable_path


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: rhl-clear-db <path>", file=sys.stderr)
        sys.exit(1)

    path = validate_clearable_path(sys.argv[1])

    if not path.exists():
        print(f"rhl-clear-db: path does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    if not path.is_dir():
        print(f"rhl-clear-db: not a directory: {path}", file=sys.stderr)
        sys.exit(1)

    db = path / "seamless.db"
    if not db.exists():
        print(f"rhl-clear-db: seamless.db not found in {path}")
        return

    try:
        db.unlink()
        print(f"rhl-clear-db: removed {db}")
    except OSError as exc:
        print(f"rhl-clear-db: could not remove {db}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
