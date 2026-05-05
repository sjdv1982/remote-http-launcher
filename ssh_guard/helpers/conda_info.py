"""rhl-conda-info -- emit the cached conda setup JSON."""

from __future__ import annotations

import json
import sys

from ssh_guard._cli import die, handle_help
from ssh_guard._state import conda_cache_path

USAGE = "rhl-conda-info"
DESCRIPTION = "Print ~/.remote-http-launcher/conda-setup.json as compact JSON."


def main() -> None:
    args = sys.argv[1:]
    handle_help(args, USAGE, DESCRIPTION)
    if args:
        die("rhl-conda-info", "no arguments are accepted")
    path = conda_cache_path()
    if not path.exists():
        die("rhl-conda-info", f"cache file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        die("rhl-conda-info", f"malformed JSON in {path}: {exc}")
    except OSError as exc:
        die("rhl-conda-info", f"could not read {path}: {exc}")
    if not isinstance(data, dict):
        die("rhl-conda-info", "cache file must contain a JSON object")
    print(json.dumps(data, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
