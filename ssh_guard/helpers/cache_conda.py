"""rhl-cache-conda — discover conda setup and write a local cache file.

Run this once on the remote server (directly or via SSH through the guard)
to prime ~/.remote-http-launcher/conda-setup.json.  The launcher reads that
file instead of issuing individual conda probe SSH commands.

Re-run whenever the conda environment changes.
"""

import json
import os
import pathlib
import re
import subprocess
import sys

from ssh_guard._cli import handle_help
from ssh_guard._state import conda_cache_path

_CONDA_PATH_RE = re.compile(r"""[\"']([^\"']+/etc/profile\.d/conda\.sh)[\"']""")
_CONDA_INIT_BLOCK_RE = re.compile(
    r"# >>> conda initialize >>>(.*?)# <<< conda initialize <<<",
    re.DOTALL,
)


def _find_conda_source() -> str | None:
    """Return path to conda.sh, or None if conda is already on PATH."""
    result = subprocess.run(
        ["bash", "-lc", "command -v conda >/dev/null 2>&1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        return None  # conda is on PATH

    # Try to find it in ~/.bashrc
    bashrc = pathlib.Path("~/.bashrc").expanduser()
    if not bashrc.exists():
        return None
    text = bashrc.read_text(encoding="utf-8", errors="replace")
    block_match = _CONDA_INIT_BLOCK_RE.search(text)
    if not block_match:
        return None
    path_match = _CONDA_PATH_RE.search(block_match.group(1))
    if not path_match:
        return None
    return path_match.group(1)


def _run_conda(conda_source: str | None, *args: str) -> subprocess.CompletedProcess:
    if conda_source:
        cmd = f"source {conda_source} && conda {' '.join(args)}"
        return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    return subprocess.run(["conda", *args], capture_output=True, text=True)


def main() -> None:
    handle_help(sys.argv[1:], "rhl-cache-conda", __doc__ or "")
    conda_source = _find_conda_source()
    if conda_source is None:
        # conda is on PATH — verify it's really there
        check = subprocess.run(
            ["bash", "-lc", "command -v conda >/dev/null 2>&1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if check.returncode != 0:
            print("rhl-cache-conda: conda not found on PATH and no conda.sh in ~/.bashrc",
                  file=sys.stderr)
            sys.exit(1)

    base_result = _run_conda(conda_source, "info", "--base")
    if base_result.returncode != 0:
        print(f"rhl-cache-conda: 'conda info --base' failed:\n{base_result.stderr.strip()}",
              file=sys.stderr)
        sys.exit(1)
    conda_base = base_result.stdout.strip()
    if not conda_base:
        print("rhl-cache-conda: 'conda info --base' returned empty output", file=sys.stderr)
        sys.exit(1)

    env_result = _run_conda(conda_source, "env", "list", "--json")
    if env_result.returncode != 0:
        print(f"rhl-cache-conda: 'conda env list --json' failed:\n{env_result.stderr.strip()}",
              file=sys.stderr)
        sys.exit(1)
    try:
        envs = json.loads(env_result.stdout).get("envs", [])
    except json.JSONDecodeError as exc:
        print(f"rhl-cache-conda: could not parse conda env list output: {exc}", file=sys.stderr)
        sys.exit(1)

    cache = {"conda_source": conda_source, "conda_base": conda_base, "envs": envs}
    cache_path = conda_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print(f"rhl-cache-conda: wrote {cache_path}")


if __name__ == "__main__":
    main()
