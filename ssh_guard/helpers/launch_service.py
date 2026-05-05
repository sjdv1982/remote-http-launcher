"""rhl-launch-service -- launch a whitelisted remote-http-launcher service."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
from typing import Any

from ssh_guard._state import (
    conda_cache_path,
    key_to_server_json,
    key_to_server_log,
    server_dir,
    validate_key,
    validate_workdir,
)
from ssh_guard._tools import load_service_binaries


def _die(message: str) -> None:
    print(f"rhl-launch-service: {message}", file=sys.stderr)
    raise SystemExit(1)


def _json_object(raw: str | None, label: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _die(f"{label} must be valid JSON: {exc}")
    if not isinstance(data, dict):
        _die(f"{label} must be a JSON object")
    return data


def _is_relative_to(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _expand_service_arg(arg: str) -> str:
    if arg == "~" or arg.startswith("~/"):
        return pathlib.Path(arg).expanduser().as_posix()
    return arg


def _normalize_status_file_args(args: list[str]) -> list[str]:
    normalized = [_expand_service_arg(arg) for arg in args]
    root = server_dir().expanduser()
    root_resolved = root.resolve(strict=False)
    index = 0
    while index < len(normalized):
        token = normalized[index]
        value: str | None = None
        if token == "--status-file":
            if index + 1 >= len(normalized):
                _die("--status-file requires a value")
            value = normalized[index + 1]
            value_index = index + 1
            index += 2
        elif token.startswith("--status-file="):
            value = token.split("=", 1)[1]
            value_index = index
            index += 1
        else:
            index += 1
        if value is None:
            continue
        status_path = pathlib.Path(value).expanduser()
        if not status_path.is_absolute():
            _die("--status-file must be absolute")
        resolved = status_path.resolve(strict=False)
        if not _is_relative_to(resolved, root_resolved):
            _die("--status-file must live under the launcher server directory")
        expanded = status_path.as_posix()
        if token == "--status-file":
            normalized[value_index] = expanded
        else:
            normalized[value_index] = f"--status-file={expanded}"
    return normalized


def _read_conda_cache() -> dict[str, Any]:
    path = conda_cache_path()
    if not path.exists():
        _die(f"conda cache not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _die(f"malformed conda cache: {exc}")
    except OSError as exc:
        _die(f"could not read conda cache: {exc}")
    if not isinstance(data, dict):
        _die("conda cache must contain a JSON object")
    return data


def _spawn(
    argv: list[str],
    *,
    cwd: pathlib.Path,
    stdout_handle: Any,
    conda_env: str | None,
) -> subprocess.Popen:
    if conda_env is None:
        return subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    if not conda_env.strip() or "\n" in conda_env:
        _die("conda environment name is invalid")
    cache = _read_conda_cache()
    conda_source = cache.get("conda_source")
    if conda_source is not None and not isinstance(conda_source, str):
        _die("conda_source in cache must be a string or null")
    prefix = ""
    if conda_source:
        prefix = f"source {shlex.quote(conda_source)} && "
    command = f"{prefix}conda activate {shlex.quote(conda_env)} && exec {shlex.join(argv)}"
    return subprocess.Popen(
        ["bash", "-lc", command],
        cwd=cwd,
        stdout=stdout_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _write_state(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rhl-launch-service",
        description="Launch a whitelisted service and write launcher state.",
    )
    parser.add_argument("--key", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--conda-env")
    parser.add_argument("--network-interface")
    parser.add_argument("--parameters")
    parser.add_argument("--meta")
    parser.add_argument("service_argv", nargs=argparse.REMAINDER)
    ns = parser.parse_args()

    service_argv = list(ns.service_argv)
    if service_argv and service_argv[0] == "--":
        service_argv = service_argv[1:]
    if not service_argv:
        _die("service binary is required after --")

    validate_key(ns.key)
    workdir = validate_workdir(ns.workdir)
    parameters = _json_object(ns.parameters, "--parameters")
    meta = _json_object(ns.meta, "--meta")

    binary = service_argv[0]
    if binary not in load_service_binaries():
        _die(f"service binary is not whitelisted: {binary!r}")
    service_argv[1:] = _normalize_status_file_args(service_argv[1:])

    srv_dir = server_dir()
    srv_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    json_path = key_to_server_json(ns.key)
    log_path = key_to_server_log(ns.key)

    try:
        stdout_handle = log_path.open("wb", buffering=0)
    except OSError as exc:
        _die(f"could not open log file: {exc}")

    try:
        proc = _spawn(
            service_argv,
            cwd=workdir,
            stdout_handle=stdout_handle,
            conda_env=ns.conda_env,
        )
        data: dict[str, Any] = {
            "workdir": ns.workdir,
            "log": log_path.as_posix(),
            "command": shlex.join(service_argv),
            "uid": os.getuid(),
            "pid": proc.pid,
            "status": "starting",
        }
        if ns.network_interface is not None:
            data["network_interface"] = ns.network_interface
        if parameters is not None:
            data["parameters"] = parameters
        if meta is not None:
            data["meta"] = meta
        _write_state(json_path, data)
    except Exception as exc:
        _die(str(exc))
    finally:
        stdout_handle.close()


if __name__ == "__main__":
    main()
