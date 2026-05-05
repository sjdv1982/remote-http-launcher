"""Shared path resolution, state loading, and key validation for ssh_guard helpers."""

import pathlib
import json
import os
import re
from typing import Any

_BASE_ENV = "REMOTE_HTTP_LAUNCHER_DIR"
_BASE_DEFAULT = "~/.remote-http-launcher"

_KEY_RE = re.compile(r'^[a-zA-Z0-9._-]+$')

# Keys are cluster, project, stage tokens separated by '-' or '--'.
# Realistically they also contain letters/digits/dots/underscores.
_CLUSTER_RE = re.compile(r'^[a-zA-Z0-9._-]+$')


def base_dir() -> pathlib.Path:
    raw = os.environ.get(_BASE_ENV, _BASE_DEFAULT)
    return pathlib.Path(raw).expanduser()


def server_dir() -> pathlib.Path:
    return base_dir() / "server"


def client_dir() -> pathlib.Path:
    return base_dir() / "client"


def conda_cache_path() -> pathlib.Path:
    return base_dir() / "conda-setup.json"


def key_to_server_json(key: str) -> pathlib.Path:
    return server_dir() / f"{key}.json"


def key_to_server_log(key: str) -> pathlib.Path:
    return server_dir() / f"{key}.log"


def key_to_client_json(key: str) -> pathlib.Path:
    return client_dir() / f"{key}.json"


def validate_key(key: str) -> None:
    if not _KEY_RE.fullmatch(key):
        raise SystemExit(f"rhl: invalid key {key!r} (only letters, digits, . _ - allowed)")


def validate_cluster(cluster: str) -> None:
    if not _CLUSTER_RE.fullmatch(cluster):
        raise SystemExit(f"rhl: invalid cluster name {cluster!r}")


# System directories that must never be cleared
_FORBIDDEN_ROOTS = {"/", "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64",
                    "/boot", "/sys", "/proc", "/run", "/var", "/opt"}


def validate_clearable_path(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    if not p.is_absolute():
        raise SystemExit(f"rhl: path must be absolute: {path!r}")
    # Resolve to catch any symlink tricks, but only if it exists
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    if str(resolved) in _FORBIDDEN_ROOTS or str(p) in _FORBIDDEN_ROOTS:
        raise SystemExit(f"rhl: refusing to operate on system directory: {path!r}")
    # No '..' components allowed in the raw path
    if ".." in p.parts:
        raise SystemExit(f"rhl: path must not contain '..': {path!r}")
    return p


def validate_workdir(path: str) -> pathlib.Path:
    p = pathlib.Path(path).expanduser()
    if not p.is_absolute():
        raise SystemExit(f"rhl: workdir must be absolute: {path!r}")
    if ".." in pathlib.Path(path).parts:
        raise SystemExit(f"rhl: workdir must not contain '..': {path!r}")
    try:
        resolved = p.resolve(strict=False)
    except OSError:
        resolved = p
    if str(resolved) in _FORBIDDEN_ROOTS or str(p) in _FORBIDDEN_ROOTS:
        raise SystemExit(f"rhl: refusing to use system directory as workdir: {path!r}")
    return p


def read_json_file(path: pathlib.Path, *, prefix: str = "rhl") -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{prefix}: malformed JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise SystemExit(f"{prefix}: could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{prefix}: state file must contain a JSON object: {path}")
    return data


def iter_state_files(directory: pathlib.Path) -> list[pathlib.Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def pid_is_live(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _int_pid(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def server_row(path: pathlib.Path, *, check_pid: bool = True) -> dict[str, Any]:
    data = read_json_file(path, prefix="rhl-ps")
    if data is None:
        data = {}
    row = dict(data)
    row["key"] = path.stem
    json_status = data.get("status")
    if json_status is not None:
        row["json_status"] = json_status

    pid = _int_pid(data.get("pid"))
    live: bool | None = None
    status = json_status if isinstance(json_status, str) else "failed"
    if status == "starting" and pid is None:
        live = None
    elif status == "failed" or data.get("dry_run"):
        live = None
    elif pid is not None:
        live = pid_is_live(pid) if check_pid else None
        if status == "running" and check_pid:
            status = "running" if live else "stale"
    elif status == "running":
        status = "stale" if check_pid else "running"
    row["status"] = status
    row["live"] = live
    if "meta" not in row:
        row["meta"] = {}
    return row


def client_row(path: pathlib.Path) -> dict[str, Any]:
    data = read_json_file(path, prefix="rhl-ps")
    if data is None:
        data = {}
    row = dict(data)
    row["key"] = path.stem
    row["host"] = data.get("hostname")
    row["ssh_host"] = (
        data.get("tunneled-host")
        or data.get("ssh_hostname")
        or data.get("hostname")
    )
    row["port"] = data.get("port")
    if "meta" not in row:
        row["meta"] = {}
    return row
