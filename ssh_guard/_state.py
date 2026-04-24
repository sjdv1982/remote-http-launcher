"""Shared path resolution and key validation for ssh_guard helpers."""

import os
import re
import pathlib

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
                    "/boot", "/sys", "/proc", "/dev", "/run", "/var", "/opt"}


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
