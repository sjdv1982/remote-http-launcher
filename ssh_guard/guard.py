"""SSH guard for remote-http-launcher.

Install on the remote server by adding to ~/.ssh/authorized_keys:

    command="rhl-guard <path-policy-flags>" ssh-rsa AAAA... key-comment

Every SSH session arriving on that key is validated against a narrow
top-level helper whitelist before the command is exec'd. Interactive
sessions (no SSH_ORIGINAL_COMMAND) are rejected outright.

Path policy
===========

Three of the rhl-* helpers accept filesystem paths chosen by the SSH
client:

    rhl-clear <path>
    rhl-ps-persistent <path> [...]
    rhl-launch-service --workdir <path> ...

Without an explicit policy these helpers would have to trust the client
about which directories they may walk, wipe, or use as a working
directory. To bound that surface, rhl-guard requires the operator to
declare a path policy via one of the following mutually-exclusive
flags in the authorized_keys command= directive.

--data-roots PATH
    Strictest mode; recommended for production. PATH points at a
    text file listing the absolute (or '~'-prefixed) directories that
    helpers are permitted to touch. One path per line; blank lines and
    lines beginning with '#' are ignored. A helper-supplied path is
    accepted only if, after '~' expansion and symlink resolution, it
    equals or is contained in one of the listed roots. The same list
    governs rhl-clear, rhl-ps-persistent, and rhl-launch-service
    --workdir, so there is no back door.

--clear-policy seamless
    Marker-based mode for Seamless deployments. rhl-clear and
    rhl-ps-persistent accept a target directory only if it contains
    either a 'seamless.db' file or a '.HASHSERVER_PREFIX' file. No
    allowlist file is required; the launcher's own data directories
    are self-identifying. rhl-launch-service --workdir falls back to
    the always-on heuristics below, since a workdir that has not yet
    been created cannot contain a marker.

--clear-policy marker:NAME
    Marker-based mode, generic. NAME is the simple filename (no path
    separators) that must be present as a regular file in any
    directory passed to rhl-clear or rhl-ps-persistent. Drop the
    marker once into each directory you wish to authorise (e.g.
    'touch ~/my-buffers/.rhl-clearable'). rhl-launch-service --workdir
    falls back to the always-on heuristics, as with the seamless
    preset.

--permissive-paths
    Disables the policy. Helpers accept any path that passes the
    always-on heuristics below. This flag exists so that deployments
    which cannot easily configure either of the modes above can keep
    working; it is not a recommended default. It MUST NOT be combined
    with --data-roots or --clear-policy; doing so is a fatal
    configuration error.

Refusal behaviour
-----------------

If none of the four flags above is present, rhl-clear,
rhl-ps-persistent, and rhl-launch-service --workdir refuse to run with
an error message pointing the operator at this docstring. The other
rhl-* helpers (rhl-ps, rhl-rm, rhl-stop, rhl-logs, rhl-inspect,
rhl-pid-alive, rhl-handshake, rhl-verify-port, rhl-cache-conda,
rhl-conda-info) are unaffected -- they do not accept client-chosen
paths.

Always-on heuristics
====================

Regardless of which path policy is in effect (including
--permissive-paths), every helper-supplied path is rejected if any of
the following is true:

  * The path is not absolute after '~' expansion.
  * The path equals $HOME, or is a strict ancestor of $HOME (e.g.
    '/', '/home', '/home/user' when $HOME is '/home/user/sub').
  * Any segment of the path begins with '.' (e.g. '~/.ssh',
    '/tmp/.cache/foo'). Dotfile-prefixed segments are conventionally
    out-of-band metadata and are never legitimate launcher workdirs.
  * The path resolves to one of the system-root directories: '/',
    '/etc', '/usr', '/bin', '/sbin', '/lib', '/lib64', '/boot',
    '/sys', '/proc', '/run', '/var', '/opt'.

In addition, rhl-clear skips any direct child whose name begins with
'.' while iterating, even when the parent directory itself was
accepted by the policy. This protects dotfile children of a permitted
parent (e.g. a stray '.ssh' inside a configured data root).

Examples
========

Strictest -- explicit allowlist file:

    command="rhl-guard --data-roots /home/svc/.config/rhl/data-roots"
        ssh-rsa AAAA... key-comment

Seamless preset (no allowlist file):

    command="rhl-guard --clear-policy seamless"
        ssh-rsa AAAA... key-comment

Generic marker for non-Seamless workflows:

    command="rhl-guard --clear-policy marker:.rhl-clearable"
        ssh-rsa AAAA... key-comment

Loosest mode, kept for compatibility (not recommended):

    command="rhl-guard --permissive-paths"
        ssh-rsa AAAA... key-comment

Direct invocation
=================

Running rhl-guard from a normal shell (no SSH_ORIGINAL_COMMAND set)
prints an installation-oriented error and exits non-zero -- the guard
is intended to run only as the SSH 'command=' target. To smoke-test a
single guarded command on the server itself:

    SSH_ORIGINAL_COMMAND="rhl-ps" rhl-guard
"""

from __future__ import annotations

import os
import pathlib
import re
import shlex
import sys
from dataclasses import dataclass

_HELPER_RE = re.compile(r"rhl-[a-z][a-z-]*\Z")
_VARSET_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_PATH_HELPERS = {"rhl-clear", "rhl-ps-persistent", "rhl-launch-service"}
_FORBIDDEN_ROOTS = {
    "/",
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/boot",
    "/sys",
    "/proc",
    "/run",
    "/var",
    "/opt",
}
_POLICY_HELP = (
    "configure rhl-guard with one path-policy flag: --data-roots PATH, "
    "--clear-policy seamless, --clear-policy marker:NAME, or --permissive-paths; "
    "see the rhl-guard docstring / README Path policy section"
)


@dataclass(frozen=True)
class _PathPolicy:
    mode: str
    roots: tuple[pathlib.Path, ...] = ()
    marker_names: tuple[str, ...] = ()


def _is_relative_to(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve(path: pathlib.Path) -> pathlib.Path:
    try:
        return path.resolve(strict=False)
    except OSError:
        return path


def _has_dot_segment(path: pathlib.Path) -> bool:
    return any(part not in (path.anchor, os.sep) and part.startswith(".") for part in path.parts)


def _simple_filename(name: str) -> bool:
    return name not in ("", ".", "..") and "/" not in name and "\\" not in name


def _read_data_roots(path: str) -> tuple[pathlib.Path, ...]:
    roots_path = pathlib.Path(path).expanduser()
    try:
        lines = roots_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"could not read --data-roots file {path!r}: {exc}") from exc

    roots: list[pathlib.Path] = []
    for lineno, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        root = pathlib.Path(raw).expanduser()
        if not root.is_absolute():
            raise ValueError(f"--data-roots entry on line {lineno} is not absolute after ~ expansion: {raw!r}")
        roots.append(_resolve(root))
    if not roots:
        raise ValueError("--data-roots file does not contain any usable roots")
    return tuple(roots)


def _parse_policy_args(argv: list[str]) -> tuple[_PathPolicy | None, str | None]:
    data_roots: str | None = None
    clear_policy: str | None = None
    permissive = False
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--permissive-paths":
            permissive = True
            index += 1
        elif arg == "--data-roots":
            if index + 1 >= len(argv):
                return None, "--data-roots requires a path"
            data_roots = argv[index + 1]
            index += 2
        elif arg.startswith("--data-roots="):
            data_roots = arg.split("=", 1)[1]
            index += 1
        elif arg == "--clear-policy":
            if index + 1 >= len(argv):
                return None, "--clear-policy requires a value"
            clear_policy = argv[index + 1]
            index += 2
        elif arg.startswith("--clear-policy="):
            clear_policy = arg.split("=", 1)[1]
            index += 1
        else:
            return None, f"unknown rhl-guard option: {arg}"

    selected = sum(value is not None for value in (data_roots, clear_policy)) + int(permissive)
    if selected == 0:
        return None, None
    if selected > 1:
        return None, "--data-roots, --clear-policy, and --permissive-paths are mutually exclusive"

    if permissive:
        return _PathPolicy("permissive"), None
    if data_roots is not None:
        try:
            roots = _read_data_roots(data_roots)
        except ValueError as exc:
            return None, str(exc)
        return _PathPolicy("data_roots", roots=roots), None
    assert clear_policy is not None
    if clear_policy == "seamless":
        return _PathPolicy("marker", marker_names=("seamless.db", ".HASHSERVER_PREFIX")), None
    if clear_policy.startswith("marker:"):
        marker = clear_policy.split(":", 1)[1]
        if not _simple_filename(marker):
            return None, "--clear-policy marker:NAME requires a simple marker filename"
        return _PathPolicy("marker", marker_names=(marker,)), None
    return None, "--clear-policy must be 'seamless' or 'marker:NAME'"


def _extract_guarded_paths(args: list[str]) -> tuple[list[tuple[str, pathlib.Path]], str | None]:
    helper = args[0]
    values: list[tuple[str, pathlib.Path]] = []
    if helper == "rhl-clear":
        if len(args) >= 2:
            values.append(("clear", pathlib.Path(args[1]).expanduser()))
        return values, None
    if helper == "rhl-launch-service":
        index = 1
        while index < len(args):
            arg = args[index]
            if arg == "--":
                break
            if arg == "--workdir":
                if index + 1 >= len(args):
                    return values, "rhl-launch-service --workdir requires a path"
                values.append(("workdir", pathlib.Path(args[index + 1]).expanduser()))
                index += 2
            elif arg.startswith("--workdir="):
                values.append(("workdir", pathlib.Path(arg.split("=", 1)[1]).expanduser()))
                index += 1
            else:
                index += 1
        return values, None
    if helper == "rhl-ps-persistent":
        index = 1
        while index < len(args):
            arg = args[index]
            if arg in ("--json",):
                index += 1
            elif arg in ("--level", "--file", "--marker"):
                if index + 1 >= len(args):
                    return values, f"rhl-ps-persistent {arg} requires a value"
                index += 2
            elif arg.startswith("--level=") or arg.startswith("--file=") or arg.startswith("--marker="):
                index += 1
            elif arg.startswith("-"):
                index += 1
            else:
                values.append(("clear", pathlib.Path(arg).expanduser()))
                index += 1
        return values, None
    return values, None


def _check_heuristics(path: pathlib.Path) -> str | None:
    if not path.is_absolute():
        return f"path must be absolute after ~ expansion: {path}"
    resolved = _resolve(path)
    home = _resolve(pathlib.Path.home())
    if resolved == home or _is_relative_to(home, resolved):
        return f"path must not be $HOME or an ancestor of $HOME: {path}"
    if _has_dot_segment(path) or _has_dot_segment(resolved):
        return f"path must not contain dot-prefixed segments: {path}"
    if resolved.as_posix() in _FORBIDDEN_ROOTS:
        return f"refusing system-root directory: {path}"
    return None


def _check_path_policy(args: list[str], policy: _PathPolicy | None) -> str | None:
    if args[0] not in _PATH_HELPERS:
        return None
    if policy is None:
        return _POLICY_HELP
    guarded_paths, error = _extract_guarded_paths(args)
    if error is not None:
        return error
    for kind, path in guarded_paths:
        error = _check_heuristics(path)
        if error is not None:
            return error
        resolved = _resolve(path)
        if policy.mode == "data_roots":
            if not any(resolved == root or _is_relative_to(resolved, root) for root in policy.roots):
                return f"path is outside configured --data-roots: {path}"
        elif policy.mode == "marker" and kind == "clear":
            if not any((resolved / marker).is_file() for marker in policy.marker_names):
                markers = ", ".join(policy.marker_names)
                return f"path does not contain required marker file ({markers}): {path}"
    return None


def _parse_and_check(command: str, policy: _PathPolicy | None = None) -> tuple[list[str] | None, str]:
    if not command.strip():
        return None, "empty command (interactive session not allowed)"
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return None, f"command parse error: {exc}"
    if not parts:
        return None, "empty command"
    # Strip leading VAR=value assignments so that clients may prepend PATH=...
    # to cover conda-base installs without breaking the guard whitelist check.
    # The assignments are dropped — they are not passed to exec.
    while parts and _VARSET_RE.match(parts[0]):
        parts = parts[1:]
    if not parts:
        return None, "empty command after stripping variable assignments"
    if not _HELPER_RE.fullmatch(parts[0]):
        return None, f"command not in whitelist: {command[:120]!r}"
    policy_error = _check_path_policy(parts, policy)
    if policy_error is not None:
        return None, policy_error
    return parts, "rhl helper"


def main() -> None:
    policy, error = _parse_policy_args(sys.argv[1:])
    if error is not None:
        print(f"rhl-guard: configuration error: {error}", file=sys.stderr)
        sys.exit(1)

    command = os.environ.get("SSH_ORIGINAL_COMMAND", "")
    if not command:
        print(
            "rhl-guard: this program is an SSH guard for remote-http-launcher.\n"
            "It must be invoked via SSH, not run directly.\n\n"
            "To install, add to ~/.ssh/authorized_keys on the remote server:\n"
            '    command="rhl-guard --data-roots /path/to/data-roots" ssh-rsa AAAA... your-key-comment\n\n'
            "To test a specific command:\n"
            '    SSH_ORIGINAL_COMMAND="rhl-ps" rhl-guard --permissive-paths',
            file=sys.stderr,
        )
        sys.exit(1)

    args, reason = _parse_and_check(command, policy)
    if args is None:
        print(f"rhl-guard: rejected: {reason}", file=sys.stderr)
        sys.exit(1)

    argv0 = pathlib.Path(sys.argv[0])
    if argv0.is_absolute():
        sibling = argv0.with_name(args[0])
        if sibling.exists():
            os.execv(sibling.as_posix(), args)

    os.execvp(args[0], args)
    print(f"rhl-guard: exec failed: {args[0]}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
