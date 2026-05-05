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
    '/sys', '/proc', '/dev', '/run', '/var', '/opt'.

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

_HELPER_RE = re.compile(r"rhl-[a-z][a-z-]*\Z")


def _parse_and_check(command: str) -> tuple[list[str] | None, str]:
    if not command.strip():
        return None, "empty command (interactive session not allowed)"
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return None, f"command parse error: {exc}"
    if not parts:
        return None, "empty command"
    if not _HELPER_RE.fullmatch(parts[0]):
        return None, f"command not in whitelist: {command[:120]!r}"
    return parts, "rhl helper"


def main() -> None:
    command = os.environ.get("SSH_ORIGINAL_COMMAND", "")
    if not command:
        print(
            "rhl-guard: this program is an SSH guard for remote-http-launcher.\n"
            "It must be invoked via SSH, not run directly.\n\n"
            "To install, add to ~/.ssh/authorized_keys on the remote server:\n"
            '    command="rhl-guard" ssh-rsa AAAA... your-key-comment\n\n'
            "To test a specific command:\n"
            '    SSH_ORIGINAL_COMMAND="rhl-ps" rhl-guard',
            file=sys.stderr,
        )
        sys.exit(1)

    args, reason = _parse_and_check(command)
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
