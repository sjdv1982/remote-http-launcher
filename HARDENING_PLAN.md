# SSH-guard hardening plan

## Injection Vulnerabilities in guard.py

### Critical — Arbitrary Python execution in "non-launch" heredoc scripts

`_check_python_heredoc` uses a single heuristic: if `subprocess.Popen` is absent from the body, it returns `True` unconditionally:

```python
if _POPEN_MARKER not in body:
    return True  # non-launch script
```

Any Python that avoids that string runs freely. An attacker sends:
```
bash -lc $'python3 - <<\'__RHL_REMOTE_SCRIPT__\'\nimport os; os.system("rm -rf ~")\n__RHL_REMOTE_SCRIPT__'
```
No `subprocess.Popen` → allowed. `os.execvp`, `exec()`, `open()`, `__import__` etc. are all equivalent and all pass.

---

### Critical — Code injection alongside a valid service binary

When `subprocess.Popen` *is* present, only the extracted binary name is checked. The rest of the script is completely unvalidated:

```python
# guard sees: binary == 'hashserver' ✓
command = 'hashserver --status-file /tmp/x --timeout 600 /tmp/wd'
import os; os.system("rm -rf ~")   # ← executes, guard doesn't care
subprocess.Popen(...)
```

`_COMMAND_LINE_RE` finds the first `command = '...'` match and approves the binary. Everything else runs.

---

### High — Shell injection via conda prefix path/env name

`_CONDA_PREFIX_RE` uses `\S+` for both the source path and the conda env name, accepting any non-whitespace characters including shell metacharacters:

```
bash -lc 'source /path/$(rm -rf ~)/conda.sh && conda activate env && command -v conda >/dev/null 2>&1'
```

After `_strip_conda_prefix`, the remainder is `command -v conda >/dev/null 2>&1` — a whitelisted pattern. The guard approves it, but bash executes `$(rm -rf ~)` before reaching the allowed command.

---

### Moderate — `cat ~/.bashrc` leaks server secrets

The whitelist admits `bash -lc 'cat ~/.bashrc'`. `~/.bashrc` routinely contains exported environment variables, API tokens, and infrastructure paths. This is already superseded by `rhl-cache-conda` running server-side, so the pattern provides no legitimate benefit while exposing sensitive data.

---

### Root cause and recommendation

Vulnerabilities 1 and 2 stem from the fundamental problem: **the guard is trying to validate code (Python scripts) as data**. Text-based analysis of code bodies has no reliable upper bound — any blocklist/allowlist of Python constructs can be bypassed. The only robust fix for those two is to stop sending code over SSH and instead move all remote actions to server-side `rhl-*` helpers that accept *data* arguments (paths, keys, ports), not code. The conda probe patterns (3 of the 4 `bash -lc` patterns plus `cat ~/.bashrc`) and `ps -p <pid> -o pid=` can similarly be retired once their corresponding `rhl-*` helpers are required (the infrastructure for this already exists: `rhl-cache-conda`, `rhl-ps`, `rhl-stop`, etc.).

## Summary 

Closing the four injection vulnerabilities above identified in `ssh_guard/guard.py`
by retiring all `bash -lc` and `python3 - <<HEREDOC` patterns from the guard
whitelist in favor of dedicated `rhl-*` helpers.

## Status

### Done (this branch)
- `Executor.invoke_helper(argv, ...)` and `Executor._try_helper(argv)` added
  to `remote_http_launcher.py`. `_try_helper` auto-detects whether `rhl-*`
  helpers exist on the target host (per-Executor cache; detects via
  exit-code 127 / `command not found` / `no such file`). Returns `None` when
  helpers are absent so callers can fall back.
- Four `RemoteState` methods now use existing helpers with heredoc fallback:
  - `RemoteState.exists` → `rhl-inspect` (+ `rhl-rm --server` when `dry_run`)
  - `RemoteState.remove` → `rhl-rm --server`
  - `RemoteState.read`   → `rhl-inspect`
  - `RemoteState.read_log` → `rhl-logs`
- `SSHExecutor.kill_service` and `LocalExecutor.kill_service` now use
  `_try_helper(["rhl-stop", key])`. When helpers are absent they return
  `False` and the existing `RemoteState.kill_process` falls through to its
  `kill -1 PID` shell path.

### Pending — needs new helpers
Six remote operations still use heredocs / shell probes. Each should be
replaced with a new helper, then the launcher's call site rewritten on the
same `_try_helper(...) ... else fallback` pattern as the four above.

| Launcher call site                          | New helper                       | Fallback policy |
|---------------------------------------------|----------------------------------|-----------------|
| `RemoteState.launch_process`                | `rhl-launch-service`             | upload-and-run  |
| `RemoteState.stat_and_read`                 | `rhl-inspect --with-mtime`       | keep heredoc    |
| `RemoteState.verify_port_in_use`            | `rhl-verify-port`                | keep heredoc    |
| `RemoteState.handshake`                     | `rhl-handshake`                  | keep heredoc    |
| `Executor.process_exists` / `SSHExecutor.process_exists` | `rhl-pid-alive`     | keep `ps -p`    |
| `Executor._read_conda_cache`                | `rhl-conda-info`                 | keep heredoc    |

"keep heredoc" means: leave the existing `run_python` / `run_shell` body in
place as the fallback when `_try_helper` returns `None`. These bodies are all
small (≤15 lines, no Popen, no shell metachars from user input).
"upload-and-run" applies only to `rhl-launch-service` — see below.

## New helpers — specifications

All helpers live in `ssh_guard/helpers/`. Register each in
`pyproject.toml [project.scripts]`. Follow the conventions in the existing
helpers: import `handle_help`, `die` from `ssh_guard._cli`; use
`validate_key` / `validate_clearable_path` / `pid_is_live` from
`ssh_guard._state`.

### 1. `rhl-launch-service` (high priority — closes vulns 1 & 2)

```
rhl-launch-service --key KEY --workdir DIR
                   [--conda-env ENV] [--network-interface IF]
                   [--parameters JSON] [--meta JSON]
                   -- BINARY ARG...
```

- Validates `KEY` (`validate_key`), `DIR` (`validate_clearable_path` against
  the parent of the workdir, or a new `validate_workdir` that allows any
  user-writable absolute path that isn't a forbidden root).
- Loads service binaries from `tools.yaml` using the same
  `_load_service_binaries()` logic as `guard.py` (refactor to a shared
  helper module to avoid duplication).
- Validates `BINARY in service_binaries`. The remaining `ARG...` are passed
  through as argv tokens — never re-shell-parsed — so metacharacters in
  argument values cannot escape.
- If `--conda-env` is set: read `~/.remote-http-launcher/conda-setup.json`
  (fail with a clear message if absent), build the activation prefix
  server-side as `source <conda_source> && conda activate <env> && exec
  <shlex.join(BINARY ARG...)>` and Popen via `bash -lc`.
- If `--conda-env` is not set: Popen argv directly (no shell).
- Open log at `~/.remote-http-launcher/server/{key}.log` (truncate),
  `start_new_session=True`, `cwd=workdir`, redirect stdout/stderr.
- Write `{key}.json` with
  `{workdir, log, command, uid, pid, status: "starting", network_interface?,
  parameters?, meta?}`.
- Validate (server-side) that any `--status-file PATH` arg embedded in
  `ARG...` resolves inside `~/.remote-http-launcher/server/`. Reject
  otherwise.
- Exit 0.

**Edge case for `--parameters` / `--meta`**: parse as JSON, validate they're
objects, fail closed if malformed. Pass through to the JSON file unchanged.

**Tests** (mirror `tests/test_ssh_guard_integration.py`):
- happy path: launches a fake binary, JSON written, pid live;
- non-whitelist binary rejected;
- bad workdir rejected;
- conda activation prefix built correctly (mock cache file);
- `--status-file` outside server dir rejected.

### 2. `rhl-inspect --with-mtime` (extension to existing helper)

Modify `ssh_guard/helpers/inspect.py`:
- Add a `--with-mtime` flag.
- When set, emit `{"mtime": <float>, "data": <object>}` (compact JSON,
  single line). Without the flag, current pretty-printed behavior unchanged.

### 3. `rhl-verify-port HOST PORT`

```
rhl-verify-port HOST PORT
```

- Two positional args. Validate `HOST` is a hostname/IP (reuse
  `_is_valid_hostname_or_ip` — promote to `ssh_guard._state` or duplicate).
- Validate `PORT` is integer in `[1, 65535]`.
- TCP `socket.connect((host, port))` with `settimeout(2.0)`, 3 retries with
  2s sleep between failed attempts (matches existing logic in
  `RemoteState.verify_port_in_use`).
- Exit 0 on success, 1 on failure (print exception to stderr).

### 4. `rhl-handshake URL`

```
rhl-handshake URL
```

- Single positional arg.
- Validate scheme is `http` or `https` (reject `file://`, `gopher://`, etc.).
- Single GET via `urllib.request.urlopen(url, timeout=10)`.
- Exit 0 if status `2xx`, 1 on URLError, 2 on non-2xx status (preserves the
  existing distinction in the heredoc).
- Caller is responsible for retry loops (matches existing usage at the two
  call sites).

### 5. `rhl-pid-alive PID`

```
rhl-pid-alive PID
```

- Validate `PID` parses as a positive int.
- Use existing `pid_is_live` from `ssh_guard._state`.
- Exit 0 if alive, 1 if `ProcessLookupError`. `PermissionError` counts as
  alive (matches `pid_is_live` semantics).

### 6. `rhl-conda-info`

```
rhl-conda-info
```

- No args.
- Read `~/.remote-http-launcher/conda-setup.json` (path via
  `conda_cache_path()` from `ssh_guard._state`).
- Print contents verbatim to stdout. Exit 1 if file is absent. Exit 1 with
  a clear stderr message if malformed.
- Pairs with the existing `rhl-cache-conda` (which writes the cache).

### `pyproject.toml` registration

Add to `[project.scripts]`:
```
rhl-launch-service    = "ssh_guard.helpers.launch_service:main"
rhl-verify-port       = "ssh_guard.helpers.verify_port:main"
rhl-handshake         = "ssh_guard.helpers.handshake:main"
rhl-pid-alive         = "ssh_guard.helpers.pid_alive:main"
rhl-conda-info        = "ssh_guard.helpers.conda_info:main"
```

## Launcher follow-up edits (after helpers land)

Apply the `_try_helper(...) else fallback` pattern to each remaining
heredoc/probe call site:

- `RemoteState.launch_process` → `rhl-launch-service`. Build argv:
  ```python
  argv = ["rhl-launch-service",
          "--key", self.cfg.key,
          "--workdir", self.cfg.workdir]
  if self.cfg.conda_env:
      argv += ["--conda-env", self.cfg.conda_env]
  if self.cfg.network_interface:
      argv += ["--network-interface", self.cfg.network_interface]
  if self.cfg.file_parameters is not None:
      argv += ["--parameters", json.dumps(self.cfg.file_parameters)]
  if self.cfg.meta is not None:
      argv += ["--meta", json.dumps(self.cfg.meta)]
  argv += ["--", *shlex.split(evaluated_command)]
  ```
  Note: `evaluated_command` is split on the launcher side because the helper
  expects argv tokens, not a single string. The split is safe — it's the
  output of an f-string-evaluated `command_template`, no user-provided
  shell metachars are present.

- `RemoteState.stat_and_read` → `rhl-inspect --with-mtime` (one-line argv).

- `RemoteState.verify_port_in_use` → `rhl-verify-port HOST PORT`.

- `RemoteState.handshake` → `rhl-handshake URL` (URL pre-built by
  `build_handshake_url`).

- `Executor.process_exists` (both subclasses) → `rhl-pid-alive PID`.

- `Executor._read_conda_cache` → `rhl-conda-info`. Existing
  `_ensure_conda_setup` cache-then-prime-then-probe sequence stays;
  the heredoc body is replaced with `_try_helper(["rhl-conda-info"])`.

### Special case: `rhl-launch-service` upload-and-run fallback

For unguarded hosts that don't have `rhl-launch-service` installed, the
existing `launch_process` heredoc is large enough (~40 lines, runs
`subprocess.Popen` with shell-evaluated input) that we should not keep it as
a permanent fallback. Recommended fallback when `_try_helper` returns
`None` for `rhl-launch-service`:

1. `scp` the helper's main module (or a self-contained single-file copy) to
   `~/.remote-http-launcher/launch_service.py` on the remote.
2. `ssh HOST python3 ~/.remote-http-launcher/launch_service.py <argv...>`.
3. Optionally cache the upload (skip on subsequent calls in the same
   Executor lifetime).

This keeps the heredoc out of the launcher entirely once the helper exists,
while still supporting unguarded hosts. The other five helpers are small
enough that keeping the heredoc/bash fallback in-line is fine.

## Guard whitelist simplification (final step)

After the launcher rewrite is complete and tested, edit
`ssh_guard/guard.py`:

- Delete `_CONDA_PREFIX_RE`, `_INNER_PATTERNS`, `_HEREDOC_SENTINEL`,
  `_POPEN_MARKER`, `_COMMAND_LINE_RE`, `_check_python_heredoc`,
  `_strip_conda_prefix`, `_is_allowed_bash_lc`.
- `_is_allowed` collapses to:
  ```python
  def _is_allowed(command: str, _service_binaries) -> tuple[bool, str]:
      if not command.strip():
          return False, "empty command (interactive session not allowed)"
      try:
          parts = shlex.split(command)
      except ValueError as exc:
          return False, f"command parse error: {exc}"
      if not parts:
          return False, "empty command"
      if re.match(r"^rhl-[a-z][a-z-]*$", parts[0]):
          return True, "rhl helper"
      return False, f"command not in whitelist: {command[:120]!r}"
  ```
- `_load_service_binaries` is no longer needed in the guard (move to a
  shared module imported by `rhl-launch-service`).
- Remove the `tools.yaml` package-data entry from the guard's import path
  if no other guard code references it (it now lives with the launcher).

This collapse closes Sonnet's vulnerabilities 1, 2, 3, and 4 simultaneously.

## Suggested implementation order

1. `rhl-pid-alive`, `rhl-conda-info`, `rhl-handshake`, `rhl-verify-port` —
   small, independent, low-risk. Land with tests.
2. `rhl-inspect --with-mtime` extension. Add tests.
3. Launcher edits to use the five helpers above (with heredoc fallback).
4. `rhl-launch-service` — the largest helper. Land with thorough tests
   covering binary whitelist, conda activation, status-file path validation.
5. Launcher edit for `launch_process` with upload-and-run fallback.
6. Final guard whitelist simplification.
7. Update `tests/test_ssh_guard_integration.py` to cover the new helpers
   and to verify the simplified guard rejects all heredocs and shell
   metachars.

## Files involved

- `remote-http-launcher/remote_http_launcher.py` — six call-site rewrites,
  one upload-and-run helper.
- `remote-http-launcher/ssh_guard/guard.py` — whitelist simplification.
- `remote-http-launcher/ssh_guard/helpers/launch_service.py` (new)
- `remote-http-launcher/ssh_guard/helpers/verify_port.py` (new)
- `remote-http-launcher/ssh_guard/helpers/handshake.py` (new)
- `remote-http-launcher/ssh_guard/helpers/pid_alive.py` (new)
- `remote-http-launcher/ssh_guard/helpers/conda_info.py` (new)
- `remote-http-launcher/ssh_guard/helpers/inspect.py` — extend with
  `--with-mtime`.
- `remote-http-launcher/ssh_guard/_state.py` — possibly add
  `validate_workdir` and `is_valid_hostname_or_ip`.
- `remote-http-launcher/pyproject.toml` — register five new entry points.
- `remote-http-launcher/tests/test_ssh_guard_integration.py` — extend.
