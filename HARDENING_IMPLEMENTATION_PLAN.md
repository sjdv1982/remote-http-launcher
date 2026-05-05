# SSH guard hardening implementation plan

## Goal

Eliminate the remaining SSH guard injection surface by replacing all guarded
`bash -lc ...` and `python3 - <<HEREDOC` launcher paths with dedicated
`rhl-*` helper commands that accept structured argv/data instead of remote
code. Once the launcher no longer needs heredoc or shell-probe commands on
guarded hosts, simplify `ssh_guard/guard.py` so it accepts only top-level
`rhl-*` helper invocations.

This plan is intended to be handed to an implementer without requiring the
original vulnerability analysis.

## Background

The current guard tries to validate selected launcher commands by parsing
shell strings and Python heredocs. That approach is unsafe:

- Python heredocs without `subprocess.Popen` are currently allowed, which
  permits arbitrary non-launch Python code.
- Python launch heredocs are accepted if the first extracted command binary is
  whitelisted, while the rest of the Python body is not validated.
- The optional conda activation prefix accepts shell metacharacters in the
  source path and env name before the inner command is checked.
- `cat ~/.bashrc` is whitelisted and can leak secrets.

The robust fix is to stop sending remote code through the guarded SSH key.
The guard should validate a simple command name (`rhl-*`) and pass argv to
server-side helpers.

## Current branch state

Already completed in this branch:

- `Executor.invoke_helper(argv, ...)` and `Executor._try_helper(argv, ...)`
  exist in `remote_http_launcher.py`.
- `_try_helper` detects missing helpers via exit code 127, `command not found`,
  or `no such file or directory`, caches helper availability per executor, and
  returns `None` so callers can use legacy fallback paths.
- These launcher operations already try existing helpers first:
  - `RemoteState.exists` uses `rhl-inspect` and `rhl-rm --server`.
  - `RemoteState.remove` uses `rhl-rm --server`.
  - `RemoteState.read` uses `rhl-inspect`.
  - `RemoteState.read_log` uses `rhl-logs`.
  - `SSHExecutor.kill_service` and `LocalExecutor.kill_service` use
    `rhl-stop`.

Still pending:

- Add the missing helper commands listed below.
- Rewrite the remaining heredoc/probe call sites to use those helpers first.
- Add an upload-and-run fallback for service launch on unguarded hosts or
  hosts where the package is not installed server-side.
- Simplify the guard whitelist after the helper rewrites are complete.

Helper absence note:

- The guard and helpers are shipped in the same package. A host running the
  simplified guard should also have the helpers available.
- If `_try_helper` reports missing helpers, treat that as either an unguarded
  remote without the server-side package installed or a broken/partial install.
- Do not design for a supported "simplified guard but no helpers" mode.

## Implementation phases

### Phase 1: Small independent helpers

Implement and register these helpers first. They are independent and low risk.

Files to add:

- `ssh_guard/helpers/pid_alive.py`
- `ssh_guard/helpers/conda_info.py`
- `ssh_guard/helpers/handshake.py`
- `ssh_guard/helpers/verify_port.py`

Files to update:

- `pyproject.toml`
- `tests/test_ssh_guard_integration.py`
- optionally `ssh_guard/_state.py` for shared validators

#### `rhl-pid-alive PID`

Purpose: replace `ps -p <pid> -o pid=`.

Behavior:

- Accept exactly one positional argument.
- Validate that `PID` is a positive integer and not a bool-like value.
- Use `pid_is_live` from `ssh_guard._state`.
- Exit 0 if live.
- Exit 1 if not live.
- Treat `PermissionError` as live, matching `pid_is_live`.

Test coverage:

- Current process PID returns 0.
- A very large/nonexistent PID returns 1.
- Non-integer, zero, and negative PIDs are rejected with non-zero status.
- `--help` returns usage text.

#### `rhl-conda-info`

Purpose: replace the heredoc that reads
`~/.remote-http-launcher/conda-setup.json`.

Behavior:

- Accept no arguments.
- Read the path from `conda_cache_path()` in `ssh_guard._state`.
- Validate that the file exists and contains a JSON object.
- Print compact JSON to stdout.
- Exit 1 with clear stderr on missing file, malformed JSON, or non-object JSON.

Test coverage:

- Valid cache file is emitted as parseable JSON.
- Missing cache exits 1.
- Malformed JSON exits 1.
- JSON array/string exits 1.
- `--help` returns usage text.

#### `rhl-handshake URL`

Purpose: replace the Python heredoc in `RemoteState.handshake`.

Behavior:

- Accept exactly one positional URL.
- Validate scheme is `http` or `https`.
- Reject schemes such as `file`, `ftp`, `gopher`, or missing scheme.
- Perform one `urllib.request.urlopen(url, timeout=10)`.
- Exit 0 for HTTP status 2xx.
- Exit 1 on connection/URL errors.
- Exit 2 on reachable non-2xx HTTP status, preserving the existing distinction.

Caller responsibility:

- Keep retry loops and URL construction in the launcher.

Test coverage:

- Local test HTTP server returning 200 exits 0.
- Local test HTTP server returning 500 exits 2.
- Unreachable localhost port exits 1.
- Rejected scheme exits non-zero before opening.
- `--help` returns usage text.

#### `rhl-verify-port HOST PORT`

Purpose: replace the Python heredoc in `RemoteState.verify_port_in_use`.

Behavior:

- Accept exactly two positional args.
- Validate `HOST` as a hostname or IP address.
- Validate `PORT` is an integer in `[1, 65535]`.
- Connect to `(host, port)` with timeout 2.0 seconds.
- Retry 3 times total with 2 seconds of sleep between failures.
- Exit 0 on success.
- Exit 1 on failure and print the final exception to stderr.

Implementation note:

- Put hostname/IP validation in `ssh_guard._state` if it will also be useful
  elsewhere. Otherwise keep it local to `verify_port.py`.

Test coverage:

- Local listening socket exits 0.
- Closed localhost port exits 1.
- Bad host strings and out-of-range ports are rejected.
- `--help` returns usage text.

### Phase 2: Extend inspect helper

File to update:

- `ssh_guard/helpers/inspect.py`

Add `rhl-inspect <key> --with-mtime`.

Behavior:

- Preserve current behavior without `--with-mtime`: pretty-print the server
  state JSON object with sorted keys.
- With `--with-mtime`, emit compact one-line JSON:

  ```json
  {"mtime": 123.456, "data": {...}}
  ```

- `mtime` must come from `path.stat().st_mtime`.
- `data` is the parsed state object.

Test coverage:

- Existing pretty-print expectations still pass.
- `--with-mtime` returns parseable JSON with float/int `mtime` and object
  `data`.
- Missing key still exits non-zero.
- `--help` returns usage text.

### Phase 3: Launcher rewrites for small helpers

File to update:

- `remote_http_launcher.py`

Apply the existing `_try_helper(...) else fallback` pattern to these call sites.

#### `Executor._read_conda_cache`

New helper path:

```python
helper_result = self._try_helper(["rhl-conda-info"])
```

Expected handling:

- If helper result is `None`, keep the current heredoc fallback.
- If return code is non-zero, return `None`.
- If stdout is invalid JSON or not an object, return `None`.
- If valid, return the parsed dict.

Keep `_ensure_conda_setup` behavior intact:

- Try cache.
- Run `rhl-cache-conda` on cache miss.
- Try cache again.
- Fall back to legacy probes only for unguarded hosts.

#### `SSHExecutor.process_exists` and `LocalExecutor.process_exists`

New helper path:

```python
helper_result = self._try_helper(["rhl-pid-alive", str(pid)])
```

Expected handling:

- If helper result is `None`, keep existing `ps -p` fallback.
- Return `True` for exit 0.
- Return `False` for non-zero.

#### `RemoteState.stat_and_read`

New helper path:

```python
helper_result = self.executor._try_helper(
    ["rhl-inspect", self.cfg.key, "--with-mtime"]
)
```

Expected handling:

- If helper result is `None`, keep existing heredoc fallback.
- If helper exits non-zero, raise `LauncherError`.
- Parse stdout as JSON and return the payload.
- Validate payload shape enough to avoid confusing downstream errors:
  `payload` is a dict with `mtime` and `data`.

#### `RemoteState.verify_port_in_use`

New helper path:

```python
helper_result = self.executor._try_helper(
    ["rhl-verify-port", host, str(port)]
)
```

Expected handling:

- If helper result is `None`, keep existing heredoc fallback.
- If helper exits non-zero, raise `LauncherError` with helper stderr.
- Return normally on exit 0.

#### `RemoteState.handshake`

New helper path:

```python
url = build_handshake_url(host, port, handshake)
helper_result = self.executor._try_helper(["rhl-handshake", url])
```

Expected handling:

- If helper result is `None`, keep existing heredoc fallback.
- Return normally on exit 0.
- Preserve existing non-zero semantics as much as possible:
  - exit 1 means request/connection failure.
  - exit 2 means reachable non-2xx response.
- Raise `LauncherError` with stderr if this method currently expects checked
  execution to raise on failure at the call site.

Launcher test coverage:

- Unit-test or integration-test each rewritten path with a fake executor that
  returns helper success.
- Test helper-missing fallback still executes the legacy body.
- Test helper non-zero failure paths raise or return according to existing
  behavior.

### Phase 4: Shared service binary loading

Files to update/add:

- Add `ssh_guard/_tools.py` or similarly named shared module.
- Update `ssh_guard/guard.py` temporarily to import from it, if useful during
  transition.
- Use it from `ssh_guard/helpers/launch_service.py`.

Move the current `_load_service_binaries()` logic out of `guard.py` so the
new launch helper can validate the service binary using the same source of
truth.

Behavior:

- Preserve `RHL_TOOLS_YAML` override support.
- Preserve packaged `ssh_guard/tools.yaml` fallback.
- Return a `frozenset[str]` of first tokens from each `command_template`.
- Handle missing/invalid YAML with clear errors at helper startup.

Test coverage:

- Default `tools.yaml` loads at least the expected known service binary names.
- `RHL_TOOLS_YAML` override loads from a temporary YAML file.
- Empty/malformed tool definitions fail closed.

### Phase 5: `rhl-launch-service`

File to add:

- `ssh_guard/helpers/launch_service.py`

Purpose: replace the large Python heredoc in `RemoteState.launch_process`.
This is the highest priority helper because it closes the Python heredoc code
injection issues.

Command:

```text
rhl-launch-service --key KEY --workdir DIR
                   [--conda-env ENV] [--network-interface IF]
                   [--parameters JSON] [--meta JSON]
                   -- BINARY ARG...
```

Validation:

- `KEY`: use `validate_key`.
- `DIR`: add `validate_workdir` in `ssh_guard._state` or use
  `validate_clearable_path` only if its semantics are appropriate.
- Workdir must be absolute after expansion and must not be a forbidden system
  root.
- Workdir must not contain raw `..` path components.
- `--parameters` and `--meta`, if present, must parse as JSON objects.
- `BINARY` must be present and must be in the shared service binary whitelist.
- `ARG...` are argv tokens. Never re-parse them through a shell except inside
  the controlled conda activation wrapper described below.
- Validate any `--status-file PATH` argument pair inside `ARG...`:
  - Resolve/expand the path.
  - Require it to live under `server_dir()`.
  - Reject missing value or path escape.

Launch behavior without conda:

- Ensure `server_dir()` exists.
- Ensure workdir exists.
- Truncate/open `key_to_server_log(key)` in binary append or write mode.
- Start the process with:

  ```python
  subprocess.Popen(
      [binary, *args],
      cwd=workdir,
      stdout=stdout_handle,
      stderr=subprocess.STDOUT,
      start_new_session=True,
  )
  ```

- Do not use `shell=True`.

Launch behavior with conda:

- Read `conda_cache_path()` and require a valid object.
- Use `conda_source` from the cache if present.
- Build a controlled activation command server-side:

  ```text
  source <conda_source> && conda activate <env> && exec <shlex.join(argv)>
  ```

- Run that through `subprocess.Popen(["bash", "-lc", command], ...)`.
- This is the only shell usage in the helper. It is server-side and constructed
  from validated/quoted argv tokens.
- If `conda_source` is absent, use `conda activate <env>` assuming conda is on
  PATH, or fail closed if that is not reliable in the current environment.

State JSON:

- Write `key_to_server_json(key)` after the process starts.
- Use an atomic write if practical: write to a temporary file in `server_dir()`
  and replace.
- Include:

  ```json
  {
    "workdir": "...",
    "log": "...",
    "command": "BINARY ARG...",
    "uid": 1000,
    "pid": 12345,
    "status": "starting",
    "network_interface": "...",
    "parameters": {},
    "meta": {}
  }
  ```

- Omit optional fields when not supplied, matching current launcher behavior.

Exit behavior:

- Exit 0 after state is written.
- Exit non-zero with clear stderr for validation errors, missing conda cache,
  failed process spawn, or failed state write.

Test coverage:

- Happy path launches a fake binary/script, writes JSON, and PID is live.
- Non-whitelisted binary is rejected.
- Missing binary after `--` is rejected.
- Bad key is rejected.
- Bad workdir is rejected.
- `--parameters` and `--meta` reject malformed JSON and non-object JSON.
- Valid parameters/meta are preserved in state JSON.
- `--status-file` under `server_dir()` is accepted.
- `--status-file` outside `server_dir()` is rejected.
- Conda path builds the expected activation command using a mock cache.
- Log file is created/truncated.
- `--help` returns usage text.

### Phase 6: Launcher launch rewrite

File to update:

- `remote_http_launcher.py`

Replace `RemoteState.launch_process` helper path first. Keep a fallback only
for unguarded hosts or hosts where the server-side package is not installed.

Build argv from the evaluated command:

```python
evaluated_command = self.evaluate_command()
argv = [
    "rhl-launch-service",
    "--key", self.cfg.key,
    "--workdir", self.cfg.workdir,
]
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

Important:

- `evaluated_command` is currently the result of an f-string-evaluated
  `command_template`.
- The launcher must split it into argv tokens because the helper accepts data,
  not a single command string.
- If `shlex.split` fails or produces no tokens, raise `LauncherError`.
- Keep the existing validation that the evaluated command is non-empty and
  does not contain newlines.

Helper handling:

- If `rhl-launch-service` exits 0, return.
- If it exits non-zero, raise `LauncherError` with stderr.
- If `_try_helper` returns `None`, use the upload-and-run fallback below.

#### Upload-and-run fallback

Do not keep the large launch heredoc as the permanent fallback. Instead, when
`rhl-launch-service` is missing on a host that does not have the server-side
package installed:

1. Upload a self-contained copy of the launch helper to
   `~/.remote-http-launcher/launch_service.py`.
2. Invoke:

   ```text
   python3 ~/.remote-http-launcher/launch_service.py <same argv without rhl-launch-service>
   ```

3. Cache the upload per executor instance so repeated launches do not re-upload
   unnecessarily.

Implementation options:

- Preferred: create a self-contained helper script generator that embeds the
  minimal validation/state code needed by `launch_service.py`.
- Acceptable: upload the installed module source only if all imports are
  guaranteed to be present on the target.

Acceptance criteria:

- Guarded hosts use only top-level `rhl-launch-service`.
- Unguarded hosts without a server-side package install can still launch
  services through upload-and-run.
- The old `subprocess.Popen(..., shell=True)` heredoc is removed from the
  launcher launch path.

### Phase 7: Register all helpers

File to update:

- `pyproject.toml`

Add:

```toml
rhl-launch-service    = "ssh_guard.helpers.launch_service:main"
rhl-verify-port       = "ssh_guard.helpers.verify_port:main"
rhl-handshake         = "ssh_guard.helpers.handshake:main"
rhl-pid-alive         = "ssh_guard.helpers.pid_alive:main"
rhl-conda-info        = "ssh_guard.helpers.conda_info:main"
```

Keep existing helpers registered.

### Phase 8: Simplify guard whitelist

Start this phase only after phases 1 through 7 pass tests.

File to update:

- `ssh_guard/guard.py`

Delete the legacy parser surface:

- `_CONDA_PREFIX_RE`
- `_INNER_PATTERNS`
- `_HEREDOC_SENTINEL`
- `_POPEN_MARKER`
- `_COMMAND_LINE_RE`
- `_check_python_heredoc`
- `_strip_conda_prefix`
- `_is_allowed_bash_lc`
- guard-local `_load_service_binaries` if moved to the launch helper only

Collapse `_is_allowed` to top-level helpers only:

```python
def _is_allowed(command: str, _service_binaries=None) -> tuple[bool, str]:
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

Also update `main()` so it no longer loads service binaries if they are not
needed by the guard.

Preserve helper exec behavior:

- Parse `SSH_ORIGINAL_COMMAND` with `shlex.split`.
- If the requested helper exists next to `rhl-guard`, exec that sibling path.
- Otherwise `os.execvp(args[0], args)`.

Guard test coverage:

- `rhl-ps` and other registered helper names are accepted.
- Empty/interactive sessions are rejected.
- Raw commands are rejected:
  - `kill -1 999999`
  - `python3 -c 'print(1)'`
  - `bash -lc 'ps -p 1 -o pid='`
  - `bash -lc 'cat ~/.bashrc'`
  - any Python heredoc
- Shell metacharacter attempts are rejected because they are not top-level
  `rhl-*` helpers.

## End-to-end acceptance criteria

The hardening is complete when all of the following are true:

- No guarded launcher success path requires `bash -lc` except inside
  server-side helpers where explicitly justified, such as conda activation.
- No guarded launcher success path sends Python heredocs over SSH.
- `ssh_guard/guard.py` accepts only top-level `rhl-*` commands.
- Service launch validates the service binary against `tools.yaml`.
- Service launch passes service arguments as argv tokens and does not use
  `shell=True` unless executing the controlled conda wrapper.
- Conda cache reads use `rhl-conda-info`.
- Process existence checks use `rhl-pid-alive`.
- Port verification uses `rhl-verify-port`.
- Handshake uses `rhl-handshake`.
- State stat/read uses `rhl-inspect --with-mtime`.
- The legacy conda probes and small heredoc fallbacks are reachable only for
  unguarded hosts or hosts without the server-side package installed.
- The large launch heredoc is removed in favor of upload-and-run fallback.
- Tests cover helper happy paths, validation failures, launcher fallback paths,
  and final guard rejection of all legacy shell/heredoc patterns.

## Suggested test commands

Run from `remote-http-launcher/`:

```bash
python -m pytest tests/test_ssh_guard_integration.py
```

If guarded localhost is configured:

```bash
python -m pytest tests/test_ssh_guard_integration.py -k guard
```

Also run any existing launcher tests in this package after changing
`remote_http_launcher.py`.

## Files checklist

Expected new files:

- `ssh_guard/helpers/launch_service.py`
- `ssh_guard/helpers/verify_port.py`
- `ssh_guard/helpers/handshake.py`
- `ssh_guard/helpers/pid_alive.py`
- `ssh_guard/helpers/conda_info.py`
- `ssh_guard/_tools.py` if service binary loading is shared

Expected changed files:

- `remote_http_launcher.py`
- `ssh_guard/guard.py`
- `ssh_guard/helpers/inspect.py`
- `ssh_guard/_state.py`
- `pyproject.toml`
- `tests/test_ssh_guard_integration.py`

## Rollout notes

- This is unreleased code, and guard plus helpers are distributed together.
  There is no supported rollout state where the simplified guard exists without
  the helpers.
- Keep helper-first plus fallback behavior for unguarded hosts or hosts where
  the package is not installed server-side.
- If a guarded host reports missing helpers after this work, treat it as an
  installation/PATH error to fix, not as a fallback scenario.
- Treat any new remote action as data-oriented helper argv. Do not add new
  guard patterns for shell strings or Python source.
