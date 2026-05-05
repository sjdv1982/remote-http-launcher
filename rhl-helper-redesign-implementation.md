# Low-level implementation plan: rhl helper redesign

This is the handoff-ready implementation plan for
`seamless/plans/rhl-helper-redesign.md`, with the resolver contract in
`seamless/plans/seamless-service-resolve-contract.md` treated as binding.

The work has two packages:

- `remote-http-launcher/`: rename and redesign the guarded `rhl-*` helpers.
- `seamless-config/`: add client-side `seamless-service-*` wrappers and the
  agent-facing resolver.

The boundary is load-bearing: `rhl-*` remains Seamless-agnostic and operates on
opaque keys, paths, and JSON state. Seamless-specific resolution, cluster-wide
selection, and project/stage semantics belong in `seamless-config`.

## Contract reminders

- Do not reverse-engineer or reimplement key, workdir, or host construction in
  either the helper layer or the resolver CLI.
- `seamless-service-resolve` is an extractor over the installed
  `seamless_config.tools.configure_*` functions, not a synthesizer with a stable
  public schema.
- Seamless is not required on the service host. Server-side helpers must not
  import `seamless_config` or rely on Seamless YAML files.
- The launcher may carry a caller-provided `meta` block through state JSONs, but
  must treat that block as opaque transport data.
- All readers must tolerate state JSONs with no `meta` block.

## Suggested implementation order

1. Add helper infrastructure and new `rhl-*` modules.
2. Add launcher `meta` pass-through and update the one helper call from
   `rhl-kill-service` to `rhl-stop`.
3. Update entry points and README.
4. Run remote-http-launcher tests.
5. Add `seamless-config/service/` resolver and wrappers.
6. Add seamless-config entry points and run smoke tests.
7. Delete old helper modules and old entry points in the same change. The
   current helper/guard interface has not been released, so no compatibility
   rollout is required.

## Part 1: remote-http-launcher

Working directory: `remote-http-launcher/`.

### 1. Add shared CLI helpers

Create `ssh_guard/_cli.py`.

Functions to include:

- `handle_help(args, usage, description)`: if `-h` or `--help` is present,
  print usage, blank line, description, and exit `0`.
- `die(prefix, message, code=1)`: optional, useful for consistent stderr
  errors.
- `parse_int_flag(args, flag, default=None)`: optional helper for `--tail` and
  `--level`.
- `print_table(headers, rows)`: optional helper for aligned output. Keep it
  dependency-free.
- `emit_ndjson(rows)`: optional helper that prints `json.dumps(row,
  sort_keys=True)` once per row.

Do not use `argparse` unless you want full parser behavior in every helper. The
current helpers are tiny and simple argv parsing is acceptable.

### 2. Implement state-row utilities

Extend `ssh_guard/_state.py` or create `ssh_guard/helpers/_rows.py`.

Recommended functions:

- `read_json_file(path) -> dict | None`: return `None` for missing files; raise
  `SystemExit` with a helper-specific prefix for malformed JSON.
- `iter_state_files(directory) -> list[pathlib.Path]`: sorted `*.json`, empty if
  the directory is absent.
- `pid_is_live(pid) -> bool`: `os.kill(int(pid), 0)`, true unless
  `ProcessLookupError`; `PermissionError` should count as live but inaccessible.
- `server_row(path, *, check_pid=True) -> dict`: include:
  - all original JSON fields
  - `key`
  - original status as `json_status` if useful
  - computed `status`: `starting`, `running`, `failed`, or `stale`
  - `live`: boolean or `None`
  - `meta`: preserve original `meta` if present; omit or set `{}` for legacy rows
- `client_row(path) -> dict`: include all JSON fields, `key`, `host`,
  `ssh_host`, `port`, and effective status if present.

Server status rules:

- Missing or non-int PID with `status == "starting"`: keep `starting`.
- `status == "failed"` or `dry_run` rows: report `failed` or the existing status
  without PID probing.
- `status == "running"` and PID live: `running`.
- `status == "running"` and PID dead: `stale`.
- Any other status string: keep it, but still add `live` when PID is int.

Client effective SSH host:

- Prefer `tunneled-host`.
- Then `ssh_hostname`.
- Then `hostname`.
- Missing values display as `-` in table mode.

### 3. Implement `rhl-ps`

Create `ssh_guard/helpers/ps.py`.

CLI:

```bash
rhl-ps [--client] [--host SSH_HOST] [--status STATE] [--key] [--no-status] [--json]
```

Parsing rules:

- `--host` requires `--client`.
- `--key` and `--json` are mutually exclusive.
- `--status` filters on computed `status`.
- `--no-status` skips PID liveness checks in server mode.

Behavior:

- Server mode reads `server_dir()`.
- Client mode reads `client_dir()`.
- `--key`: print only keys, one per line.
- `--json`: NDJSON, one object per row, including original state fields and
  optional `meta`.
- Table mode:
  - server columns: `KEY STATUS PORT`
  - client columns: `KEY SSH-HOST HOST PORT`

This replaces `rhl-ls-services`; remove the old helper and entry point in this
rewrite.

### 4. Implement `rhl-ps-persistent`

Create `ssh_guard/helpers/ps_persistent.py`.

CLI:

```bash
rhl-ps-persistent <path> [<path>...] [--level N] [--file FILENAME] [--json]
```

Implementation details:

- Validate each root with `validate_clearable_path`.
- Reject relative paths, paths containing `..`, and forbidden roots.
- Walk depth is inclusive:
  - `--level 0`: report only the input path.
  - `--level 1`: report input path and direct children.
  - `--level 2`: report input path, children, grandchildren.
- Report all visited levels, not only leaves.
- If a path is absent, emit one row with `state: absent`.
- If path exists but is not a directory, exit non-zero.
- With no `--file`, populated means the directory has any direct entries.
- With `--file NAME`, populated means `path / NAME` exists; size/modified come
  from that file.
- `size` is bytes or `None`; `modified` is ISO 8601 or `None`.
- Exit `0` for absent/empty/populated. Exit non-zero only for validation or OS
  errors.

Rows:

```json
{"path": "/abs/path", "state": "populated", "size": 1234, "modified": "2026-05-04T12:34:56"}
```

Table columns: `PATH STATE SIZE MODIFIED`.

### 5. Implement `rhl-stop`

Create `ssh_guard/helpers/stop.py`.

CLI:

```bash
rhl-stop <key> [<key>...]
```

Behavior:

- Validate every key before acting.
- Read server JSON for each key.
- Do not remove any JSON files.
- Send signals in three phases across all keys:
  1. `SIGINT`, then poll every 0.5s up to 5s.
  2. Survivors get `SIGTERM`, then poll every 0.5s up to 5s.
  3. Survivors get `SIGKILL`.
- Print one line per key with outcome:
  - `stopped`
  - `already gone`
  - `missing state`
  - `kill required`
  - `permission denied`
- Exit non-zero for `PermissionError`, malformed JSON, or invalid PID. Missing
  state can be stdout and exit `0` unless the implementation wants stricter
  behavior; document whichever choice is made in help.

Parallelism can be simple in-process batching: send one signal to all current
survivors, then poll the group. No threads are needed.

This replaces the kill half of `rhl-kill-service` and `rhl-restart-cluster`;
remove the old helpers and entry points in this rewrite.

### 6. Implement `rhl-rm`

Create `ssh_guard/helpers/rm.py`.

CLI:

```bash
rhl-rm <key> [<key>...] [--client] [--server]
```

Behavior:

- Default with no side flags removes whichever client/server JSONs exist on the
  current machine.
- `--client` restricts to `client_dir()`.
- `--server` restricts to `server_dir()`.
- Both flags may be provided.
- Accept multiple keys.
- Print `removed <path>` only when a file existed.
- Print `<key>: not found` when no selected file existed.
- Do not delete log files.

Help text must mention:

- For non-persistent services, removing server JSON discards the key-to-log
  handle used by `rhl-logs`.
- Read logs before `rhl-rm` when using the `stale` post-mortem window.
- Log files remain on disk and are normally overwritten on the next launch with
  the same key.

### 7. Implement `rhl-logs`

Create `ssh_guard/helpers/logs.py`.

CLI:

```bash
rhl-logs <key> [--tail N]
```

Behavior:

- Validate key.
- Resolve log path using `key_to_server_log(key)`.
- No `--tail`: stream bytes to stdout exactly as `rhl-cat-log` does today.
- `--tail N`: read bytes, split lines, emit last N lines. Preserve binary-safe
  behavior as much as practical; logs are text in normal use.
- Exit non-zero if log file is missing.

Help text should cross-reference the `stale` post-mortem window and `rhl-rm`.

### 8. Implement `rhl-inspect`

Create `ssh_guard/helpers/inspect.py`.

CLI:

```bash
rhl-inspect <key>
```

Behavior is the current `cat_json.py` behavior with:

- new command name in messages,
- `--help`,
- pretty JSON output,
- server JSON only.

### 9. Implement `rhl-clear`

Create `ssh_guard/helpers/clear.py`.

CLI:

```bash
rhl-clear <path>
```

Behavior:

- Validate with `validate_clearable_path`.
- Require existing directory.
- Remove all direct children:
  - files and symlinks: `unlink()`
  - directories: `shutil.rmtree()`
- This is intentionally one-level clearing of children, not deletion of the
  directory itself.
- Print `removed N item(s) from <path>`.
- Exit non-zero if any child cannot be removed.

This replaces both `rhl-clear-buffer` and `rhl-clear-db`. Database clearing is
handled by passing the database workdir; `rhl-clear` removes `seamless.db` along
with any sibling direct children.

### 10. Add help to `rhl-cache-conda`

Edit `ssh_guard/helpers/cache_conda.py`:

- Import `handle_help`.
- If `-h` or `--help`, print usage and the module description and exit `0`.
- Keep runtime behavior unchanged.

### 11. Improve `rhl-guard` interactive message

Edit `ssh_guard/guard.py`.

Before loading service binaries, detect missing `SSH_ORIGINAL_COMMAND` and
print this message to stderr, then exit `1`:

```text
rhl-guard: this program is an SSH guard for remote-http-launcher.
It must be invoked via SSH, not run directly.

To install, add to ~/.ssh/authorized_keys on the remote server:
    command="rhl-guard" ssh-rsa AAAA... your-key-comment

To test a specific command:
    SSH_ORIGINAL_COMMAND="rhl-ps" rhl-guard
```

Current `_is_allowed()` permits any top-level `rhl-*` command, so no fixed
allowlist change is needed in `guard.py`. Keep this permissive helper pattern
unless the project intentionally moves to an explicit helper allowlist.

### 12. Update launcher state JSON metadata

Edit `remote_http_launcher.py`.

Add a `meta` field to `Configuration`:

```python
meta: Dict[str, Any] | None
```

Parsing:

- In `_validate_and_build`, read `data.get("meta")`.
- If absent: `None`.
- If present: require it is a dict and JSON-serializable.
- Do not validate field names or values beyond JSON serializability.
- Store it on `Configuration`.
- Add `meta` to `namespace` only if useful, but do not interpret it.

Remote server JSON writes:

- In `RemoteState.launch()` generated script, add:
  - `meta = json.loads(<literal>)`
  - `if meta is not None: data["meta"] = meta`
- In `RemoteState.write_dry_run_metadata()`, add `meta` when present.

Client JSON writes:

- In `create_local_file()`, after the payload is assembled, add
  `payload["meta"] = dict(cfg.meta)` when `cfg.meta is not None`.

Validation:

- Existing `validate_remote_running_state()` should not require `meta`.
- Existing local JSON validation should not require `meta`.

### 13. Update launcher stop helper call

Edit `SSHExecutor.kill_service()` in `remote_http_launcher.py`:

- Replace `rhl-kill-service` invocation with `rhl-stop`.
- Update error text accordingly.
- Do not add fallback to `rhl-kill-service`; these helpers have not been
  released yet.

### 14. Update package entry points

Edit `remote-http-launcher/pyproject.toml`.

Remove all old helper entry points and add only the new names:

```toml
[project.scripts]
remote-http-launcher = "remote_http_launcher:main"
rhl-guard = "ssh_guard.guard:main"
rhl-cache-conda = "ssh_guard.helpers.cache_conda:main"
rhl-ps = "ssh_guard.helpers.ps:main"
rhl-ps-persistent = "ssh_guard.helpers.ps_persistent:main"
rhl-stop = "ssh_guard.helpers.stop:main"
rhl-rm = "ssh_guard.helpers.rm:main"
rhl-logs = "ssh_guard.helpers.logs:main"
rhl-inspect = "ssh_guard.helpers.inspect:main"
rhl-clear = "ssh_guard.helpers.clear:main"
```

### 15. Update tests

Edit `remote-http-launcher/tests/test_ssh_guard_integration.py`:

- `_require_guarded_ssh()` should call `_ssh("rhl-ps")`.

Add unit tests if time permits:

- `rhl-ps --json` over temporary `REMOTE_HTTP_LAUNCHER_DIR`.
- `rhl-ps --key` bare-key output.
- stale PID classification with a definitely dead PID.
- `rhl-rm` leaves logs in place.
- `rhl-clear` removes files and directories directly below the target.
- `rhl-ps-persistent` reports absent/empty/populated and `--file` behavior.
- every new helper exits `0` on `--help`.

### 16. Update README

Edit `remote-http-launcher/README.md`.

Required sections:

- SSH Guard:
  - explain the improved interactive message,
  - mention `SSH_ORIGINAL_COMMAND="rhl-ps" rhl-guard` as a local guard test.
- Lifecycle:
  - `absent`, `starting`, `running`, `failed`, `stale`, `persistent`.
  - `stale` is the post-mortem window for non-persistent services.
  - persistent state can cause false-pass test results.
- Helper table:
  - include command name,
  - runs on client/server,
  - short purpose.
- JSON state schema:
  - `meta` is caller-provided opaque metadata,
  - readers tolerate absence,
  - launcher does not interpret it.
- Cluster-wide operations:
  - remove manual grep/xargs workaround as the preferred path,
  - state that cluster-wide operations live in `seamless-service-stop` and
    `seamless-service-rm`, not `rhl-*`.

### 17. Delete old helper modules

Delete these in the same rewrite. No backwards-compatible wrappers are needed
because the current helper set has not been released.

- `ssh_guard/helpers/kill_service.py`
- `ssh_guard/helpers/rm_state.py`
- `ssh_guard/helpers/cat_log.py`
- `ssh_guard/helpers/cat_json.py`
- `ssh_guard/helpers/ls_services.py`
- `ssh_guard/helpers/clear_buffer.py`
- `ssh_guard/helpers/clear_db.py`
- `ssh_guard/helpers/restart_cluster.py`

## Part 2: seamless-config service layer

Working directory: `seamless-config/`.

### 18. Add package skeleton

Create:

- `seamless_config/service/__init__.py`
- `seamless_config/service/_args.py`
- `seamless_config/service/_dispatch.py`
- `seamless_config/service/resolve.py`
- `seamless_config/service/ps.py`
- `seamless_config/service/stop.py`
- `seamless_config/service/rm.py`
- `seamless_config/service/logs.py`
- `seamless_config/service/inspect.py`
- `seamless_config/service/clear.py`

### 19. Add parser factory

In `seamless_config/service/_args.py`, implement:

```python
def make_parser(prog, description, *, agent_mode=False):
    ...
```

Common args:

- `--service`: choices `hashserver`, `database`, `jobserver`, `daskserver`,
  `pure-daskserver`.
- `--cluster`
- `--project`
- `--subproject`
- `--stage`
- `--substage`
- `--mode`: choices `ro`, `rw`, default `rw`.
- `--queue`
- `--frontend-name`: add this even though the high-level plan does not list it;
  current `configure_*` functions already support it and it is useful for exact
  service resolution.
- `--workdir PATH`: resolver only.

Do not add `--key`.

### 20. Implement dispatch resolution

In `seamless_config/service/_dispatch.py`, implement:

```python
def resolve(args, *, from_cwd=True):
    """Return (key, ssh_hostname, full_config)."""

def run_remote(ssh_hostname, *cmd):
    ...

def run_remote_capture(ssh_hostname, *cmd) -> str:
    ...

def run_local(*cmd):
    ...

def run_local_capture(*cmd) -> str:
    ...
```

Resolution details:

- Always call `seamless_config.config_files.load_tools()` before
  `configure_*`.
- When `from_cwd=True`, load config files using the existing config loading
  path. Current loader uses `seamless_config.get_workdir()`, so inspect
  `seamless_config.__init__` and use the existing workdir setter if available.
- When `from_cwd=False`, do not consult `os.getcwd()`. Register cluster/tool
  definitions from the standard cluster locations and `SEAMLESS_CACHE`, but do
  not read `seamless.yaml` or `seamless.profile.yaml` from cwd.
- When `args.workdir` is provided to the resolver, temporarily resolve config
  from that path instead of cwd.
- Route services:
  - `hashserver`: `configure_hashserver(args.mode, ...)`
  - `database`: `configure_database(args.mode, ...)`
  - `jobserver`: `configure_jobserver(...)`
  - `daskserver`: `configure_daskserver(...)`
  - `pure-daskserver`: `configure_pure_daskserver(...)`
- Return `key = config["key"]`.
- Return `ssh_hostname = config.get("ssh_hostname")`; local clusters have no
  host after `_configure_tool()` removes it.

Add helper:

```python
def add_meta(config, args) -> dict:
    ...
```

It should attach:

```json
{
  "service": "...",
  "cluster": "...",
  "mode": "...",
  "project": "...",
  "subproject": null,
  "stage": null,
  "substage": null,
  "queue": null
}
```

Use values that the `configure_*` path actually resolved. If current APIs do
not expose resolved project/stage cleanly, add small accessors in
`seamless_config.select` or return values from `_prepare_tool()` rather than
parsing the key.

### 21. Implement resolver CLI

In `seamless_config/service/resolve.py`:

- Use `make_parser(..., agent_mode=True)`.
- Include the required disclaimer from
  `seamless-service-resolve-contract.md` in `--help`.
- Default call: `resolve(args, from_cwd=False)`.
- If `--workdir PATH`: resolve from that workdir explicitly.
- Print JSON to stdout with:
  - `key`
  - `ssh_hostname`
  - `workdir`
  - `log_path`: `~/.remote-http-launcher/server/<key>.log`
  - service fields from `meta`
- No side effects.

Smoke check from `/tmp`:

```bash
seamless-service-resolve --service hashserver --cluster MYCLUSTER --project myproject
```

### 22. Implement action wrappers

Each wrapper should:

- parse args,
- call `_dispatch.resolve(args, from_cwd=True)`,
- dispatch to the appropriate `rhl-*` command,
- exit non-zero when subprocess return code is non-zero.

`stop.py`:

- Single-service: `run_remote(ssh_host, "rhl-stop", key)`.
- Cluster-wide: allow `--cluster C` without `--service`, enumerate
  `rhl-ps --json` on that cluster host, filter rows by cluster, batch
  `rhl-stop`.

`rm.py`:

- Flags: `--client`, `--server`; default both.
- Single-service:
  - server side: `run_remote(ssh_host, "rhl-rm", "--server", key)`
  - client side: `run_local("rhl-rm", "--client", key)`
- Cluster-wide:
  - server keys from remote `rhl-ps --json`,
  - client keys from local `rhl-ps --client --json`,
  - filter by cluster,
  - batch each side.

`logs.py`:

- Add `--tail N`.
- Dispatch `rhl-logs key [--tail N]`.

`inspect.py`:

- Dispatch `rhl-inspect key`.

`clear.py`:

- Resolve config.
- Require service is `hashserver` or `database`.
- Reject `workdir == "/tmp"`.
- Dispatch `rhl-clear <workdir>`.

Cluster filtering helper:

- Prefer `row["meta"]["cluster"] == cluster`.
- Legacy fallback must be anchored around the mode segment, not a broad
  substring match. For known service key layouts, accept prefixes such as:
  - `hashserver-<cluster>-rw-`
  - `hashserver-<cluster>-ro-`
  - `database-<cluster>-rw-`
  - `database-<cluster>-ro-`
  - `jobserver-<cluster>-rw-`
  - `daskserver-<cluster>-rw-`
  - `pure-daskserver-<cluster>-`

### 23. Implement `seamless-service-ps`

In `seamless_config/service/ps.py`.

CLI:

```bash
seamless-service-ps [--cluster CLUSTER] [--service SVC] [--project PROJ]
                    [--status STATE] [--client | --server | --all-clusters]
                    [--persistent] [--json]
```

Mode behavior:

- `--client` default: local `rhl-ps --client --json`, no SSH.
- `--server`: one cluster frontend, `rhl-ps --json` via SSH or local.
- `--all-clusters`: load all known clusters, fan out best-effort with a
  per-host timeout around 10s.

Persistent behavior:

- If requested, call on the service host:
  - hashserver: `rhl-ps-persistent --json --level 2 <bufferdir>`
  - database: `rhl-ps-persistent --json --level 2 --file seamless.db <database_dir>`
- Persistent composition lives here, not in `rhl-ps-persistent`.

Join strategy:

- Process rows are keyed by `meta` tuple:
  `(service, project, subproject, stage)`.
- Persistent rows derive service from which root was walked and project/stage
  from path relative to that root:
  - `<root>/<project>`
  - `<root>/<project>/<subproject>`
  - `<root>/<project>/STAGE-<stage>`
  - Be conservative; if path layout is ambiguous, emit the path and leave
    project/stage blank rather than guessing deeply.
- Output rows include:
  - `service`
  - `cluster`
  - `project`
  - `stage`
  - `process`
  - `port`
  - `persistent`
  - `size`
  - `key`

Table columns:

```text
SERVICE PROJECT STAGE PROCESS PORT PERSISTENT SIZE
```

`--json` emits NDJSON rows.

### 24. Update seamless-config entry points

Edit `seamless-config/pyproject.toml`.

Move to `[project.scripts]` for new tools while keeping
`script-files = ["bin/seamless-init"]`.

Add:

```toml
[project.scripts]
seamless-service-resolve = "seamless_config.service.resolve:main"
seamless-service-ps = "seamless_config.service.ps:main"
seamless-service-stop = "seamless_config.service.stop:main"
seamless-service-rm = "seamless_config.service.rm:main"
seamless-service-logs = "seamless_config.service.logs:main"
seamless-service-inspect = "seamless_config.service.inspect:main"
seamless-service-clear = "seamless_config.service.clear:main"
```

### 25. Verification

Remote HTTP launcher:

```bash
cd /home/agent/seamless1/remote-http-launcher
pip install -e .
rhl-ps --help
rhl-ps-persistent --help
rhl-logs --help
rhl-inspect --help
rhl-stop --help
rhl-rm --help
rhl-clear --help
rhl-guard
pytest tests/
```

Manual state tests:

```bash
tmp=$(mktemp -d)
REMOTE_HTTP_LAUNCHER_DIR="$tmp" rhl-ps
mkdir -p "$tmp/server" "$tmp/client"
printf '{"pid": 999999999, "status": "running", "port": 1234, "meta": {"cluster": "C"}}' > "$tmp/server/demo.json"
REMOTE_HTTP_LAUNCHER_DIR="$tmp" rhl-ps --json
REMOTE_HTTP_LAUNCHER_DIR="$tmp" rhl-rm --server demo
test ! -e "$tmp/server/demo.json"
```

Persistent tests:

```bash
root=$(mktemp -d)
REMOTE_HTTP_LAUNCHER_DIR="$(mktemp -d)" rhl-ps-persistent "$root" --level 0
touch "$root/seamless.db"
REMOTE_HTTP_LAUNCHER_DIR="$(mktemp -d)" rhl-ps-persistent "$root" --file seamless.db --json
```

Seamless config:

```bash
cd /home/agent/seamless1/seamless-config
pip install -e .
seamless-service-resolve --help
cd /tmp
seamless-service-resolve --service hashserver --cluster MYCLUSTER --project myproject
```

From a configured project directory:

```bash
seamless-service-ps
seamless-service-inspect --service hashserver
seamless-service-logs --service hashserver --tail 100
seamless-service-clear --service jobserver
```

Expected `jobserver` clear result: clear error explaining that the service uses
`/tmp` or has no persistent data.

## Known risks and decisions to preserve

- No old-name compatibility is required for the helper redesign because the
  current SSH guard/helper interface has not been released.
- `rhl-rm` leaving logs in place is intentional. Do not "clean up" logs in that
  command.
- `rhl-stop` not removing JSON is intentional. Stale JSON is useful diagnostic
  state until an operator explicitly runs `rhl-rm`.
- `rhl-ps-persistent` does not know what a Seamless project is. It walks paths
  and reports filesystem state only.
- `seamless-service-ps` is client-side by design. Do not whitelist it in
  `rhl-guard`; that would imply Seamless is installed on the server.
- The resolver disclaimer must be present anywhere users might infer output
  stability.
