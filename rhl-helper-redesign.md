# Implementation plan: rhl-* helper redesign for human-facing support

## Context

The rhl-* helper tools were designed for agentic use — agents read source to understand
them. Adding human support exposes seven UX problems:
confusing signal choice, opaque lifecycle, no --help, no log tailing, misleading tool
names, unclear execution semantics, and an unhelpful guard message when run interactively.

All changes are in /home/agent/seamless1/remote-http-launcher.

---

## Summary of changes

| Old name            | New name     | Disposition                          |
|---------------------|-------------|--------------------------------------|
| rhl-ls-services     | rhl-ps       | rewrite (status table, PID check, --host filter, --json) |
| (new)               | rhl-ps-persistent | new (directory walker for persistent state) |
| rhl-kill-service    | rhl-stop     | rewrite (signal escalation, batch keys) |
| rhl-rm-state        | rhl-rm       | rewrite (honest output, batch keys, leaves logs) |
| rhl-cat-log         | rhl-logs     | rewrite (--tail N)                   |
| rhl-cat-json        | rhl-inspect  | rename + add --help                  |
| rhl-clear-buffer    | rhl-clear    | merge into one generic tool          |
| rhl-clear-db        | (absorbed)   | delete; rhl-clear covers it          |
| rhl-restart-cluster | (removed)    | cluster-wide ops live in seamless-service-stop / -rm (Part 2) |
| rhl-cache-conda     | keep         | add --help only                      |
| rhl-guard           | keep         | improve interactive message          |

**rhl-* layer purity**: the rhl-* tools are seamless-agnostic. They operate on
opaque keys and do not interpret the structure of those keys. Cluster-wide
operations (which require knowing the key's cluster segment) belong to the
seamless-service-* layer, which enumerates keys via `rhl-ps --json` and then
batches them to `rhl-stop` / `rhl-rm` (both of which accept multiple keys).
This boundary is load-bearing — see `seamless-service-resolve-contract.md`,
"Server-side nuance".

---

## 1. New shared CLI utility: ssh_guard/_cli.py

Add a tiny module imported by every helper.

```python
def handle_help(args: list[str], usage: str, description: str) -> None:
    """Print help and exit 0 if -h or --help is in args."""
    if "-h" in args or "--help" in args:
        print(f"{usage}\n\n{description}")
        raise SystemExit(0)
```

Each helper calls `handle_help(sys.argv[1:], "usage: rhl-xxx ...", "...")` as its
first action. The description is the existing module docstring, reformatted.

---

## 2. rhl-ps  (ssh_guard/helpers/ps.py)

Replaces: rhl-ls-services  
Runs on: **server** (default) or **local** (--client)

```
rhl-ps [--client] [--host <ssh_hostname>] [--status <state>] [--key] [--no-status] [--json]
```

Flags:
- `--client`: list client-side connections instead of server-side services
- `--host <ssh_hostname>`: (only with --client) filter to connections whose SSH host
  matches; checks `tunneled-host` first, then `ssh_hostname`, then `hostname`
- `--status <state>`: filter output to rows matching the given lifecycle state
  (running | starting | failed | stale); generic, no Seamless knowledge required
- `--key`: output just the key, one per line (reproduces current rhl-ls-services
  behaviour; useful for scripting and piping to rhl-stop / rhl-rm)
- `--no-status`: show table without doing PID liveness checks (faster; reports
  the status field from the JSON as-is, without verifying the process is alive)
- `--json`: emit one JSON object per line (NDJSON) instead of an aligned table.
  Each object contains every field from the underlying state JSON (key, status,
  port, workdir, log, pid, plus the `meta` block — see §11) augmented with the
  computed live-status. Mutually exclusive with --key. This is the format
  consumed by `seamless-service-ps`.

Behaviour:
- Globs `*.json` in server_dir() (or client_dir() with --client)
- Server mode (default): read pid and status from each JSON; probe `os.kill(pid, 0)`
  to distinguish running/stale; derive actual state: starting | running | failed | stale
- Client mode: read hostname, port, and SSH host from each JSON; no PID to check
- Print aligned table to stdout (or bare keys with --key, or NDJSON with --json)

Server output columns: KEY  STATUS  PORT
Client output columns: KEY  SSH-HOST  HOST  PORT

The SSH-HOST column shows `tunneled-host` when present, else `ssh_hostname`, else
`hostname` (the effective SSH target for that connection). Missing values shown as `-`.
State shown as `stale` when JSON exists but PID is dead.

Note: --service / --project / --stage filtering is Seamless-specific and belongs in
the seamless-config wrapper, not here.

---

## 2a. rhl-ps-persistent  (ssh_guard/helpers/ps_persistent.py)

New tool (no predecessor)
Runs on: **server** (wherever the data lives)

```
rhl-ps-persistent <path> [<path>...] [--level N] [--file FILENAME] [--json]
```

Walks one or more directory roots and reports populated/empty/absent state per
visited path. Generic and seamless-agnostic — the seamless-specific composition
(which roots, which `--file`) lives in `seamless-service-ps`. Designed for
machine consumption: humans rarely call this directly, they use
`seamless-service-ps`.

Flags:
- `--level N`: walk depth, default `0`. `0` is a point check on the given path(s);
  `N >= 1` walks N levels deep below each input path. **Reports all levels
  traversed**, not just leaves — Seamless callers benefit from seeing the
  project-level row alongside the stage-level rows.
- `--file FILENAME`: a directory is "populated" only if it contains a file named
  `FILENAME`. Without `--file`, any contents (files or subdirs) count as
  populated. When `--file` is set, the reported `size`/`modified` are for that
  file specifically, not the directory.
- `--json`: emit one JSON object per line (NDJSON) instead of an aligned table.
  Expected to be the dominant mode.

Behaviour:
- Validate each input path: absolute, no `..`, not a system root (reuse
  `validate_clearable_path`)
- For each input path, walk to the requested depth (inclusive of all intermediate
  levels)
- Per visited path, report state:
  - `absent`: path does not exist
  - `empty`: directory exists but no entries (or no entry matching `--file`)
  - `populated`: directory has entries (or contains the named file)
- Exit 0 in all cases — `absent` and `empty` are valid states, not errors
- Exit non-zero only if path validation fails or an OS error occurs

Output (table mode):
```
PATH                                            STATE        SIZE     MODIFIED
/data/buffers/myproject                         populated    42 MB    2026-04-12
/data/buffers/myproject/STAGE-fingertip         populated    17 MB    2026-04-29
/data/buffers/oldproject                        empty        -        -
/data/buffers/myproject/STAGE-archived          absent       -        -
```

Output (`--json`, one object per line):
```
{"path": "/data/buffers/myproject", "state": "populated", "size": 44040192, "modified": "2026-04-12T..."}
{"path": "/data/buffers/myproject/STAGE-fingertip", "state": "populated", "size": ..., "modified": "..."}
```

Why this matters: persistent state is the cause of the false-pass scenario the
seamless-remote-debugging skill warns about (a test passes because cached buffers
or DB rows still exist from a prior run, not because the computation actually
ran). `seamless-service-ps --persistent` uses this tool to surface that data so
operators can decide what to clear before re-running.

---

## 3. rhl-stop  (ssh_guard/helpers/stop.py)

Replaces: rhl-kill-service (and absorbs the kill half of rhl-restart-cluster)
Runs on: **server** (the machine running the service)

```
rhl-stop <key> [<key>...]
```

Accepts one or more keys positionally. Cluster-wide stops are not a concept
at this layer — `rhl-stop` does not know what a cluster is. Callers that want
to stop every service for a cluster enumerate the keys themselves (via
`rhl-ps`, optionally with --json) and pass the resulting list to a single
`rhl-stop` invocation. The seamless-service-stop wrapper does this with
`--cluster`; see Part 2.

Signal escalation, applied to every key in parallel:
1. Send SIGINT to all PIDs; poll every 0.5 s for up to 5 s
2. To survivors: send SIGTERM; poll every 0.5 s for up to 5 s
3. To remaining survivors: send SIGKILL

Does NOT remove state JSONs (that is rhl-rm's job). Prints a line per key:
signal sent, outcome (stopped / already gone / kill required). Exits non-zero
if any PermissionError is encountered.

---

## 4. rhl-rm  (ssh_guard/helpers/rm.py)

Replaces: rhl-rm-state (and absorbs the cleanup half of rhl-restart-cluster)
Runs on: **server** (--server), **local** (--client), or both (default)

```
rhl-rm <key> [<key>...] [--client] [--server]
```

Accepts one or more keys positionally. Cluster-wide removes are not a concept
at this layer; the seamless-service-rm wrapper composes `rhl-ps` enumeration
with batched `rhl-rm` (see Part 2).

Behaviour:
- Default (no flags): remove both server and client JSON for each key, where
  each exists. On a frontend (server JSONs only) or laptop (client JSONs only),
  this naturally degrades to "whichever side is present" — absence of one side
  is not an error.
- Explicit flags: restrict to that side only.
- Per key: only print "removed X" when the file actually existed; print
  "X: not found" otherwise (to stdout, not stderr — absence is not an error).
- **Does NOT delete log files.** Server-side log files at
  `~/.remote-http-launcher/server/<key>.log` are left in place. They are
  typically overwritten by the next service launch with the same key, so
  manual cleanup is rarely needed. This deliberate asymmetry preserves
  post-mortem inspectability even after JSONs are removed.

Help text caveat (must appear in --help):
- For non-persistent services (jobserver, daskserver, pure-daskserver), the
  JSON is the only handle that lets `rhl-logs <key>` find the log file. After
  rhl-rm, the log file may still be on disk (and readable directly by
  filename) but `rhl-logs` will not find it. Read logs via `rhl-logs` first if
  post-mortem analysis may be needed. The `stale` state is the post-mortem
  window.

---

## 5. rhl-logs  (ssh_guard/helpers/logs.py)

Replaces: rhl-cat-log  
Runs on: **server**

```
rhl-logs <key> [--tail N]
```

Without --tail: stream the full log (current behaviour).
With --tail N: print the last N lines only (read bytes, split on \n, slice).

Note on log persistence: log files persist across service restarts (the same
path is overwritten on next launch with the same key) and across `rhl-rm`
(logs are not deleted with the JSON; see §4). For non-persistent services
(jobserver, daskserver, pure-daskserver), the `stale` state is the post-mortem
window — read logs here, before `rhl-rm` discards the JSON-to-log handle.

---

## 6. rhl-inspect  (ssh_guard/helpers/inspect.py)

Replaces: rhl-cat-json  
Runs on: **server**

```
rhl-inspect <key>
```

Behaviour identical to current rhl-cat-json. Only changes: new name, --help support,
updated error prefix.

---

## 7. rhl-clear  (ssh_guard/helpers/clear.py)

Replaces: rhl-clear-buffer AND rhl-clear-db  
Runs on: **server** (wherever the data lives)

```
rhl-clear <path>
```

Behaviour:
- Validate path: absolute, no `..`, not a system root (reuse validate_clearable_path)
- Path must exist and be a directory
- Remove all direct children (files AND subdirectories, non-recursive) using
  shutil.rmtree for subdirs and entry.unlink() for files
- Print "removed N item(s) from <path>"
- Exit non-zero if any removal fails

This supersedes rhl-clear-buffer (files only) and rhl-clear-db (hardcoded filename).
The Seamless-layer wrapper (in seamless-config) is responsible for constructing the
right paths to pass.

Note: `_FORBIDDEN_ROOTS` in _state.py already protects against dangerous paths.
Add `shutil` import in the new clear.py.

---

## 8. rhl-guard improvements  (ssh_guard/guard.py)

When `SSH_ORIGINAL_COMMAND` is absent (interactive run):

```
rhl-guard: this program is an SSH guard for remote-http-launcher.
It must be invoked via SSH, not run directly.

To install, add to ~/.ssh/authorized_keys on the remote server:
    command="rhl-guard" ssh-rsa AAAA... your-key-comment

To test a specific command:
    SSH_ORIGINAL_COMMAND="rhl-ps" rhl-guard
```

Exit 1 (unchanged).

---

## 9. Files to delete

- ssh_guard/helpers/kill_service.py
- ssh_guard/helpers/rm_state.py
- ssh_guard/helpers/cat_log.py
- ssh_guard/helpers/cat_json.py
- ssh_guard/helpers/ls_services.py
- ssh_guard/helpers/clear_buffer.py
- ssh_guard/helpers/clear_db.py
- ssh_guard/helpers/restart_cluster.py

---

## 10. pyproject.toml  [project.scripts]

Remove all old entries; add:

```toml
remote-http-launcher = "remote_http_launcher:main"
rhl-guard            = "ssh_guard.guard:main"
rhl-cache-conda      = "ssh_guard.helpers.cache_conda:main"
rhl-ps               = "ssh_guard.helpers.ps:main"
rhl-ps-persistent    = "ssh_guard.helpers.ps_persistent:main"
rhl-stop             = "ssh_guard.helpers.stop:main"
rhl-rm               = "ssh_guard.helpers.rm:main"
rhl-logs             = "ssh_guard.helpers.logs:main"
rhl-inspect          = "ssh_guard.helpers.inspect:main"
rhl-clear            = "ssh_guard.helpers.clear:main"
```

---

## 11. remote_http_launcher.py

Two changes:

1. One reference: `["ssh", self._host, "rhl-kill-service", key]`
   → change to `["ssh", self._host, "rhl-stop", key]`

2. **Add a `meta` block to server-side and client-side state JSONs at write
   time.** The block contains the structured config that produced the key:

   ```json
   "meta": {
     "service": "hashserver",
     "cluster": "MYCLUSTER",
     "mode": "rw",
     "project": "myproject",
     "subproject": null,
     "stage": null,
     "substage": null,
     "queue": null
   }
   ```

   The launcher receives this block as part of the config dict from its caller
   (seamless-config's `configure_*` functions, or any other consumer); it
   writes it through verbatim, treating the contents as opaque metadata. This
   lets `rhl-ps --json` emit structured fields without parsing the key, and it
   lets `seamless-service-ps` join process state with persistent state by
   `(service, project, [stage])` tuples without key reverse-engineering.

   **Backwards tolerance**: rhl-* readers must treat `meta` as optional. JSONs
   written by older launcher versions, or by callers that don't populate meta,
   still display correctly — the key remains the canonical identifier;
   meta-derived columns are blank for legacy rows.

   **Boundary respect**: the launcher does not interpret the meta block. It is
   seamless-specific data carried through a seamless-agnostic transport. The
   contract laid out in `seamless-service-resolve-contract.md` (server-side
   nuance) is preserved: no Seamless logic on the launcher side, no
   server-side Seamless install required.

---

## 12. tests/test_ssh_guard_integration.py

One reference: `_ssh("rhl-ls-services")` in `_require_guarded_ssh()`  
→ change to `_ssh("rhl-ps")`

---

## 13. README.md

Update the helper reference table (new names, new descriptions including "Runs on").
Add a Lifecycle section before the helper table explaining the six states:
absent | starting | running | failed | stale | persistent
and which tool drives each transition. The persistent state applies when a service's
data directory (workdir) is non-empty after the JSON has been removed;
`rhl-ps-persistent` enumerates it, `rhl-clear` transitions out of it.

Within the Lifecycle section, call out two operationally important cases:

- **Persistent state is the cause of false-pass test results.** A new service
  launched against a populated bufferdir or seamless.db will return cached
  results without exercising the underlying computation. Operators debugging
  a suspicious pass should use `seamless-service-ps --persistent` to surface
  cached data and `seamless-service-clear` (or `rhl-clear`) to wipe it.
- **The `stale` state is the post-mortem window for non-persistent services.**
  jobserver/daskserver/pure-daskserver have no persistent data; their log file
  is the only post-mortem artefact, and it is reachable via `rhl-logs` only
  while the JSON exists. Read logs *before* `rhl-rm`; the log file itself
  survives the JSON's removal but is no longer addressable by key through the
  helper.

Add a brief JSON state schema reference (three or four lines) documenting the
`meta` block introduced in §11 — its fields, its opaque-to-launcher status,
and the backwards-tolerance contract for readers.

Update the SSH Guard section to reflect the new interactive message.
Remove the manual grep-xargs workaround for cluster cleanup. The replacement
is `seamless-service-stop --cluster <C>` followed by
`seamless-service-rm --cluster <C>` (Part 2) — both compose `rhl-ps`
enumeration with batched `rhl-stop` / `rhl-rm` calls. The README should make
clear that cluster-wide ops live in the seamless-service-* layer; the rhl-*
layer itself has no notion of "cluster".

---

## 14. SSH guard allowlist

`rhl-guard` permits a fixed set of helper commands. Update its allowlist to
match the new entry-point names:

- Remove: `rhl-ls-services`, `rhl-kill-service`, `rhl-rm-state`, `rhl-cat-log`,
  `rhl-cat-json`, `rhl-clear-buffer`, `rhl-clear-db`, `rhl-restart-cluster`
- Add: `rhl-ps`, `rhl-ps-persistent`, `rhl-stop`, `rhl-rm`, `rhl-logs`,
  `rhl-inspect`, `rhl-clear`

If the allowlist is derived from the `[project.scripts]` entries in
`pyproject.toml` or from a list in `tools.yaml`, no manual maintenance is
needed; otherwise update the relevant constant in `ssh_guard/`.

Sequencing: the new helpers must be allowed before the old entry points are
removed from disk on any guarded server. In practice this means: ship a
release where both old and new are allowed, install it everywhere, then ship
the cleanup release that removes the old entries. A single-release rename
breaks SSH calls during upgrade windows.

## Additional improvements from design discussion

| Topic | Resolution |
|-------|-----------|
| False-pass debugging needs persistent-state enumeration | rhl-ps-persistent walks roots, reports absent/empty/populated; seamless-service-ps composes it for human view |
| Agents need structured access to service identity without key parsing | meta block added to launcher JSONs (§11); rhl-ps --json emits it verbatim |
| Log post-mortem for non-persistent services | rhl-rm leaves log files in place; rhl-logs/rhl-rm --help cross-reference the stale-state window |
| Cluster glob fragility | anchored on mode segment (rw/ro) in §3, §4 |
| Agent-friendly seamless-specific resolver | new seamless-service-resolve (Part 2) — extractor contract documented in seamless-service-resolve-contract.md |
| Server-side Seamless never required | preserved by treating meta as opaque in launcher; seamless-service-ps lives client-side only |

---

---

# Part 2: seamless-config service layer

## Context

The rhl-* tools operate on raw keys and run on whatever machine the user invokes them
on (with explicit SSH). For human use, Seamless users think in terms of cluster, project,
stage, and service — not opaque synthesised keys.

The seamless-service-* family fills this gap. Two distinct audiences are served:

- **Human action wrappers** (`seamless-service-stop`, `-rm`, `-logs`, `-inspect`,
  `-clear`, `-ps`): semantic args, cwd-resolved defaults, table output. Humans
  use these directly.
- **Agent-friendly resolver** (`seamless-service-resolve`): the same resolution
  logic exposed as a pure data tool — explicit args (no cwd magic), JSON out,
  no side effects. Agents use this to translate Seamless-level inputs into
  rhl-level identifiers, then call rhl-* directly. The contract for this tool
  (extractor, not synthesizer) is laid out in
  `seamless-service-resolve-contract.md` and is load-bearing for the whole
  agent/human split — read it before changing anything in this Part.

The action wrappers reuse the resolver's underlying library function
(`_dispatch.resolve()`); the resolver CLI exposes that same function to
external callers. There is exactly one synthesis implementation in the
codebase.

`rhl-guard` and `rhl-cache-conda` have no equivalent here — they are server
installation/maintenance tools, not operational tools.

---

## Location

New package: `seamless_config/service/`

Files:
- `seamless_config/service/__init__.py`  (empty)
- `seamless_config/service/_args.py`     (shared argument parser factory)
- `seamless_config/service/_dispatch.py` (resolution + SSH dispatch library)
- `seamless_config/service/resolve.py`   (agent-friendly resolver CLI)
- `seamless_config/service/ps.py`        (composer over rhl-ps + rhl-ps-persistent)
- `seamless_config/service/stop.py`
- `seamless_config/service/rm.py`
- `seamless_config/service/logs.py`
- `seamless_config/service/inspect.py`
- `seamless_config/service/clear.py`

---

## Shared argument parser: _args.py

```python
def make_parser(prog, description, *, agent_mode: bool = False) -> argparse.ArgumentParser
```

Adds the following arguments. By default (action wrappers) all args are optional
and defaults are resolved from cwd YAML files. With `agent_mode=True` (used by
`seamless-service-resolve`), args are still optional at the parser level but
the dispatcher does not consult cwd YAMLs — see _dispatch.py.

| Flag | Applies to | Notes |
|------|-----------|-------|
| `--service` | all | required; choices: hashserver database jobserver daskserver pure-daskserver |
| `--cluster` | all | default: from seamless.profile.yaml (action wrappers only) |
| `--project` | all except pure-daskserver | default: from seamless.yaml (action wrappers only) |
| `--subproject` | all except pure-daskserver | optional |
| `--stage` | all except pure-daskserver | optional |
| `--substage` | jobserver, daskserver | optional |
| `--mode` | hashserver, database | choices: ro rw; default: rw |
| `--queue` | pure-daskserver | optional |
| `--workdir PATH` | resolver only | opt-in to cwd-style YAML defaulting from PATH |

No `--key` flag: callers that already have a key go straight to rhl-* without
this layer.

---

## Shared dispatch: _dispatch.py

```python
def resolve(args, *, from_cwd: bool = True) -> tuple[str, str | None, dict]
    """Return (key, ssh_hostname, full_config).
    ssh_hostname is None for a local cluster.
    If from_cwd is False, do not read seamless.yaml / seamless.profile.yaml
    from cwd; rely solely on explicit args + environment.
    """

def run_remote(ssh_hostname: str | None, *cmd: str) -> None
    """If ssh_hostname is not None, exec ssh <ssh_hostname> <cmd>.
    Otherwise exec <cmd> directly. Exits non-zero on failure."""
```

`resolve()` (when `from_cwd=True`, the action-wrapper default): loads
seamless.yaml and seamless.profile.yaml from cwd (same mechanism as
`seamless.config.init`) to populate defaults in `select.py`, then calls the
appropriate `configure_*` function from `tools.py` based on `--service`.
Returns the `key` from the config dict and the `ssh_hostname` (None when the
cluster is local, i.e., when `configure_*` stripped the hostname key).

When `from_cwd=False` (the resolver default): skips the YAML-from-cwd step
entirely. `os.getcwd()` is never consulted. All inputs must come from explicit
args plus environment (e.g., `SEAMLESS_CACHE`). If `--workdir PATH` was
provided, load YAML files from `PATH` instead. The resolver passes
`from_cwd=False` by default; passing `--workdir PATH` flips it back on with
`PATH` as the source directory.

This single function is the canonical resolution path for Seamless service
configs. All consumers — action wrappers, `seamless-service-resolve`,
`seamless-service-ps`, and any future tool — must go through this function or
one of the `configure_*` functions it calls. **Do not reimplement key, path,
or host derivation.** See `seamless-service-resolve-contract.md` for the
rationale (extractor model — single source of truth, drift impossible).

Tunnel note: for tunnelled connections, the tunnel process is monitored by the launcher
and tears itself down when the remote process dies. seamless-service-stop does not need
to manage tunnels explicitly.

---

## seamless-service-resolve

Agent-friendly counterpart to the human action wrappers. Translates
Seamless-level arguments into rhl-level identifiers (key, ssh_hostname,
workdir, log_path) without performing any side effects. Used both by agents
directly and by humans/scripts that need the resolved values without acting
on them.

```
seamless-service-resolve --service <svc> [common flags] [--workdir PATH]
```

Behaviour:
- Identical argument parsing to the action wrappers (via `_args.make_parser`,
  `agent_mode=True`)
- Calls `_dispatch.resolve(args, from_cwd=False)` — or `from_cwd=True` with
  `args.workdir` as the source if `--workdir` was given
- Emits the resolved fields as JSON to stdout
- No side effects; safe to call from anywhere, any number of times

Output schema:

```json
{
  "key": "hashserver-MYCLUSTER-rw-myproject--STAGE-fingertip",
  "ssh_hostname": "frontend.lab",
  "workdir": "/data/buffers/myproject/STAGE-fingertip",
  "log_path": "~/.remote-http-launcher/server/hashserver-MYCLUSTER-rw-myproject--STAGE-fingertip.log",
  "service": "hashserver",
  "cluster": "MYCLUSTER",
  "mode": "rw",
  "project": "myproject",
  "subproject": null,
  "stage": "fingertip",
  "substage": null,
  "queue": null
}
```

Contract: see `seamless-service-resolve-contract.md`. The tool is an
**extractor**, not a synthesizer — its output reflects what the
currently-installed Seamless runtime would compute. The required disclaimer
(quoted in the contract document) must appear in the tool's `--help` output.

Agent-friendliness specifics:
- All inputs come from `(argv, environ)`. The tool does NOT consult
  `os.getcwd()` to discover seamless.yaml / seamless.profile.yaml.
- `--workdir PATH` (optional) opts in to the human-style cwd defaulting:
  load config files from `PATH` instead of cwd.
- `SEAMLESS_CACHE` environment variable is respected as documented in
  seamless-config (selects the `__SEAMLESS_CACHE__` pseudo-cluster). Unset
  to opt out.
- No side effects; the tool is a pure read.

Internal reuse: each action wrapper (`stop`, `rm`, `logs`, `inspect`,
`clear`, `ps`) calls `_dispatch.resolve(args, from_cwd=True)` directly — same
library function, just without going through the resolver CLI. The resolver
CLI exists for external callers (agents, scripts, debugging) that want the
resolved values without taking an action.

---

## seamless-service-stop

```
seamless-service-stop --service <svc> [common flags]
seamless-service-stop --cluster <cluster>            # cluster-wide
```

Single-service mode:

```python
key, ssh_host, _ = resolve(args)
run_remote(ssh_host, "rhl-stop", key)
```

Cluster-wide mode (`--cluster <C>` without `--service`): enumerate keys for
the cluster via `rhl-ps --json`, then batch them to a single `rhl-stop`
invocation. The seamless-specific knowledge (which keys belong to cluster
C — anchored on the mode segment that always follows the cluster name in the
templates) lives here, in the wrapper, not in `rhl-stop`.

```python
ssh_host = ssh_hostname_for_cluster(args.cluster)
rows = json.loads(run_remote_capture(ssh_host, "rhl-ps", "--json"))
keys = [r["key"] for r in rows if belongs_to_cluster(r, args.cluster)]
if keys:
    run_remote(ssh_host, "rhl-stop", *keys)
```

`belongs_to_cluster` prefers the meta block when present (`r["meta"]["cluster"] == args.cluster`)
and falls back to anchored prefix matching on the key for legacy JSONs without
meta.

---

## seamless-service-rm

```
seamless-service-rm --service <svc> [common flags] [--client] [--server]
seamless-service-rm --cluster <cluster>           [--client] [--server]   # cluster-wide
```

Default (no side flag): removes both sides.

Single-service mode:

```python
key, ssh_host, _ = resolve(args)
if do_server:
    run_remote(ssh_host, "rhl-rm", "--server", key)
if do_client:
    run_local("rhl-rm", "--client", key)   # always local, never SSH
```

Cluster-wide mode: same enumeration pattern as seamless-service-stop, but
applied to both sides:

```python
ssh_host = ssh_hostname_for_cluster(args.cluster)
if do_server:
    server_rows = json.loads(run_remote_capture(ssh_host, "rhl-ps", "--json"))
    server_keys = [r["key"] for r in server_rows if belongs_to_cluster(r, args.cluster)]
    if server_keys:
        run_remote(ssh_host, "rhl-rm", "--server", *server_keys)
if do_client:
    client_rows = json.loads(run_local_capture("rhl-ps", "--client", "--json"))
    client_keys = [r["key"] for r in client_rows if belongs_to_cluster(r, args.cluster)]
    if client_keys:
        run_local("rhl-rm", "--client", *client_keys)
```

`run_local` is just `subprocess.run` without SSH, regardless of cluster type.

---

## seamless-service-logs

```
seamless-service-logs --service <svc> [common flags] [--tail N]
```

```python
key, ssh_host, _ = resolve(args)
cmd = ["rhl-logs", key]
if args.tail:
    cmd += ["--tail", str(args.tail)]
run_remote(ssh_host, *cmd)
```

---

## seamless-service-inspect

```
seamless-service-inspect --service <svc> [common flags]
```

```python
key, ssh_host, _ = resolve(args)
run_remote(ssh_host, "rhl-inspect", key)
```

---

## seamless-service-clear

```
seamless-service-clear --service <svc> [common flags]
```

Only meaningful for hashserver and database, which have persistent data in their
workdir. For jobserver, daskserver, and pure-daskserver the workdir is /tmp — the
tool errors with a clear message rather than clearing /tmp.

```python
key, ssh_host, config = resolve(args)
workdir = config["workdir"]
if workdir == "/tmp":
    sys.exit(f"seamless-service-clear: {args.service} uses /tmp as workdir; nothing to clear")
run_remote(ssh_host, "rhl-clear", workdir)
```

The `workdir` from `configure_hashserver()` is `<bufferdir>/<project>[/<subproject>][/STAGE-<stage>]`
and from `configure_database()` is `<database_dir>/<project>[/<subproject>][/STAGE-<stage>]` —
exactly the right paths to pass to rhl-clear.

---

## pyproject.toml additions (seamless-config)

Switch from `script-files` to `[project.scripts]` entry points for the new tools
(keeping `bin/seamless-init` as a script-file):

```toml
[project.scripts]
seamless-service-resolve = "seamless_config.service.resolve:main"
seamless-service-ps      = "seamless_config.service.ps:main"
seamless-service-stop    = "seamless_config.service.stop:main"
seamless-service-rm      = "seamless_config.service.rm:main"
seamless-service-logs    = "seamless_config.service.logs:main"
seamless-service-inspect = "seamless_config.service.inspect:main"
seamless-service-clear   = "seamless_config.service.clear:main"
```

---

# Part 2a: seamless-service-ps

A composer of `rhl-ps` and `rhl-ps-persistent` that joins process state with
persistent state for a unified human-readable cluster overview. Lives
client-side. It cannot live server-side: that would require Seamless on the
server *and* whitelisting `seamless-service-ps` in `rhl-guard`, both ruled out
by the project contract (see `seamless-service-resolve-contract.md`,
"Server-side nuance").

`seamless-service-ps` is structurally a script: rhl-ps + analysis +
rhl-ps-persistent + analysis + formatting. It contains no Seamless-specific
key/path logic of its own beyond reading the cluster YAML and dispatching;
the meta block (§11) and `seamless-service-resolve` carry the structured
fields it needs.

## Location

`seamless_config/service/ps.py`

## CLI

```
seamless-service-ps [--cluster CLUSTER] [--service SVC] [--project PROJ]
                    [--status STATE] [--client | --server | --all-clusters]
                    [--persistent] [--json]
```

Mode flags (mutually exclusive):
- `--client` (default): list client-side connections from local
  `~/.remote-http-launcher/client/`. Instant, no SSH.
- `--server`: SSH to the cluster's frontend and list server-side services
  on that cluster. Cluster comes from `--cluster` or, absent that, from
  `seamless.profile.yaml` in cwd.
- `--all-clusters`: server-side, fanned out across every cluster YAML the
  client knows about. Adds a `CLUSTER` column. Best-effort: per-host SSH
  timeout ~10 s; unreachable frontends produce a warning row but do not
  fail the command.

Other flags:
- `--persistent`: add persistent-state columns (state, size). Implicit when
  `--service hashserver` or `--service database` is the only filter and the
  user is on a server mode. Mostly used for false-pass debugging.
- `--service`, `--project`, `--status`: row filters.
- `--json`: emit raw JSON (one object per row) instead of a table — for
  scripting and agent consumption.

## Implementation: three-step composition

1. **Resolve scope.** Load cluster YAML(s) via the same library function
   `seamless-service-resolve` exposes. From each cluster yield: `ssh_hostname`,
   `bufferdir`, `database_dir`. (For `--client`, no SSH targets needed.)

2. **Process state.** One SSH call per cluster (or local invocation for
   `--client`): `rhl-ps --json [--client]`. Returns one object per JSON file
   on disk, each carrying the full state including the `meta` block (added in
   §11). The meta block gives `(service, cluster, project, [subproject],
   [stage], [substage], mode)` directly — no key parsing.

3. **Persistent state** (when requested or implicit). One SSH call per
   persistent service type. With currently two persistent types (this is not
   expected to grow significantly):
   - hashserver: `rhl-ps-persistent --json --level 2 <bufferdir>`
   - database: `rhl-ps-persistent --json --level 2 --file seamless.db <database_dir>`
   Two calls total per cluster, regardless of how many projects/stages exist.
   The seamless-specific knowledge of "which file matters per service" lives
   here in seamless-service-ps, not in `rhl-ps-persistent`.

4. **Join + format.** Union the rows from (2) and (3), keyed by
   `(service, project, [subproject], [stage])`. Each output row presents
   process state and persistent state side by side. Aligned table for
   humans; pass through raw JSON with `--json`.

## Output (process + persistent)

```
$ seamless-service-ps --server --persistent --cluster MYCLUSTER
SERVICE      PROJECT      STAGE       PROCESS    PORT     PERSISTENT  SIZE
hashserver   myproject    -           running    10501    populated   42 MB
hashserver   myproject    fingertip   stale      -        populated   17 MB
hashserver   oldproject   -           absent     -        empty       -
database     myproject    -           running    10502    populated   3 MB
database     myproject    fingertip   absent     -        populated   1 MB
jobserver    myproject    fingertip   failed     -        n/a         -
```

Process-state values: `running` | `starting` | `failed` | `stale` | `absent`
Persistent-state values: `populated` | `empty` | `absent` | `n/a` (for
non-persistent service types)

## Why no registry is needed

The union of `rhl-ps` rows + `rhl-ps-persistent` rows IS the enumeration. If
something exists on disk (process state or data), it shows up; otherwise it
doesn't. seamless-service-ps does not maintain or consult any project/stage
registry.

The walker (`rhl-ps-persistent`) cannot distinguish "data from current
Seamless layout" from "data from a previous layout" or "data from another
user's project under the same root". It reports honestly what is on disk;
interpretation is the operator's job.

## Use case: false-pass debugging

A test passes because cached buffers or DB rows still exist from a prior run,
not because the current computation actually succeeded. The skill
`seamless-remote-debugging` warns about this scenario; this view is the
operator's primary tool for reasoning about it:

```
$ seamless-service-ps --persistent --project myproject
... shows populated bufferdirs and seamless.db files ...
$ seamless-service-clear --service hashserver --project myproject
$ seamless-service-clear --service database --project myproject
$ # re-run test on cold cache
```

---

## Verification (Part 2: seamless-config layer)

1. `pip install -e .` in seamless-config — verify entry points registered.
2. `seamless-service-resolve --service hashserver --cluster MYCLUSTER --project myproject`
   — produces JSON with `key`, `ssh_hostname`, `workdir`, `log_path`, plus
   structured fields. Run from `/tmp` (no seamless.yaml in cwd) — should
   still succeed. Re-run with `--workdir /path/to/project` — should pick up
   that project's defaults.
3. In a directory with seamless.yaml + seamless.profile.yaml:
   `seamless-service-inspect --service hashserver` — should SSH and print JSON.
4. `seamless-service-stop --service hashserver` — should SSH and stop,
   printing escalation.
5. `seamless-service-clear --service jobserver` — should error with /tmp message.
6. `seamless-service-ps` — should list client-side connections, no SSH.
7. `seamless-service-ps --server --persistent --cluster MYCLUSTER` — should
   SSH once for `rhl-ps`, twice for `rhl-ps-persistent` (hashserver +
   database roots), join, and print combined table.

---

## Verification (Part 1: rhl-* layer)

1. `pip install -e .` in remote-http-launcher — verify new entry points are
   registered.
2. `rhl-ps`, `rhl-ps-persistent --help`, `rhl-logs --help`,
   `rhl-inspect --help`, `rhl-stop --help`, `rhl-rm --help`,
   `rhl-clear --help` — verify help text and exit 0.
3. `rhl-ps --json` — verify NDJSON output including `meta` block on JSONs
   written by the new launcher.
4. `rhl-ps-persistent /tmp/empty-dir --level 0` — verify `empty` state.
5. `rhl-ps-persistent /tmp/populated-dir --level 1 --json` — verify per-row
   JSON with `state`, `size`, `modified`.
6. `rhl-guard` directly (no SSH_ORIGINAL_COMMAND) — verify explanatory message.
7. `pytest tests/` — verify `test_ssh_guard_integration` passes with `rhl-ps`
   probe.
8. Verify old names are gone: `which rhl-kill-service` should fail.
9. After `rhl-rm <key>` for a previously-launched service, verify the log
   file at `~/.remote-http-launcher/server/<key>.log` is still on disk.