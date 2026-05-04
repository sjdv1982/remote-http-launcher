# remote-http-launcher

`remote-http-launcher` is a command-line tool and Python library for launching, supervising, and reconnecting to long-running HTTP services — either locally or on remote hosts via SSH. It is driven by a single YAML configuration file that describes the service to launch, and it manages all process lifecycle state through JSON files on the client and server side.

Although it originated in the [Seamless](https://github.com/sjdv1982/seamless) scientific-workflow framework, `remote-http-launcher` is fully generic: it can launch any HTTP service that follows its status-file protocol.

## How it works

Given a YAML config, the launcher follows a deterministic sequence:

1. **Check for an existing local connection file** (`~/.remote-http-launcher/client/<key>.json`). If one exists and the service responds to the configured handshake, the launcher exits immediately — the service is already running.

2. **Check the remote (or local) server directory** (`~/.remote-http-launcher/server/<key>.json`):
   - If the remote JSON shows `"status": "running"`, the launcher verifies the port and handshake, then writes the local connection file.
   - If it shows `"status": "starting"`, the launcher monitors until the service comes up (or times out).
   - If the JSON file is absent or stale, the launcher starts the service.

3. **Launch the service** as a daemonic subprocess (locally or over SSH), writing a server-side JSON file with the PID, workdir, and `"status": "starting"`. The path to this JSON file is passed to the service via the `{status_file}` template variable in the `command` field.

4. **Optionally create an SSH tunnel** forwarding the remote port to a local port, with a background monitor that tears down the tunnel when the remote process exits.

5. **Write the local connection file** containing the hostname and port (local or tunneled) for consumption by client code.

## Status-file protocol

`remote-http-launcher` imposes a specific contract on any service it launches. The service **must**:

1. Read the JSON status file whose path was passed to it (via the `{status_file}` template variable in the command).
2. Acquire a free port and start listening on it.
3. Update that same JSON file: set `"status"` to `"running"` and add a `"port"` field with the chosen port number.

If the service fails to start, it should set `"status"` to `"failed"`.

The launcher monitors the status file after starting the process. If the file is not updated within roughly one minute (and the process has exited), the launcher treats the launch as failed.

This is the only requirement `remote-http-launcher` places on the launched service — beyond this protocol, the service can be anything that speaks HTTP.

## Configuration

The YAML configuration file supports the following fields:

| Field | Required | Description |
|-------|----------|-------------|
| `workdir` | yes | Working directory on the (remote) host |
| `key` | yes | Template string evaluated to a filename used for the JSON state files |
| `command` | yes | Template string for the bash command to launch the service |
| `hostname` | no | Target HTTP hostname or IP; omit to run locally |
| `ssh_hostname` | no | SSH host to connect to (defaults to `hostname`) |
| `network_interface` | no | Interface the service binds to (default: `localhost`) |
| `tunnel` | no | Create an SSH tunnel for the service port (default: `false`) |
| `handshake` | no | HTTP path (and optional query parameters) for a health-check GET request |
| `conda` | no | Conda environment to activate before launching the command |
| `file_parameters` | no | Arbitrary parameters written into the server-side JSON file |
| `meta` | no | Opaque caller-provided JSON metadata copied into client/server state |

The `key` and `command` fields are Python f-string templates evaluated against the full config namespace.

### Example

```yaml
workdir: /home/user/my-service-data
hostname: my-server.example.com
key: 'myservice-{workdir.strip("/").replace("/", "--")}'
command: >-
  myservice --port-range {port_start} {port_end}
  --status-file {status_file}
  --host {config['network_interface']}
  --timeout {timeout}
  {workdir}
network_interface: "0.0.0.0"
handshake: healthcheck
conda: myservice-env
timeout: 600
port_start: 10000
port_end: 19999
```

A JSON schema is included at [config.schema.yaml](config.schema.yaml) for validation and editor support.

## Installation

```bash
pip install remote-http-launcher
```

The only runtime dependency is PyYAML.

## Usage

```bash
# Launch (or reconnect to) the service described in config.yaml
remote-http-launcher config.yaml

# Override the client connection directory
remote-http-launcher config.yaml --connection-dir /tmp/my-connections

# Print the evaluated command without launching
remote-http-launcher config.yaml --dry-run
```

### Python API

```python
from remote_http_launcher import run

result = run({
    "workdir": "/home/user/data",
    "key": "my-service",
    "command": "myservice --port-range 10000 19999 --status-file {status_file} {workdir}",
    "handshake": "healthcheck",
})
print(result["hostname"], result["port"])
```

## SSH Guard

`remote-http-launcher` ships an SSH guard (`rhl-guard`) that restricts what commands can be run on the remote server under the service user account. When installed, only the specific command patterns sent by the launcher itself and a set of named helper programs are permitted — naked shell commands such as `pkill`, `rm -rf`, or arbitrary `python3 -c` are rejected.

### How it works

The SSH `command=` option in `authorized_keys` forces every incoming SSH session through `rhl-guard`. The guard reads `SSH_ORIGINAL_COMMAND`, validates it against a whitelist, and either `exec`s the command or exits with an error. Interactive sessions (no `SSH_ORIGINAL_COMMAND`) are always rejected.

The whitelist covers:

- `bash -lc` commands matching the exact patterns that `remote-http-launcher` generates: `ps -p <int> -o pid=` and Python heredoc scripts bearing the launcher's `__RHL_REMOTE_SCRIPT__` sentinel.
- Inside Python heredoc launch scripts, the guard verifies that the service binary is one of the tools listed in `ssh_guard/tools.yaml` (a vendored copy of the Seamless `tools.yaml`).
- Any `rhl-*` helper command installed by this package (see below).
- Conda probe fallbacks (`command -v conda`, `cat ~/.bashrc`, etc.) — only reached on servers where the conda cache has not been primed.

Direct process-management commands such as `kill -1 <pid>` are rejected by
the guard. Use helpers such as `rhl-stop <key>` instead.

### Installation

On the remote server, add to `~/.ssh/authorized_keys`:

```
command="rhl-guard" ssh-rsa AAAA... your-key-comment
```

If you use a non-standard tools.yaml, set `RHL_TOOLS_YAML=/path/to/tools.yaml` in the server environment.

Running `rhl-guard` directly prints an installation-oriented error explaining
that it must be invoked by SSH. To test one guarded command locally on the
server:

```bash
SSH_ORIGINAL_COMMAND="rhl-ps" rhl-guard
```

### Conda cache (guarded servers)

When the guard is active, the launcher replaces the individual conda probe SSH commands with a single cached read. Prime the cache once after installing the guard:

```bash
ssh <remote_host> rhl-cache-conda
```

Re-run this if the conda environment changes. On unguarded servers the launcher falls back to its original probe behavior automatically — `rhl-cache-conda` is a no-op there too.

## Lifecycle States

Launcher state normally moves through these states:

| State | Meaning |
|-------|---------|
| `absent` | No launcher state or persistent directory exists for the service |
| `starting` | The process exists but has not yet reported a listening port |
| `running` | The service reported a port and the process appears alive |
| `failed` | Startup failed or a dry-run/server row explicitly records failure-like state |
| `stale` | A non-persistent service has dead process state left for post-mortem inspection |
| `persistent` | Filesystem-backed service data exists even if no process is running |

The `stale` window is intentional for non-persistent services: inspect logs
before removing the server JSON with `rhl-rm`. Persistent state can also make
tests falsely pass because old data remains even when no process is alive; use
`rhl-ps-persistent` or the Seamless service wrappers to inspect and clear it.

## JSON State Schema

Server JSON files live under `~/.remote-http-launcher/server/<key>.json` and
client files under `~/.remote-http-launcher/client/<key>.json`. Readers tolerate
older rows without `meta`. When present, `meta` is copied from the caller and is
opaque to `remote-http-launcher`; the launcher does not validate service,
cluster, project, or stage semantics.

## Helper commands (`rhl-*`)

Installing `remote-http-launcher` adds a set of server-side helper programs that perform specific, safe operations on launcher state. These are the commands agents and operators should use instead of raw `kill`, `rm`, or shell loops.

| Command | Runs on | Purpose |
|---------|---------|---------|
| `rhl-guard` | server | SSH guard entry point; validates `SSH_ORIGINAL_COMMAND` before exec |
| `rhl-cache-conda` | server | Discover conda setup and write `~/.remote-http-launcher/conda-setup.json` |
| `rhl-ps [--client]` | client/server | List process state rows; client mode lists local connection state |
| `rhl-ps-persistent <path>` | server | Report absent, empty, or populated filesystem-backed state |
| `rhl-stop <key>` | server | Stop service processes without deleting JSON state |
| `rhl-rm <key> [--client] [--server]` | client/server | Remove launcher JSON state files while leaving logs on disk |
| `rhl-logs <key> [--tail N]` | server | Print the stdout/stderr log for a service |
| `rhl-inspect <key>` | server | Pretty-print the server state JSON |
| `rhl-clear <path>` | server | Remove direct children of a validated persistent directory |

`rhl-clear` and `rhl-ps-persistent` validate that target paths are absolute and
not system directories before touching anything.

All state-file helpers respect the `REMOTE_HTTP_LAUNCHER_DIR` environment variable (same as the launcher).

Cluster-wide Seamless operations are intentionally not implemented by `rhl-*`
helpers. Use `seamless-service-stop`, `seamless-service-rm`, and
`seamless-service-ps` from `seamless-config`; those tools resolve Seamless
cluster/project/stage semantics on the client side and then dispatch safe
`rhl-*` operations.

## CLI scripts

Installing `remote-http-launcher` also provides:

- `remote-http-launcher` — main launcher CLI
- All `rhl-*` helpers listed in the table above
