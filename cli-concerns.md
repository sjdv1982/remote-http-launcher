# CLI concerns for human-facing support

These issues have been low-priority while the tools were used exclusively by agents
reading the source. Adding huma\n support makes them worth resolving first.

---

## 1. `rhl-kill-service` sends SIGHUP, not SIGTERM

The command is named "kill" but delivers SIGHUP (signal 1), which Unix convention
treats as "reload config" or "hangup", not "terminate". A human operator will expect
"kill" to stop the process.

**Question:** Do the Seamless services (hashserver, seamless-database, etc.) treat
SIGHUP as a clean shutdown? If yes, document that explicitly. If no, change to SIGTERM.

**Response:** The tool should send do it gracefully, sending progressively stronger
signals and then inspecting status codes. SIGHUP => SIGTERM => KILL . 
Please verify the source code of the current services to check that SIGHUP does the right thing.

---

## 2. `rhl-rm-state` defaults to removing both sides, which is surprising on a remote server

With no flags, `rhl-rm-state <key>` removes both the server JSON and the client JSON.
When run on a remote frontend (the common case), the client directory doesn't exist
there — the tool silently succeeds but does nothing useful on the client side.

**Options:**
- Require an explicit flag (no default, error if neither `--client` nor `--server` given).
- Default to `--server` only, since that's the natural context for a guarded SSH session.

**Response:**
This is closely related to execution semantics, defer to point 8.

---

## 3. `rhl-restart-cluster` leaves client state orphaned

The tool kills and removes server-side state for all services in a cluster, but does
nothing about client-side connection files. The README's workaround is a grep-xargs
pipeline that humans will not remember or get right.

**Suggestion:** Add a `--client` flag that also removes matching client-side JSONs.
Since client state lives locally (not on the remote), this flag would only be useful
when `rhl-restart-cluster` is run locally — that's fine, it's still better than
the manual pipeline.

**Response:**
This is closely related to execution semantics, defer to point 8.

---

## 4. `rhl-cat-log` dumps the entire log

No options to limit output. For any non-trivial service the log will be long, and
reading it over SSH compounds the problem. A human debugger's first instinct is to
look at the tail.

**Suggestion:** Add `--tail N` (default 50? or no default — show all unless flag given).
`--lines` is an alias some users expect.

**Response:** Agreed.
---

## 5. `rhl-clear-buffer` silently skips subdirectories, but documentation doesn't say so

The implementation removes only files at the top level of the given path — subdirectories
are left intact. This is intentional (preserving the project/stage tree structure), but:
- The README table just says "Remove all files inside a buffer directory" — ambiguous.
- A human expecting a recursive wipe will be confused when subdirectory content remains.

**Suggestion:** Clarify in the usage string and documentation: "removes files (not
subdirectories) directly inside `<path>`". Consider whether `--recursive` is ever needed.

**Response:**
This is closely related to key semantics, defer to point 9.

---

## 6. No `--help` on any tool

All tools print a usage line to stderr only on wrong invocation. Running
`rhl-cat-log --help` exits with an error rather than showing help. This is fine for
agents but hostile for humans.

**Suggestion:** Intercept `-h` / `--help` on every tool and print the usage + a short
description, then exit 0. The descriptions already exist as module docstrings — they
just need to be wired up.

**Response:**
Agreed.

---

## 7. `rhl-guard` is opaque when run interactively

If a human runs `rhl-guard` directly in a shell (e.g. to test or debug the installation),
it fails with "empty command (interactive session not allowed)" — no hint that it is an
SSH guard or how it works.

**Suggestion:** Detect the absence of `SSH_ORIGINAL_COMMAND` and print a brief
explanatory message before exiting, e.g.:
  "rhl-guard: this program is an SSH guard and must be invoked via SSH.
   Set SSH_ORIGINAL_COMMAND to test a specific command."

**Response** Yes, but not quite. Intended usage is via .ssh/authorized_keys. 
This must be explained clearly both in the explanatory message and in the documentation.

---

## 8. Execution semantics are confusing to a human user

It is not clear if a particular tool:
- Runs server-side all by itself (i.e. launching SSH commands internally), like `seamless-init` does
- Is supposed to (or may) run server-side, wrapped in an ssh command
- Is supposed to (or may) run client side

## 9. `key` is very un-ergonomic/unintuitive for humans

Normal usage of `seamless-init` has the key synthesized SEAMLESS_CACHE OR from seamless.X.yaml
and modified with --project, --stage etc. This is the human-expected way to deal with keys. 
Obviously, for the rhl helpers, --service needs to be added, and also a --pure-dask since that modified the key too (TASK: check the seamless-config source code if there are more such modifiers. /TASK). For agentic use, an explicit synthesized key (the current usage) can be supported via --key (mutually exclusive with --project, --service etc.)


## 10. The lifecycle contract of each of the tools onto the services is not clear

An existing service can be dead (not existing), running (or starting up), stale, or persisted. 
Stale means that the JSON exists the PID is a dead process. Removing this JSON (and the logs) cleans up all records of the dead service. HOWEVER: for hashserver and database, they don't become dead but "persisted": their state is still recorded in the buffer dir / seamless.db . 
The lifecycle needs to be made explicit in documentation, and the role of each tool explained:
`rhl-kill` to change from running to stale. `rhl-rm` to change from stale to dead/persisted. `rhl-clear` to change from persisted to dead. TASK: the tool names are the current ones, but this is up for discussion. Compare with the Docker CLI for inspiration. /TASK . 
`rhl-ls` must be brought in line with this and with point 9. By default, it should return a table with service name, project, stage, etc. , and status (lifecycle stage). This table can be filtered with --service, --project, --status etc. To get the current behavior, use --key to get the keys and --no-status to omit the status.