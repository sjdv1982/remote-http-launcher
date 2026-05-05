# seamless-service-resolve: contract

## Purpose

`seamless-service-resolve` is a CLI that translates Seamless-level inputs (service,
cluster, project, stage, ...) into rhl-level identifiers (key, ssh_hostname, workdir,
log path) that can be passed to the `rhl-*` helpers. It is the agent-friendly
counterpart to the human-facing `seamless-service-*` action wrappers: same resolution
logic, different ergonomics (explicit args, no cwd defaults, JSON out).

## The contract: extractor, not synthesizer

`seamless-service-resolve` reports what the **currently-installed client-side Seamless
runtime** would compute for the given inputs. It is a thin shell over
`seamless_config._dispatch.resolve()` and the `configure_*` functions in
`seamless_config.tools` — the same code paths used by `seamless-run` and the human
`seamless-service-*` wrappers.

Outputs are not covered by a stability guarantee. Key format, workdir construction
(path components, separators, conditional segments), and host-selection logic (which
frontend, which ssh_hostname) tend to remain stable in practice, but the project does
not formally commit to that — they may change between Seamless versions without
prior notice.

The resolver tracks the runtime, not a published spec. Drift between resolver output
and runtime behavior is **structurally impossible** because they share the same
implementation.

This is a deliberate design choice. The alternative — a synthesizer that implements an
independent, version-stamped specification — was considered and rejected for now
because:

1. There are no external consumers that require spec stability.
2. Agents need *predictive fidelity* ("where will Seamless put this?"), not normative
   stability ("where does the v1.x spec say this should go?").
3. Extractor → synthesizer is an additive upgrade path; the reverse is not.

## What this commits the project to

- **No spec to maintain.** No version field, no compatibility window, no schema
  document.
- **No drift surface.** Resolver and runtime are wired to the same implementation.
- **Documentation teaches a workflow, not a format.** The agentic instruction is
  "call `seamless-service-resolve`, read the named fields, pass them to `rhl-*`" —
  never "the key is constructed as `<service>-<cluster>-...`".
- **No stability guarantee for external consumers.** Tools outside this project that
  depend on resolver output must either pin a Seamless version or shell out on every
  invocation rather than caching results.

## Required disclaimer

The resolver's `--help` and any documentation referring to its output must include:

> `seamless-service-resolve` reports what the currently-installed Seamless runtime
> would compute for the given inputs. Outputs are not part of any stable contract:
> keys, workdir paths, and host-selection logic may change between Seamless versions.
> Tools that need stability should pin a Seamless version, or shell out to this
> command on every invocation rather than caching its outputs.

This disclaimer is the honest substitute for a written spec. Without it, the implicit
contract becomes invisible and consumers will assume more stability than is offered.

## Server-side nuance: Seamless is not required on the server

Seamless does not need to be — and is never required to be — installed on the machine
that runs the launched services (hashserver, seamless-database, seamless-jobserver,
seamless-dask-wrapper). The server-side requirements are:

1. The service binary itself (e.g. `hashserver`). Always required.
2. Optionally, `remote-http-launcher` (which provides the `rhl-*` helpers). Required
   only if the SSH key in `authorized_keys` is guarded by `rhl-guard`. Without the
   guard, the client-side launcher uses fallback code that does not require any
   server-side `rhl-*` infrastructure.

This strengthens the extractor model rather than complicating it: there is exactly one
place where Seamless's resolution logic lives — the **client-side** Seamless install.
The launcher takes its keys/paths/hosts from there and writes them onto the server.
There is no second source of truth on the server that could disagree about how a
service should be named or where its files should land.

### Caveat: divergent Seamless inside service conda envs

A separate concern arises when a service (jobserver, daskserver queue) activates a
conda environment that itself contains a Seamless install — and that install's
`seamless-config` would compute different identifiers than the client's. In that case,
the worker process may fail to locate services it depends on, because it is operating
on its own divergent view of how things should be named or where files should live.

This is a **configuration problem** (mismatched conda envs across machines), not a
resolver problem. The resolver continues to faithfully report what the client-side
Seamless would do — which is what the launcher actually writes. A worker disagreeing
with the launcher is a worker-config bug, orthogonal to service launching itself and
outside the resolver's scope.

## When to reopen this decision

The recommendation should be revisited if any of the following become true:

- A second tool outside `seamless-config`, `remote-http-launcher`, or
  `seamless-remote-debugging` starts depending on key/workdir format.
- Seamless gains a documented public API surface that includes service-resolution
  semantics.
- Multiple Seamless versions need to be supported in production simultaneously by the
  same agent/skill (cross-version debugging).

Promoting the resolver from extractor to synthesizer at that point is additive: write
the spec down, add equivalence tests against the runtime, stamp a version. Nothing
about choosing extractor today forecloses that path.
