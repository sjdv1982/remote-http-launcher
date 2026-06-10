# Guarded localhost integration tests

The SSH guard integration tests in this directory expect the local Seamless
test cluster profile to use `local2`.

`local2` is defined in `~/.seamless/clusters.yaml` and connects back to the
same machine through the `localhost_guard` SSH host. That host should use the
test key from `~/.ssh/id_test`, and the matching public key in
`~/.ssh/authorized_keys` must be forced through the guard:

```text
command="/home/agent/miniforge3/envs/seamless1/bin/rhl-guard" <contents of ~/.ssh/id_test.pub>
```

You can prime the guarded conda cache before running Seamless tests:

```bash
ssh localhost_guard rhl-cache-conda
```

The launcher also refreshes this cache automatically when a requested conda
environment is missing from the cached environment list. That refresh is
dispatched as the guarded `rhl-cache-conda` helper, so debug logs show it
as a normal helper command.

The local `seamless.profile.yaml` included here records the required profile
selection for this integration setup.
