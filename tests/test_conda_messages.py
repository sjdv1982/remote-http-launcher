import json
import subprocess
from types import SimpleNamespace

from remote_http_launcher import (
    CONDA_ACTIVATE_NOT_INITIALIZED,
    _conda_env_missing_message,
    _conda_env_exists,
    _repair_conda_cache_after_activation_error,
    _conda_unavailable_message,
)


class _Observer:
    def __init__(self):
        self.details = []
        self.commands = []

    def on_detail(self, message):
        self.details.append(message)

    def on_command(self, host, command, full_command):
        self.commands.append((host, command, full_command))

    def on_command_done(self, host, command):
        self.commands.append((host, f"done:{command}", None))


class _Executor:
    host = "LocalExecutor"

    def __init__(self, refreshed_cache, helper_returncode=0):
        self._conda_cache = {"envs": ["/home/agent/miniforge3/envs/old"]}
        self._conda_source = None
        self._conda_checked = True
        self._refreshed_cache = refreshed_cache
        self._helper_returncode = helper_returncode
        self.helper_calls = []
        self.observer = _Observer()

    def _try_helper(self, argv):
        self.helper_calls.append(argv)
        return SimpleNamespace(returncode=self._helper_returncode, stderr="")

    def _read_conda_cache(self):
        return self._refreshed_cache


def test_conda_env_exists_refreshes_cache_on_cache_miss():
    executor = _Executor(
        {"envs": ["/home/agent/miniforge3/envs/seamless"], "conda_source": "/conda.sh"}
    )

    assert _conda_env_exists(executor, "seamless") is True
    assert ["rhl-cache-conda"] in executor.helper_calls
    assert executor._conda_source == "/conda.sh"


def test_conda_env_exists_returns_false_when_refresh_still_misses():
    executor = _Executor({"envs": ["/home/agent/miniforge3/envs/other"]})

    assert _conda_env_exists(executor, "seamless") is False
    assert ["rhl-cache-conda"] in executor.helper_calls


def test_repair_conda_cache_after_activation_error_refreshes_null_source():
    executor = _Executor(
        {
            "envs": ["/home/agent/miniforge3/envs/seamless"],
            "conda_source": "/home/agent/miniforge3/etc/profile.d/conda.sh",
        }
    )
    executor._conda_cache = {
        "envs": ["/home/agent/miniforge3/envs/seamless"],
        "conda_source": None,
    }

    assert (
        _repair_conda_cache_after_activation_error(
            executor, f"\n{CONDA_ACTIVATE_NOT_INITIALIZED}\n"
        )
        is True
    )
    assert ["rhl-cache-conda"] in executor.helper_calls
    assert (
        executor._conda_cache["conda_source"]
        == "/home/agent/miniforge3/etc/profile.d/conda.sh"
    )


def test_repair_conda_cache_after_activation_error_ignores_other_logs():
    executor = _Executor(
        {
            "envs": ["/home/agent/miniforge3/envs/seamless"],
            "conda_source": "/home/agent/miniforge3/etc/profile.d/conda.sh",
        }
    )

    assert _repair_conda_cache_after_activation_error(executor, "different error") is False
    assert executor.helper_calls == []


def test_conda_cache_refresh_is_recorded_as_helper_command(monkeypatch):
    import remote_http_launcher as rhl

    observer = _Observer()
    executor = rhl.LocalExecutor(observer)
    executor._conda_cache = {"envs": ["/home/agent/miniforge3/envs/old"]}
    executor._conda_source = None
    executor._conda_checked = True

    def fake_run(argv, **kwargs):
        if argv == ["rhl-cache-conda"]:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if argv == ["rhl-conda-info"]:
            payload = {
                "envs": ["/home/agent/miniforge3/envs/seamless"],
                "conda_source": "/home/agent/miniforge3/etc/profile.d/conda.sh",
            }
            return subprocess.CompletedProcess(
                argv, 0, stdout=json.dumps(payload), stderr=""
            )
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(rhl.subprocess, "run", fake_run)

    assert _conda_env_exists(executor, "seamless") is True
    assert (
        "LocalExecutor",
        "rhl-cache-conda",
        "rhl-cache-conda",
    ) in observer.commands


def test_missing_conda_env_message_mentions_cache_when_cache_was_used():
    executor = SimpleNamespace(
        host="LocalExecutor",
        _conda_cache={"envs": ["/home/agent/miniforge3/envs/other"]},
    )

    assert _conda_env_missing_message(executor, "seamless") == (
        "Conda environment 'seamless' is not listed in the conda cache for "
        "LocalExecutor after running rhl-cache-conda."
    )


def test_missing_conda_env_message_without_cache_names_target():
    executor = SimpleNamespace(host="SSHExecutor[cluster]", _conda_cache=None)

    assert (
        _conda_env_missing_message(executor, "seamless")
        == "Conda environment 'seamless' does not exist on SSHExecutor[cluster]."
    )


def test_conda_unavailable_message_names_target():
    executor = SimpleNamespace(host="LocalExecutor")

    assert _conda_unavailable_message(executor) == (
        "Conda environment requested but conda is not available on LocalExecutor."
    )
