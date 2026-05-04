import pathlib
import subprocess
import sys

import pytest


SSH_HOST = "localhost_guard"


def _ssh(*remote_command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", SSH_HOST, *remote_command],
        text=True,
        capture_output=True,
        timeout=10,
    )


def _require_guarded_ssh() -> None:
    result = _ssh("rhl-ps")
    if result.returncode != 0:
        pytest.skip(
            f"{SSH_HOST} is not configured for guarded remote-http-launcher tests: "
            f"{result.stderr.strip()}"
        )


def test_local_profile_uses_guarded_local2_cluster() -> None:
    profile_path = pathlib.Path(__file__).with_name("seamless.profile.yaml")
    profile_text = profile_path.read_text(encoding="utf-8")

    assert "- cluster: local2" in profile_text.splitlines()


@pytest.mark.parametrize(
    "command",
    [
        "kill -1 999999",
        "pkill -f definitely-not-a-real-rhl-test-process",
        "rm -f /tmp/definitely-not-a-real-rhl-test-file",
        "python3 -c 'print(1)'",
        "bash -lc 'kill -1 999999'",
    ],
)
def test_guard_rejects_raw_commands(command: str) -> None:
    _require_guarded_ssh()

    result = _ssh(command)

    assert result.returncode != 0
    assert "rhl-guard: rejected:" in result.stderr


def test_guard_rejects_interactive_sessions() -> None:
    _require_guarded_ssh()

    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", SSH_HOST],
        input="",
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "this program is an SSH guard" in result.stderr


def _run_helper(tmp_path: pathlib.Path, module: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = {"REMOTE_HTTP_LAUNCHER_DIR": tmp_path.as_posix()}
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        text=True,
        capture_output=True,
        timeout=10,
        env=env,
    )


def test_rhl_ps_json_and_key_output(tmp_path: pathlib.Path) -> None:
    server = tmp_path / "server"
    server.mkdir()
    (server / "demo.json").write_text(
        '{"pid": 999999999, "status": "running", "port": 1234, "meta": {"cluster": "C"}}',
        encoding="utf-8",
    )

    result = _run_helper(tmp_path, "ssh_guard.helpers.ps", "--json")
    assert result.returncode == 0
    assert '"key": "demo"' in result.stdout
    assert '"status": "stale"' in result.stdout

    result = _run_helper(tmp_path, "ssh_guard.helpers.ps", "--key")
    assert result.returncode == 0
    assert result.stdout.strip() == "demo"


def test_rhl_rm_leaves_logs(tmp_path: pathlib.Path) -> None:
    server = tmp_path / "server"
    server.mkdir()
    (server / "demo.json").write_text("{}", encoding="utf-8")
    (server / "demo.log").write_text("log", encoding="utf-8")

    result = _run_helper(tmp_path, "ssh_guard.helpers.rm", "--server", "demo")

    assert result.returncode == 0
    assert not (server / "demo.json").exists()
    assert (server / "demo.log").exists()


def test_rhl_clear_removes_direct_children(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "clearable"
    root.mkdir()
    (root / "file").write_text("x", encoding="utf-8")
    nested = root / "dir"
    nested.mkdir()
    (nested / "nested").write_text("x", encoding="utf-8")

    result = _run_helper(tmp_path, "ssh_guard.helpers.clear", root.as_posix())

    assert result.returncode == 0
    assert list(root.iterdir()) == []


def test_rhl_ps_persistent_file_modes(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "persistent"
    root.mkdir()

    result = _run_helper(tmp_path, "ssh_guard.helpers.ps_persistent", root.as_posix(), "--level", "0", "--json")
    assert result.returncode == 0
    assert '"state": "empty"' in result.stdout

    (root / "seamless.db").write_text("db", encoding="utf-8")
    result = _run_helper(
        tmp_path,
        "ssh_guard.helpers.ps_persistent",
        root.as_posix(),
        "--file",
        "seamless.db",
        "--json",
    )
    assert result.returncode == 0
    assert '"state": "populated"' in result.stdout
    assert '"size": 2' in result.stdout


@pytest.mark.parametrize(
    "module",
    [
        "ssh_guard.helpers.cache_conda",
        "ssh_guard.helpers.ps",
        "ssh_guard.helpers.ps_persistent",
        "ssh_guard.helpers.stop",
        "ssh_guard.helpers.rm",
        "ssh_guard.helpers.logs",
        "ssh_guard.helpers.inspect",
        "ssh_guard.helpers.clear",
    ],
)
def test_new_helpers_help(module: str, tmp_path: pathlib.Path) -> None:
    result = _run_helper(tmp_path, module, "--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout
