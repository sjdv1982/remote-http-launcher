import pathlib
import subprocess

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
    result = _ssh("rhl-ls-services")
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
    assert "interactive session not allowed" in result.stderr
