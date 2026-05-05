import pathlib
import subprocess
import sys
import json
import http.server
import os
import signal
import socket
import socketserver
import importlib
import threading

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


def _run_helper(
    tmp_path: pathlib.Path,
    module: str,
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["REMOTE_HTTP_LAUNCHER_DIR"] = tmp_path.as_posix()
    if extra_env:
        env.update(extra_env)
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


def test_rhl_stop_marks_server_state_stale(tmp_path: pathlib.Path) -> None:
    server = tmp_path / "server"
    server.mkdir()
    (server / "demo.json").write_text(
        '{"pid": 999999999, "status": "running", "port": 1234}',
        encoding="utf-8",
    )

    result = _run_helper(tmp_path, "ssh_guard.helpers.stop", "demo")

    assert result.returncode == 0
    data = json.loads((server / "demo.json").read_text(encoding="utf-8"))
    assert data["status"] == "stale"


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


def test_rhl_clear_expands_home(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    root = home / "clearable"
    root.mkdir(parents=True)
    (root / "file").write_text("x", encoding="utf-8")
    monkeypatch.setenv("HOME", home.as_posix())

    result = _run_helper(tmp_path, "ssh_guard.helpers.clear", "~/clearable")

    assert result.returncode == 0
    assert list(root.iterdir()) == []


def test_rhl_clear_skips_dot_directories_but_not_dot_files(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "clearable"
    root.mkdir()
    dot_dir = root / ".metadata"
    dot_dir.mkdir()
    (dot_dir / "kept").write_text("x", encoding="utf-8")
    (root / ".file").write_text("x", encoding="utf-8")
    (root / "regular").write_text("x", encoding="utf-8")
    normal_dir = root / "normal"
    nested_dot_dir = normal_dir / ".nested"
    nested_dot_dir.mkdir(parents=True)
    (nested_dot_dir / "kept").write_text("x", encoding="utf-8")
    (normal_dir / ".dotfile").write_text("x", encoding="utf-8")
    (normal_dir / "file").write_text("x", encoding="utf-8")

    result = _run_helper(tmp_path, "ssh_guard.helpers.clear", root.as_posix())

    assert result.returncode == 0
    assert sorted(child.name for child in root.iterdir()) == [".metadata", "normal"]
    assert (dot_dir / "kept").exists()
    assert sorted(child.name for child in normal_dir.iterdir()) == [".nested"]
    assert (nested_dot_dir / "kept").exists()


def test_service_binary_loader_default_and_override(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_module = sys.modules.get("yaml")
    if yaml_module is not None and not hasattr(yaml_module, "__file__"):
        sys.modules.pop("yaml")
    real_yaml = importlib.import_module("yaml")
    from ssh_guard import _tools

    _tools.yaml = real_yaml
    load_service_binaries = _tools.load_service_binaries

    monkeypatch.delenv("RHL_TOOLS_YAML", raising=False)
    default_binaries = load_service_binaries()
    assert {"hashserver", "seamless-jobserver", "seamless-dask-wrapper"} <= default_binaries

    tools = tmp_path / "tools.yaml"
    tools.write_text(
        "demo:\n  command_template: /tmp/demo --status-file {status_file}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RHL_TOOLS_YAML", tools.as_posix())
    assert load_service_binaries() == frozenset({"/tmp/demo"})

    tools.write_text("[]\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_service_binaries()

    tools.write_text("demo:\n  command_template: ''\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_service_binaries()


def test_rhl_pid_alive(tmp_path: pathlib.Path) -> None:
    result = _run_helper(tmp_path, "ssh_guard.helpers.pid_alive", str(os.getpid()))
    assert result.returncode == 0

    result = _run_helper(tmp_path, "ssh_guard.helpers.pid_alive", "999999999")
    assert result.returncode == 1

    for raw in ["nope", "0", "-1"]:
        result = _run_helper(tmp_path, "ssh_guard.helpers.pid_alive", raw)
        assert result.returncode != 0


def test_rhl_conda_info(tmp_path: pathlib.Path) -> None:
    (tmp_path / "conda-setup.json").write_text('{"conda_source": null}', encoding="utf-8")
    result = _run_helper(tmp_path, "ssh_guard.helpers.conda_info")
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"conda_source": None}

    (tmp_path / "conda-setup.json").unlink()
    result = _run_helper(tmp_path, "ssh_guard.helpers.conda_info")
    assert result.returncode == 1

    (tmp_path / "conda-setup.json").write_text("{", encoding="utf-8")
    result = _run_helper(tmp_path, "ssh_guard.helpers.conda_info")
    assert result.returncode == 1

    (tmp_path / "conda-setup.json").write_text("[1]", encoding="utf-8")
    result = _run_helper(tmp_path, "ssh_guard.helpers.conda_info")
    assert result.returncode == 1

    (tmp_path / "conda-setup.json").write_text('"x"', encoding="utf-8")
    result = _run_helper(tmp_path, "ssh_guard.helpers.conda_info")
    assert result.returncode == 1


def test_rhl_inspect_with_mtime(tmp_path: pathlib.Path) -> None:
    server = tmp_path / "server"
    server.mkdir()
    (server / "demo.json").write_text('{"status": "running"}', encoding="utf-8")

    result = _run_helper(tmp_path, "ssh_guard.helpers.inspect", "demo", "--with-mtime")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload["mtime"], float)
    assert payload["data"] == {"status": "running"}


class _StatusHandler(http.server.BaseHTTPRequestHandler):
    status = 200

    def do_GET(self) -> None:
        self.send_response(self.status)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def _serve_status(status: int):
    handler = type("Handler", (_StatusHandler,), {"status": status})
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_rhl_handshake_status_codes(tmp_path: pathlib.Path) -> None:
    server = _serve_status(200)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/health"
        result = _run_helper(tmp_path, "ssh_guard.helpers.handshake", url)
        assert result.returncode == 0
    finally:
        server.shutdown()
        server.server_close()

    server = _serve_status(500)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/health"
        result = _run_helper(tmp_path, "ssh_guard.helpers.handshake", url)
        assert result.returncode == 2
    finally:
        server.shutdown()
        server.server_close()

    result = _run_helper(tmp_path, "ssh_guard.helpers.handshake", "file:///tmp/x")
    assert result.returncode != 0

    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    closed_port = closed.getsockname()[1]
    closed.close()
    result = _run_helper(
        tmp_path,
        "ssh_guard.helpers.handshake",
        f"http://127.0.0.1:{closed_port}/health",
    )
    assert result.returncode == 1


def test_rhl_verify_port(tmp_path: pathlib.Path) -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    try:
        port = listener.getsockname()[1]
        result = _run_helper(tmp_path, "ssh_guard.helpers.verify_port", "127.0.0.1", str(port))
        assert result.returncode == 0
    finally:
        listener.close()

    result = _run_helper(tmp_path, "ssh_guard.helpers.verify_port", "bad host", "80")
    assert result.returncode != 0

    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    closed_port = closed.getsockname()[1]
    closed.close()
    result = _run_helper(tmp_path, "ssh_guard.helpers.verify_port", "127.0.0.1", str(closed_port))
    assert result.returncode == 1

    for port in ["0", "65536", "nope"]:
        result = _run_helper(tmp_path, "ssh_guard.helpers.verify_port", "127.0.0.1", port)
        assert result.returncode != 0


def _write_fake_service(path: pathlib.Path, content: str = "#!/usr/bin/env sh\nsleep 30\n") -> pathlib.Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o700)
    return path


def _write_tools(path: pathlib.Path, binary: pathlib.Path) -> pathlib.Path:
    path.write_text(
        f"fake:\n  command_template: {binary.as_posix()} --status-file {{status_file}}\n",
        encoding="utf-8",
    )
    return path


def test_rhl_launch_service_happy_path(tmp_path: pathlib.Path) -> None:
    server = tmp_path / "server"
    server.mkdir()
    workdir = tmp_path / "work"
    workdir.mkdir()
    script = _write_fake_service(tmp_path / "fake-service")
    tools = _write_tools(tmp_path / "tools.yaml", script)

    result = _run_helper(
        tmp_path,
        "ssh_guard.helpers.launch_service",
        "--key",
        "demo",
        "--workdir",
        workdir.as_posix(),
        "--parameters",
        '{"a": 1}',
        "--meta",
        '{"b": 2}',
        "--",
        script.as_posix(),
        "--status-file",
        (server / "demo.json").as_posix(),
        extra_env={"RHL_TOOLS_YAML": tools.as_posix()},
    )

    assert result.returncode == 0, result.stderr
    data = json.loads((server / "demo.json").read_text(encoding="utf-8"))
    assert data["status"] == "starting"
    assert data["parameters"] == {"a": 1}
    assert data["meta"] == {"b": 2}
    assert (server / "demo.log").exists()
    os.kill(data["pid"], signal.SIGTERM)


@pytest.mark.parametrize(
    "args",
    [
        ["--key", "bad/key"],
        ["--key", "demo", "--workdir", "/"],
        ["--key", "demo", "--workdir", "/tmp/../tmp"],
        ["--key", "demo", "--parameters", "{"],
        ["--key", "demo", "--parameters", "[]"],
        ["--key", "demo", "--meta", "{"],
        ["--key", "demo", "--meta", "[]"],
    ],
)
def test_rhl_launch_service_rejects_bad_inputs(
    tmp_path: pathlib.Path, args: list[str]
) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    script = _write_fake_service(tmp_path / "fake-service")
    tools = _write_tools(tmp_path / "tools.yaml", script)
    normalized_args = [
        value if value != "/tmp/../tmp" else (tmp_path / ".." / tmp_path.name).as_posix()
        for value in args
    ]
    if "--workdir" not in normalized_args:
        normalized_args += ["--workdir", workdir.as_posix()]

    result = _run_helper(
        tmp_path,
        "ssh_guard.helpers.launch_service",
        *normalized_args,
        "--",
        script.as_posix(),
        "--status-file",
        (tmp_path / "server" / "demo.json").as_posix(),
        extra_env={"RHL_TOOLS_YAML": tools.as_posix()},
    )

    assert result.returncode != 0


def test_rhl_launch_service_rejects_service_argv_and_status_file_escape(
    tmp_path: pathlib.Path,
) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    script = _write_fake_service(tmp_path / "fake-service")
    tools = _write_tools(tmp_path / "tools.yaml", script)

    result = _run_helper(
        tmp_path,
        "ssh_guard.helpers.launch_service",
        "--key",
        "demo",
        "--workdir",
        workdir.as_posix(),
        extra_env={"RHL_TOOLS_YAML": tools.as_posix()},
    )
    assert result.returncode != 0

    result = _run_helper(
        tmp_path,
        "ssh_guard.helpers.launch_service",
        "--key",
        "demo",
        "--workdir",
        workdir.as_posix(),
        "--",
        "/bin/sh",
        "-c",
        "sleep 30",
        extra_env={"RHL_TOOLS_YAML": tools.as_posix()},
    )
    assert result.returncode != 0

    result = _run_helper(
        tmp_path,
        "ssh_guard.helpers.launch_service",
        "--key",
        "demo",
        "--workdir",
        workdir.as_posix(),
        "--",
        script.as_posix(),
        "--status-file",
        (tmp_path / "escape.json").as_posix(),
        extra_env={"RHL_TOOLS_YAML": tools.as_posix()},
    )
    assert result.returncode != 0


def test_rhl_launch_service_conda_cache_path(tmp_path: pathlib.Path) -> None:
    server = tmp_path / "server"
    server.mkdir()
    workdir = tmp_path / "work"
    workdir.mkdir()
    script = _write_fake_service(tmp_path / "fake-service")
    tools = _write_tools(tmp_path / "tools.yaml", script)
    conda_source = tmp_path / "conda.sh"
    conda_source.write_text("conda() { return 0; }\n", encoding="utf-8")
    (tmp_path / "conda-setup.json").write_text(
        json.dumps({"conda_source": conda_source.as_posix()}),
        encoding="utf-8",
    )

    result = _run_helper(
        tmp_path,
        "ssh_guard.helpers.launch_service",
        "--key",
        "demo",
        "--workdir",
        workdir.as_posix(),
        "--conda-env",
        "demo-env",
        "--",
        script.as_posix(),
        "--status-file",
        (server / "demo.json").as_posix(),
        extra_env={"RHL_TOOLS_YAML": tools.as_posix()},
    )

    assert result.returncode == 0, result.stderr
    data = json.loads((server / "demo.json").read_text(encoding="utf-8"))
    os.kill(data["pid"], signal.SIGTERM)


def test_guard_only_accepts_top_level_helpers() -> None:
    from ssh_guard.guard import _parse_and_check

    assert _parse_and_check("rhl-ps")[0] is not None
    for command in [
        "kill -1 999999",
        "python3 -c 'print(1)'",
        "bash -lc 'ps -p 1 -o pid='",
        "bash -lc 'cat ~/.bashrc'",
        "bash -lc \"python3 - <<'__RHL_REMOTE_SCRIPT__'\\nprint(1)\\n__RHL_REMOTE_SCRIPT__\"",
        "'rhl-ps\n'",
    ]:
        args, _reason = _parse_and_check(command)
        assert args is None


def test_guard_requires_policy_for_path_helpers() -> None:
    from ssh_guard.guard import _parse_and_check

    for command in [
        "rhl-clear /tmp/rhl-data",
        "rhl-ps-persistent /tmp/rhl-data --json",
        "rhl-launch-service --key demo --workdir /tmp/rhl-data -- /bin/true",
    ]:
        args, reason = _parse_and_check(command)
        assert args is None
        assert "path-policy" in reason


def test_guard_permissive_policy_keeps_always_on_path_heuristics(
    tmp_path: pathlib.Path,
) -> None:
    from ssh_guard.guard import _parse_and_check, _parse_policy_args

    policy, error = _parse_policy_args(["--permissive-paths"])
    assert error is None
    assert policy is not None

    args, _reason = _parse_and_check(f"rhl-clear {tmp_path.as_posix()}/data", policy)
    assert args is not None

    args, reason = _parse_and_check("rhl-clear /tmp/.cache/data", policy)
    assert args is None
    assert "dot-prefixed" in reason

    args, reason = _parse_and_check("rhl-clear /", policy)
    assert args is None
    assert "system-root" in reason or "$HOME" in reason

    for path in ["/dev/shm/rhl-data", "/ramdisk/rhl-data"]:
        args, reason = _parse_and_check(f"rhl-clear {path}", policy)
        assert args is not None, reason


def test_guard_data_roots_policy_bounds_all_path_helpers(
    tmp_path: pathlib.Path,
) -> None:
    from ssh_guard.guard import _parse_and_check, _parse_policy_args

    root = tmp_path / "allowed"
    root.mkdir()
    roots_file = tmp_path / "roots.txt"
    roots_file.write_text(f"# comment\n{root.as_posix()}\n", encoding="utf-8")
    policy, error = _parse_policy_args(["--data-roots", roots_file.as_posix()])
    assert error is None
    assert policy is not None

    inside = root / "project"
    outside = tmp_path / "outside"
    for command in [
        f"rhl-clear {inside.as_posix()}",
        f"rhl-ps-persistent {inside.as_posix()} --level 1 --json",
        f"rhl-launch-service --key demo --workdir {inside.as_posix()} -- /bin/true",
    ]:
        args, reason = _parse_and_check(command, policy)
        assert args is not None, reason

    args, reason = _parse_and_check(f"rhl-clear {outside.as_posix()}", policy)
    assert args is None
    assert "outside configured --data-roots" in reason


def test_guard_data_roots_policy_allows_custom_top_level_roots(
    tmp_path: pathlib.Path,
) -> None:
    from ssh_guard.guard import _parse_and_check, _parse_policy_args

    roots_file = tmp_path / "roots.txt"
    roots_file.write_text("/ramdisk\n/dev/shm\n", encoding="utf-8")
    policy, error = _parse_policy_args(["--data-roots", roots_file.as_posix()])
    assert error is None
    assert policy is not None

    for path in ["/ramdisk/project", "/dev/shm/project"]:
        args, reason = _parse_and_check(f"rhl-clear {path}", policy)
        assert args is not None, reason


def test_guard_marker_policy_applies_to_clear_paths_but_not_workdir(
    tmp_path: pathlib.Path,
) -> None:
    from ssh_guard.guard import _parse_and_check, _parse_policy_args

    policy, error = _parse_policy_args(["--clear-policy", "marker:.rhl-clearable"])
    assert error is None
    assert policy is not None
    clearable = tmp_path / "clearable"
    clearable.mkdir()
    (clearable / ".rhl-clearable").write_text("", encoding="utf-8")
    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()

    args, reason = _parse_and_check(f"rhl-clear {clearable.as_posix()}", policy)
    assert args is not None, reason

    args, reason = _parse_and_check(f"rhl-ps-persistent {unmarked.as_posix()} --json", policy)
    assert args is None
    assert "required marker" in reason

    args, reason = _parse_and_check(
        f"rhl-launch-service --key demo --workdir {unmarked.as_posix()} -- /bin/true",
        policy,
    )
    assert args is not None, reason


def test_guard_rejects_conflicting_path_policy_options(tmp_path: pathlib.Path) -> None:
    from ssh_guard.guard import _parse_policy_args

    roots_file = tmp_path / "roots.txt"
    roots_file.write_text(f"{tmp_path.as_posix()}\n", encoding="utf-8")
    policy, error = _parse_policy_args(
        ["--data-roots", roots_file.as_posix(), "--permissive-paths"]
    )
    assert policy is None
    assert error is not None
    assert "mutually exclusive" in error


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


def test_rhl_ps_persistent_expands_home(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    root = home / "persistent"
    root.mkdir(parents=True)
    monkeypatch.setenv("HOME", home.as_posix())

    result = _run_helper(tmp_path, "ssh_guard.helpers.ps_persistent", "~/persistent", "--json")

    assert result.returncode == 0
    row = json.loads(result.stdout)
    assert row["path"] == root.as_posix()
    assert row["state"] == "empty"


def test_rhl_ps_persistent_marker_filters_reported_directories(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "persistent"
    project = root / "project"
    bucket = project / "aa"
    bucket.mkdir(parents=True)
    (project / ".HASHSERVER_PREFIX").write_text("", encoding="utf-8")
    (bucket / "payload").write_text("data", encoding="utf-8")

    result = _run_helper(
        tmp_path,
        "ssh_guard.helpers.ps_persistent",
        root.as_posix(),
        "--level",
        "2",
        "--marker",
        ".HASHSERVER_PREFIX",
        "--json",
    )

    assert result.returncode == 0
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["path"] for row in rows] == [project.as_posix()]
    assert rows[0]["state"] == "populated"


@pytest.mark.parametrize(
    "module",
    [
        "ssh_guard.helpers.cache_conda",
        "ssh_guard.helpers.conda_info",
        "ssh_guard.helpers.pid_alive",
        "ssh_guard.helpers.handshake",
        "ssh_guard.helpers.verify_port",
        "ssh_guard.helpers.launch_service",
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
