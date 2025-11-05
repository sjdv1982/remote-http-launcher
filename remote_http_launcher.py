#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import ipaddress
import json
import os
import pathlib
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit("PyYAML is required to run remote_http_launcher.") from exc


CONFIG_DIR_ENV = "REMOTE_HTTP_LAUNCHER_DIR"
DEFAULT_DIRNAME = ".remote-http-launcher"
REMOTE_JSON_NAME = "{key}.json"
REMOTE_LOG_PREFIX = "log-{key}-"
REMOTE_LOG_SUFFIX = ".log"
LOCAL_JSON_PERMS = 0o600

FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
DIRNAME_RE = re.compile(r"^[A-Za-z0-9@_./~-]+$")


class LauncherError(RuntimeError):
    pass


@dataclasses.dataclass
class HandshakeConfig:
    path: str = ""
    parameters: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Configuration:
    workdir: str
    hostname: str
    ssh_hostname: str
    key: str
    command_template: str
    network_interface: str
    handshake: HandshakeConfig | None
    conda_env: Optional[str]
    namespace: Dict[str, Any]
    raw: Dict[str, Any]

    @staticmethod
    def from_yaml(path: pathlib.Path) -> "Configuration":
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise LauncherError("Configuration YAML must define a mapping.")
        cfg = Configuration._validate_and_build(data)
        return cfg

    @staticmethod
    def _validate_and_build(data: Dict[str, Any]) -> "Configuration":
        workdir = Configuration._require_string(data, "workdir")
        if not DIRNAME_RE.fullmatch(workdir):
            raise LauncherError("workdir must be a plausible UNIX directory name.")

        hostname = Configuration._require_string(data, "hostname")
        if not _is_valid_hostname_or_ip(hostname):
            raise LauncherError("hostname must be an HTTP hostname or IP address.")

        ssh_hostname = Configuration._get_string(data, "ssh-hostname", hostname)
        if not _is_valid_hostname_or_ip(ssh_hostname):
            raise LauncherError(
                "ssh-hostname must be a valid SSH hostname or IP address."
            )

        network_interface = Configuration._get_string(
            data, "network-interface", "localhost"
        )
        if not _is_valid_hostname_or_ip(network_interface):
            raise LauncherError(
                "network-interface must be a valid HTTP hostname or IP address."
            )

        handshake = Configuration._parse_handshake(data.get("handshake"))

        namespace: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(key, str) and key.isidentifier():
                namespace[key] = value

        namespace["config"] = data

        key_template = Configuration._require_string(data, "key")
        evaluated_key = _evaluate_template(key_template, namespace)
        if not isinstance(evaluated_key, str):
            raise LauncherError("The evaluated key must be a string.")
        if (
            not evaluated_key
            or "/" in evaluated_key
            or not FILENAME_RE.fullmatch(evaluated_key)
        ):
            raise LauncherError("The key must be a plausible UNIX file name.")

        namespace["key"] = evaluated_key

        command_template = Configuration._require_string(data, "command")

        conda_env = data.get("conda")
        if conda_env is not None and not isinstance(conda_env, str):
            raise LauncherError("The conda field, when present, must be a string.")

        return Configuration(
            workdir=workdir,
            hostname=hostname,
            ssh_hostname=ssh_hostname,
            key=evaluated_key,
            command_template=command_template,
            network_interface=network_interface,
            handshake=handshake,
            conda_env=conda_env,
            namespace=namespace,
            raw=data,
        )

    @staticmethod
    def _parse_handshake(value: Any) -> HandshakeConfig | None:
        if value is None:
            return None
        if isinstance(value, str):
            return HandshakeConfig(path=value)
        if isinstance(value, dict):
            path_value = value.get("path", "")
            if not isinstance(path_value, str):
                raise LauncherError("handshake.path must be a string.")
            parameters_value = value.get("parameters", {})
            if not isinstance(parameters_value, dict):
                raise LauncherError("handshake.parameters must be a mapping.")
            normalized: Dict[str, str] = {}
            for key, val in parameters_value.items():
                if not isinstance(key, str) or not isinstance(val, (str, int, float)):
                    raise LauncherError(
                        "handshake parameters must map strings to simple values."
                    )
                normalized[key] = str(val)
            return HandshakeConfig(path=path_value, parameters=normalized)
        raise LauncherError(
            "handshake must be a string path or a mapping with path and parameters."
        )

    @staticmethod
    def _require_string(data: Dict[str, Any], key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str):
            raise LauncherError(f"{key} must be provided and be a string.")
        return value

    @staticmethod
    def _get_string(data: Dict[str, Any], key: str, default: str) -> str:
        value = data.get(key, default)
        if not isinstance(value, str):
            raise LauncherError(f"{key} must be a string.")
        return value


@dataclasses.dataclass
class LocalState:
    connection_dir: pathlib.Path
    json_path: pathlib.Path

    @staticmethod
    def build(
        config: Configuration, override_dir: Optional[pathlib.Path]
    ) -> "LocalState":
        base_dir = override_dir or _default_directory()
        base_dir.mkdir(parents=True, exist_ok=True)
        json_path = base_dir / REMOTE_JSON_NAME.format(key=config.key)
        return LocalState(connection_dir=base_dir, json_path=json_path)

    def read(self) -> Optional[Dict[str, Any]]:
        if not self.json_path.exists():
            return None
        with self.json_path.open("r", encoding="utf-8") as handle:
            try:
                return json.load(handle)
            except json.JSONDecodeError as exc:
                raise LauncherError(
                    f"Malformed local connection file: {self.json_path}"
                ) from exc

    def write(self, payload: Dict[str, Any]) -> None:
        tmp_path = self.json_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.chmod(tmp_path, LOCAL_JSON_PERMS)
        tmp_path.replace(self.json_path)

    def delete(self) -> None:
        try:
            self.json_path.unlink()
        except FileNotFoundError:
            return


class SSHExecutor:
    def __init__(self, host: str):
        self.host = host

    def run_shell(
        self, command: str, *, check: bool = True
    ) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["ssh", self.host, "bash", "-lc", command],
            text=True,
            capture_output=True,
        )
        if check and result.returncode != 0:
            raise LauncherError(
                f"SSH command failed: {command}\n{result.stderr.strip()}"
            )
        return result

    def run_python(
        self, script: str, *, check: bool = True
    ) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["ssh", self.host, "python3", "-c", script],
            text=True,
            capture_output=True,
        )
        if check and result.returncode != 0:
            raise LauncherError(
                f"Remote Python command failed: {result.stderr.strip()}"
            )
        return result


@dataclasses.dataclass
class RemoteState:
    ssh: SSHExecutor
    cfg: Configuration
    remote_dir: str

    @property
    def json_path(self) -> str:
        return os.path.join(self.remote_dir, REMOTE_JSON_NAME.format(key=self.cfg.key))

    def exists(self) -> bool:
        script = (
            "import pathlib\n"
            f"path = pathlib.Path({self.json_path!r}).expanduser()\n"
            "import sys\n"
            "sys.exit(0 if path.exists() else 1)\n"
        )
        result = self.ssh.run_python(script, check=False)
        return result.returncode == 0

    def remove(self) -> None:
        script = (
            "import pathlib\n"
            f"path = pathlib.Path({self.json_path!r}).expanduser()\n"
            "import os\n"
            "import sys\n"
            "try:\n"
            "    os.remove(path)\n"
            "except FileNotFoundError:\n"
            "    pass\n"
        )
        self.ssh.run_python(script, check=True)

    def read(self) -> Dict[str, Any]:
        script = (
            "import json, pathlib, sys\n"
            f"path = pathlib.Path({self.json_path!r}).expanduser()\n"
            "with path.open('r', encoding='utf-8') as handle:\n"
            "    data = json.load(handle)\n"
            "import json\n"
            "json.dump(data, sys.stdout)\n"
        )
        result = self.ssh.run_python(script)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise LauncherError("Remote connection JSON is malformed.") from exc

    def stat_and_read(self) -> Dict[str, Any]:
        script = (
            "import json, pathlib, sys, time\n"
            f"path = pathlib.Path({self.json_path!r}).expanduser()\n"
            "stat = path.stat()\n"
            "with path.open('r', encoding='utf-8') as handle:\n"
            "    data = json.load(handle)\n"
            "json.dump({'mtime': stat.st_mtime, 'data': data}, sys.stdout)\n"
        )
        result = self.ssh.run_python(script)
        payload = json.loads(result.stdout)
        return payload

    def launch_process(self, conda_base: Optional[str]) -> Dict[str, Any]:
        command_template = self.cfg.command_template
        namespace = self.cfg.namespace.copy()
        namespace["status_file"] = self.json_path
        evaluated_command = _evaluate_template(command_template, self.cfg.namespace)
        if not isinstance(evaluated_command, str):
            raise LauncherError("The evaluated command must be a string.")
        if not evaluated_command.strip() or "\n" in evaluated_command:
            raise LauncherError("The command must be a plausible bash command.")

        if self.cfg.conda_env:
            if not conda_base:
                raise LauncherError(
                    "Conda base path is required to activate environment."
                )
            activation = (
                f"source {shlex.quote(os.path.join(conda_base, 'etc/profile.d/conda.sh'))} && "
                f"conda activate {shlex.quote(self.cfg.conda_env)} && {evaluated_command}"
            )
            command = activation

        script = (
            "import json, pathlib, tempfile, os, subprocess, sys\n"
            f"remote_dir = pathlib.Path({self.remote_dir!r}).expanduser()\n"
            "remote_dir.mkdir(parents=True, exist_ok=True)\n"
            f"json_path = pathlib.Path({self.json_path!r}).expanduser()\n"
            "log_file = tempfile.NamedTemporaryFile(\n"
            f"    dir=remote_dir,\n"
            f"    prefix={REMOTE_LOG_PREFIX.format(key=self.cfg.key)!r},\n"
            f"    suffix={REMOTE_LOG_SUFFIX!r},\n"
            "    delete=False,\n"
            ")\n"
            "log_file.close()\n"
            "stdout_handle = open(log_file.name, 'ab', buffering=0)\n"
            f"command = {command!r}\n"
            f"workdir = {self.cfg.workdir!r}\n"
            "proc = subprocess.Popen(\n"
            "    command,\n"
            "    shell=True,\n"
            "    cwd=workdir,\n"
            "    stdout=stdout_handle,\n"
            "    stderr=subprocess.STDOUT,\n"
            "    start_new_session=True,\n"
            "    executable='/bin/bash',\n"
            ")\n"
            "data = {\n"
            f"    'workdir': {self.cfg.workdir!r},\n"
            "    'log': log_file.name,\n"
            f"    'command': {evaluated_command!r},\n"
            "    'uid': os.getuid(),\n"
            "    'pid': proc.pid,\n"
            "    'status': 'starting',\n"
            "}\n"
            "network_interface = {network!r}\n"
            "if network_interface is not None:\n"
            "    data['network-interface'] = network_interface\n"
            "with json_path.open('w', encoding='utf-8') as handle:\n"
            "    json.dump(data, handle)\n"
            "stdout_handle.close()\n"
            "print(json.dumps(data))\n"
        ).replace("{network!r}", repr(self.cfg.network_interface))
        result = self.ssh.run_python(script)
        return json.loads(result.stdout.strip())

    def verify_port_in_use(self, host: str, port: int) -> None:
        script = (
            "import socket, sys\n"
            f"host = {host!r}\n"
            f"port = {port}\n"
            "sock = socket.socket()\n"
            "sock.settimeout(2.0)\n"
            "try:\n"
            "    sock.connect((host, port))\n"
            "except Exception as exc:\n"
            "    print(str(exc), file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "finally:\n"
            "    sock.close()\n"
        )
        self.ssh.run_python(script)

    def handshake(
        self, host: str, port: int, handshake: HandshakeConfig | None
    ) -> None:
        url = build_handshake_url(host, port, handshake)
        script = (
            "import sys, urllib.error, urllib.request\n"
            f"url = {url!r}\n"
            "try:\n"
            "    with urllib.request.urlopen(url, timeout=10) as response:\n"
            "        status = getattr(response, 'status', 200)\n"
            "        if not (200 <= status < 300):\n"
            "            sys.exit(2)\n"
            "except Exception as exc:\n"
            "    print(str(exc), file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )
        self.ssh.run_python(script)

    def process_exists(self, pid: int) -> bool:
        result = self.ssh.run_shell(f"ps -p {pid} -o pid=", check=False)
        return result.returncode == 0 and result.stdout.strip() != ""


def _default_directory() -> pathlib.Path:
    base = os.environ.get(CONFIG_DIR_ENV)
    if base:
        return pathlib.Path(base).expanduser()
    return pathlib.Path.home() / DEFAULT_DIRNAME


def _evaluate_template(template: str, namespace: Dict[str, Any]) -> Any:
    expression = f"f{template!r}"
    code = compile(expression, "<config>", "eval")
    return eval(code, {"__builtins__": {}}, namespace)


def _is_valid_hostname_or_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if len(host) > 253:
        return False
    if host.endswith("."):
        host = host[:-1]
    labels = host.split(".")
    for label in labels:
        if not label or len(label) > 63:
            return False
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label):
            return False
    return True


def build_handshake_url(host: str, port: int, handshake: HandshakeConfig | None) -> str:
    path = ""
    params: Dict[str, str] = {}
    if handshake:
        path = handshake.path or ""
        if path:
            parsed = urllib.parse.urlsplit(path)
            path = parsed.path or ""
            if parsed.query:
                params.update(
                    {
                        k: v
                        for k, v in urllib.parse.parse_qsl(
                            parsed.query, keep_blank_values=True
                        )
                    }
                )
        params.update(handshake.parameters)
    norm_path = path if path.startswith("/") or not path else f"/{path}"
    query = urllib.parse.urlencode(params)
    return urllib.parse.urlunparse(
        ("http", f"{host}:{port}", norm_path or "/", "", query, "")
    )


def perform_local_handshake(
    host: str, port: int, handshake: HandshakeConfig | None
) -> None:
    url = build_handshake_url(host, port, handshake)
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            status = getattr(response, "status", 200)
            if not (200 <= status < 300):
                raise LauncherError(f"Handshake failed with HTTP status {status}.")
    except urllib.error.URLError as exc:
        raise LauncherError(f"Handshake failed: {exc}") from exc


def validate_remote_running_state(data: Dict[str, Any]) -> None:
    port = data.get("port")
    if not isinstance(port, int):
        raise LauncherError("Remote JSON must provide integer port.")
    if port < 1 or port > 65535:
        raise LauncherError("Remote port is out of range.")
    if not isinstance(data.get("command"), str) or not data["command"]:
        raise LauncherError("Remote JSON must include a plausible command string.")
    if "network-interface" in data and not isinstance(data["network-interface"], str):
        raise LauncherError(
            "Remote JSON network-interface must be a string when present."
        )
    if not isinstance(data.get("uid"), int):
        raise LauncherError("Remote JSON must include uid.")
    if not isinstance(data.get("pid"), int):
        raise LauncherError("Remote JSON must include pid.")


def allocate_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        _, port = sock.getsockname()
        return int(port)


def establish_tunnel(
    ssh_host: str, remote_host: str, remote_port: int, local_port: int
) -> None:
    cmd = [
        "ssh",
        "-f",
        "-N",
        "-L",
        f"{local_port}:{remote_host}:{remote_port}",
        ssh_host,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise LauncherError(f"Failed to establish SSH tunnel: {result.stderr.strip()}")


def _fetch_conda_base(ssh: SSHExecutor) -> Optional[str]:
    try:
        result = ssh.run_shell("command -v conda >/dev/null 2>&1")
    except LauncherError:
        return None
    if result.returncode != 0:
        return None
    base_result = ssh.run_shell("conda info --base", check=True)
    base_path = base_result.stdout.strip()
    if not base_path:
        raise LauncherError("Unable to discover conda base path.")
    return base_path


def _conda_env_exists(ssh: SSHExecutor, env: str) -> bool:
    result = ssh.run_shell("conda env list --json", check=True)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LauncherError("Failed to parse output from conda env list.") from exc
    envs = payload.get("envs", [])
    return any(path.endswith(f"{os.sep}{env}") for path in envs)


def handle_remote(cfg: Configuration, remote: RemoteState) -> Dict[str, Any]:
    attempted_launch = False
    while True:
        if remote.exists():
            data = remote.read()
            status = data.get("status")
            if status == "running":
                validate_remote_running_state(data)
                target_host = data.get("network-interface", cfg.network_interface)
                remote.verify_port_in_use(target_host, data["port"])
                remote.handshake(target_host, data["port"], cfg.handshake)
                return data
            if status == "starting" and isinstance(data.get("pid"), int):
                monitored = monitor_remote_start(cfg, remote, data)
                if monitored is not None:
                    return monitored
                if attempted_launch:
                    raise LauncherError(
                        "Remote process did not finish starting after relaunch."
                    )
                remote.remove()
                attempted_launch = True
                continue
            if attempted_launch:
                raise LauncherError("Remote connection file verification failed twice.")
            remote.remove()
            attempted_launch = True
            continue
        if attempted_launch:
            raise LauncherError("Remote launch already attempted and failed.")
        conda_base = None
        if cfg.conda_env:
            conda_base = _fetch_conda_base(remote.ssh)
            if not conda_base:
                raise LauncherError(
                    "Conda environment requested but conda is not installed remotely."
                )
            if not _conda_env_exists(remote.ssh, cfg.conda_env):
                raise LauncherError(
                    f"Remote conda environment '{cfg.conda_env}' does not exist."
                )
        remote.launch_process(conda_base)
        attempted_launch = True
        time.sleep(2.0)


def monitor_remote_start(
    cfg: Configuration, remote: RemoteState, initial: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    payload = remote.stat_and_read()
    last_mtime = float(payload["mtime"])
    last_change = time.time()
    pid = int(initial["pid"])
    while True:
        time.sleep(5.0)
        payload = remote.stat_and_read()
        data = payload["data"]
        status = data.get("status")
        if status == "running":
            validate_remote_running_state(data)
            target_host = data.get("network-interface", cfg.network_interface)
            remote.verify_port_in_use(target_host, data["port"])
            remote.handshake(target_host, data["port"], cfg.handshake)
            return data
        mtime = float(payload["mtime"])
        if mtime != last_mtime:
            last_mtime = mtime
            last_change = time.time()
            continue
        if time.time() - last_change > 60:
            if not remote.process_exists(pid):
                return None


def handle_local(
    cfg: Configuration, local_state: LocalState
) -> Optional[Dict[str, Any]]:
    existing = local_state.read()
    if existing is None:
        return None
    port = existing.get("port")
    if not isinstance(port, int):
        raise LauncherError("Local JSON must contain integer port.")
    hostname = existing.get("hostname")
    if not isinstance(hostname, str):
        raise LauncherError("Local JSON must contain hostname.")
    try:
        perform_local_handshake(hostname, port, cfg.handshake)
        return existing
    except LauncherError:
        local_state.delete()
        return None


def create_local_file(
    cfg: Configuration,
    local_state: LocalState,
    remote_data: Dict[str, Any],
    tunnel: bool,
) -> Dict[str, Any]:
    port = int(remote_data["port"])
    if tunnel:
        local_port = allocate_local_port()
        establish_tunnel(cfg.ssh_hostname, cfg.hostname, port, local_port)
        payload: Dict[str, Any] = {"hostname": "localhost", "port": local_port}
    else:
        payload = {"hostname": cfg.hostname, "port": port}
        if cfg.ssh_hostname != cfg.hostname:
            payload["ssh-hostname"] = cfg.ssh_hostname
    local_state.write(payload)
    perform_local_handshake(payload["hostname"], payload["port"], cfg.handshake)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remote HTTP launcher.")
    parser.add_argument(
        "config", type=pathlib.Path, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--connection-dir",
        type=pathlib.Path,
        help="Override the connection directory. Defaults to ~/.remote-http-launcher/",
    )
    parser.add_argument(
        "--tunnel",
        action="store_true",
        help="Create an SSH tunnel instead of a direct connection.",
    )
    args = parser.parse_args(argv)

    cfg = Configuration.from_yaml(args.config)
    local_state = LocalState.build(cfg, args.connection_dir)

    local_result = handle_local(cfg, local_state)
    if local_result:
        return 0

    remote_dir = os.path.join("~", DEFAULT_DIRNAME)
    remote = RemoteState(SSHExecutor(cfg.ssh_hostname), cfg, remote_dir)
    remote_result = handle_remote(cfg, remote)

    create_local_file(cfg, local_state, remote_result, args.tunnel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
