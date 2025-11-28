#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import ipaddress
import json
import logging
import os
import pathlib
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, Optional, Tuple

try:
    import yaml  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit("PyYAML is required to run remote_http_launcher.") from exc


CONFIG_DIR_ENV = "REMOTE_HTTP_LAUNCHER_DIR"
DEFAULT_DIRNAME = ".remote-http-launcher"
JSON_NAME = "{key}.json"
REMOTE_LOG_NAME = "{key}.log"
LOCAL_JSON_PERMS = 0o600
CONDA_PATH_RE = re.compile(r'["\']([^"\']+/etc/profile\.d/conda\.sh)["\']')
CONDA_INIT_BLOCK_RE = re.compile(
    r"# >>> conda initialize >>>(.*?)# <<< conda initialize <<<",
    re.DOTALL,
)

FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
DIRNAME_RE = re.compile(r"^[A-Za-z0-9@_./~-]+$")


class LauncherError(RuntimeError):
    pass


LOGGER = logging.getLogger(__name__)
_LOGGER_CONFIGURED = False


def _configure_module_logger() -> None:
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return
    if LOGGER.handlers:
        _LOGGER_CONFIGURED = True
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    _LOGGER_CONFIGURED = True


@dataclasses.dataclass
class HandshakeConfig:
    path: str = ""
    parameters: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Configuration:
    workdir: str
    hostname: Optional[str]
    ssh_hostname: Optional[str]
    key: str
    command_template: str
    network_interface: str
    handshake: HandshakeConfig | None
    file_parameters: Dict[str, Any] | None
    tunnel: bool
    conda_env: Optional[str]
    namespace: Dict[str, Any]
    raw: Dict[str, Any]

    @staticmethod
    def from_yaml(path: pathlib.Path) -> "Configuration":
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return Configuration.from_mapping(data)

    @staticmethod
    def from_mapping(data: Dict[str, Any]) -> "Configuration":
        if not isinstance(data, dict):
            raise LauncherError("Configuration mapping must be a dictionary.")
        return Configuration._validate_and_build(data)

    @staticmethod
    def _validate_and_build(data: Dict[str, Any]) -> "Configuration":
        workdir = Configuration._require_string(data, "workdir")
        if not DIRNAME_RE.fullmatch(workdir):
            raise LauncherError("workdir must be a plausible UNIX directory name.")

        hostname: Optional[str]
        raw_hostname = data.get("hostname")
        if raw_hostname is None:
            hostname = None
        else:
            if not isinstance(raw_hostname, str):
                raise LauncherError("hostname must be a string when provided.")
            if not _is_valid_hostname_or_ip(raw_hostname):
                raise LauncherError("hostname must be an HTTP hostname or IP address.")
            hostname = raw_hostname

        ssh_hostname: Optional[str]
        raw_ssh = data.get("ssh_hostname", hostname)
        if raw_ssh is None:
            ssh_hostname = None
        else:
            if not isinstance(raw_ssh, str):
                raise LauncherError("ssh_hostname must be a string when provided.")
            if not _is_valid_hostname_or_ip(raw_ssh):
                raise LauncherError(
                    "ssh_hostname must be a valid SSH hostname or IP address."
                )
            ssh_hostname = raw_ssh

        network_interface = Configuration._get_string(
            data, "network_interface", "localhost"
        )
        if not _is_valid_hostname_or_ip(network_interface):
            raise LauncherError(
                "network_interface must be a valid HTTP hostname or IP address."
            )

        handshake = Configuration._parse_handshake(data.get("handshake"))
        file_parameters = Configuration._parse_file_parameters(
            data.get("file_parameters")
        )

        raw_tunnel = data.get("tunnel", False)
        if not isinstance(raw_tunnel, bool):
            raise LauncherError("tunnel must be a boolean when provided.")
        tunnel = raw_tunnel

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
            file_parameters=file_parameters,
            tunnel=tunnel,
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
    def _parse_file_parameters(value: Any) -> Dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise LauncherError("file_parameters must be a mapping when provided.")
        try:
            json.dumps(value)
        except TypeError as exc:
            raise LauncherError(
                "file_parameters must contain JSON-serializable values."
            ) from exc
        return value

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
        base_dir = override_dir or _default_client_directory()
        base_dir.mkdir(parents=True, exist_ok=True)
        json_path = base_dir / JSON_NAME.format(key=config.key)
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
        if sys.is_finalizing():
            # Precarious circumstances. Don't use "with
            open(tmp_path, "w", buffering=False).write(json.dumps(payload))
        else:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        os.chmod(tmp_path, LOCAL_JSON_PERMS)
        tmp_path.replace(self.json_path)

    def delete(self) -> None:
        try:
            self.json_path.unlink()
        except FileNotFoundError:
            return


class Executor:
    def __init__(self, host: str):
        self.host = host
        self._conda_checked = False
        self._conda_source: Optional[str] = None
        self._conda_unavailable = False

    def run_shell(
        self, command: str, *, check: bool = True, conda: bool = False, silent=False
    ) -> subprocess.CompletedProcess:
        snippet = command[:30]
        if len(command) > 30:
            snippet += " [...]"
            if len(command) > 65:
                snippet += command[-30:]

        if not silent:
            LOGGER.info(
                "%s run_shell requested command (conda=%s): %s",
                self.host,
                conda,
                snippet,
            )
        final_command = command
        if conda:
            conda_source = self._ensure_conda_setup()
            if conda_source:
                final_command = f"source {shlex.quote(conda_source)} && {final_command}"
        if not silent:
            LOGGER.debug("%s run_shell final command: %s", self.host, final_command)
        result = self._run_bash(final_command, snippet=snippet, silent=silent)
        if check and result.returncode != 0:
            raise LauncherError(
                f"{self.host} command failed: {final_command}\n{result.stderr.strip()}"
            )
        return result

    def _run_bash(
        self, command: str, *, snippet=None, silent=False
    ) -> subprocess.CompletedProcess:
        raise NotImplementedError

    def _ensure_conda_setup(self) -> Optional[str]:
        if self._conda_unavailable:
            raise LauncherError(
                "Conda is not installed on the target host or could not be initialized."
            )
        if self._conda_checked:
            return self._conda_source
        self._conda_checked = True
        LOGGER.info("%s probing for conda on PATH", self.host)
        probe = self._run_bash("command -v conda >/dev/null 2>&1")
        if probe.returncode == 0:
            LOGGER.info("%s found conda on PATH", self.host)
            self._conda_source = None
            return None
        conda_source = self._discover_conda_from_bashrc()
        if not conda_source:
            self._conda_unavailable = True
            raise LauncherError(
                "Conda is not available on the target host and ~/.bashrc does not "
                "contain a conda initialize block."
            )
        LOGGER.info(
            "%s discovered conda.sh via ~/.bashrc at %s", self.host, conda_source
        )
        self._conda_source = conda_source
        return self._conda_source

    def _discover_conda_from_bashrc(self) -> Optional[str]:
        LOGGER.info("%s scraping ~/.bashrc for conda initialize block", self.host)
        bashrc = self._run_bash("cat ~/.bashrc")
        if bashrc.returncode != 0:
            LOGGER.info(
                "%s unable to read ~/.bashrc (exit %s)", self.host, bashrc.returncode
            )
            return None
        text = bashrc.stdout
        block_match = CONDA_INIT_BLOCK_RE.search(text)
        if not block_match:
            LOGGER.info("%s no conda initialize block found in ~/.bashrc", self.host)
            return None
        block = block_match.group(1)
        path_match = CONDA_PATH_RE.search(block)
        if not path_match:
            LOGGER.info("%s conda initialize block missing conda.sh path", self.host)
            return None
        return path_match.group(1)

    def run_python(
        self, script: str, *, check: bool = True, conda: bool = False, silent=False
    ) -> subprocess.CompletedProcess:
        raise NotImplementedError

    def process_exists(self, pid: int) -> bool:
        raise NotImplementedError


class SSHExecutor(Executor):
    def __init__(self, host: str):
        super().__init__(f"SSHExecutor[{host}]")
        self._host = host

    def _run_bash(
        self, command: str, *, snippet=None, silent=False
    ) -> subprocess.CompletedProcess:
        if snippet is None:
            snippet = command
        remote_command = f"bash -lc {shlex.quote(command)}"
        if not silent:
            LOGGER.info(
                "%s executing bash over SSH: %s",
                self.host,
                snippet,
            )
        result = subprocess.run(
            ["ssh", self._host, remote_command],
            text=True,
            capture_output=True,
        )
        if not silent:
            LOGGER.info("%s FINISHED executing bash over SSH: %s", self.host, snippet)
        return result

    def run_python(
        self, script: str, *, check: bool = True, conda: bool = False, silent=False
    ) -> subprocess.CompletedProcess:
        if not silent:
            LOGGER.info("%s preparing remote Python script", self.host)
            LOGGER.debug("%s Python script contents:\n%s", self.host, script)
        heredoc_tag = "__RHL_REMOTE_SCRIPT__"
        if heredoc_tag in script:
            raise LauncherError(
                "Generated Python script unexpectedly contained remote heredoc sentinel."
            )
        command = f"python3 - <<'{heredoc_tag}'\n{script}\n{heredoc_tag}"
        return self.run_shell(command, check=check, conda=conda, silent=silent)

    def process_exists(self, pid: int) -> bool:
        result = self.run_shell(f"ps -p {pid} -o pid=", check=False, silent=True)
        return result.returncode == 0 and result.stdout.strip() != ""


class LocalExecutor(Executor):
    def __init__(self):
        super().__init__("LocalExecutor")

    def _run_bash(
        self, command: str, *, snippet=None, silent=False
    ) -> subprocess.CompletedProcess:
        if snippet is None:
            snippet = command
        if not silent:
            LOGGER.info("%s executing bash locally: %s", self.host, snippet)
        result = subprocess.run(
            ["bash", "-lc", command],
            text=True,
            capture_output=True,
        )
        if not silent:
            LOGGER.info("%s FINISHED executing bash locally: %s", self.host, snippet)
        return result

    def run_python(
        self, script: str, *, check: bool = True, conda: bool = False, silent=False
    ) -> subprocess.CompletedProcess:
        if not silent:
            LOGGER.info("%s preparing local Python script", self.host)
            LOGGER.debug("%s Python script contents:\n%s", self.host, script)
        heredoc_tag = "__RHL_LOCAL_SCRIPT__"
        if heredoc_tag in script:
            raise LauncherError(
                "Generated Python script unexpectedly contained heredoc sentinel."
            )
        command = f"python3 - <<'{heredoc_tag}'\n{script}\n{heredoc_tag}"
        return self.run_shell(command, check=check, conda=conda, silent=silent)

    def process_exists(self, pid: int) -> bool:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid="],
            text=True,
            capture_output=True,
        )
        return result.returncode == 0 and result.stdout.strip() != ""


@dataclasses.dataclass
class RemoteState:
    executor: Executor
    cfg: Configuration
    remote_dir: str

    @property
    def json_path(self) -> str:
        return os.path.join(self.remote_dir, JSON_NAME.format(key=self.cfg.key))

    @property
    def log_path(self) -> str:
        return os.path.join(self.remote_dir, REMOTE_LOG_NAME.format(key=self.cfg.key))

    def exists(self) -> bool:
        script = (
            "import pathlib\n"
            f"path = pathlib.Path({self.json_path!r}).expanduser()\n"
            "import sys\n"
            "sys.exit(0 if path.exists() else 1)\n"
        )
        result = self.executor.run_python(
            script, check=False, conda=bool(self.cfg.conda_env), silent=True
        )
        return result.returncode == 0

    def kill_process(self, pid) -> None:
        script = f"kill -1 {pid}"
        self.executor.run_shell(script, check=False, conda=False, silent=False)

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
        self.executor.run_python(
            script, check=True, conda=bool(self.cfg.conda_env), silent=True
        )

    def read(self) -> Dict[str, Any]:
        script = (
            "import json, pathlib, sys\n"
            f"path = pathlib.Path({self.json_path!r}).expanduser()\n"
            "with path.open('r', encoding='utf-8') as handle:\n"
            "    data = json.load(handle)\n"
            "import json\n"
            "json.dump(data, sys.stdout)\n"
        )
        result = self.executor.run_python(
            script, conda=bool(self.cfg.conda_env), silent=True
        )
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
        result = self.executor.run_python(
            script, conda=bool(self.cfg.conda_env), silent=True
        )
        payload = json.loads(result.stdout)
        return payload

    def read_log(self) -> Optional[str]:
        script = (
            "import json, pathlib, sys, time\n"
            f"path = pathlib.Path({self.json_path!r}).expanduser()\n"
            "stat = path.stat()\n"
            "with path.open('r', encoding='utf-8') as handle:\n"
            "    data = json.load(handle)\n"
            "logfile = data.get('log')\n"
            "if logfile is not None:\n"
            "   print(pathlib.Path(logfile).read_text())\n"
        )
        result = self.executor.run_python(
            script, conda=bool(self.cfg.conda_env), silent=True
        )
        stdout = result.stdout.strip()
        if not stdout:
            return None
        return stdout

    def launch_process(self, conda_base: Optional[str]) -> None:
        command_template = self.cfg.command_template
        namespace = self.cfg.namespace.copy()
        namespace["status_file"] = self.json_path
        evaluated_command = _evaluate_template(command_template, namespace)
        if not isinstance(evaluated_command, str):
            raise LauncherError("The evaluated command must be a string.")
        if not evaluated_command.strip() or "\n" in evaluated_command:
            raise LauncherError("The command must be a plausible bash command.")

        command = evaluated_command
        parameters_literal = (
            json.dumps(self.cfg.file_parameters)
            if self.cfg.file_parameters is not None
            else "null"
        )
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
            (
                "import json, pathlib, tempfile, os, subprocess, sys\n"
                f"remote_dir = pathlib.Path({self.remote_dir!r}).expanduser()\n"
                "remote_dir.mkdir(parents=True, exist_ok=True)\n"
                f"json_path = pathlib.Path({self.json_path!r}).expanduser()\n"
                f"log_file = pathlib.Path({self.log_path!r}).expanduser()\n"
                "try:\n"
                "    log_file.unlink()\n"
                "except FileNotFoundError:\n"
                "    pass\n"
                "stdout_handle = open(log_file.as_posix(), 'ab', buffering=0)\n"
                f"command = {command!r}\n"
                f"workdir = pathlib.Path({self.cfg.workdir!r}).expanduser()\n"
                "workdir.mkdir(parents=True, exist_ok=True)\n"
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
                "    'log': log_file.as_posix(),\n"
                f"    'command': {evaluated_command!r},\n"
                "    'uid': os.getuid(),\n"
                "    'pid': proc.pid,\n"
                "    'status': 'starting',\n"
                "}\n"
                "network_interface = {network!r}\n"
                "if network_interface is not None:\n"
                "    data['network_interface'] = network_interface\n"
                "parameters = json.loads({parameters_literal!r})\n"
                "if parameters is not None:\n"
                "    data['parameters'] = parameters\n"
                "with json_path.open('w', encoding='utf-8') as handle:\n"
                "    json.dump(data, handle)\n"
                "stdout_handle.close()\n"
            )
            .replace("{network!r}", repr(self.cfg.network_interface))
            .replace("{parameters_literal!r}", repr(parameters_literal))
        )
        p = self.executor.run_python(script, conda=bool(self.cfg.conda_env))

    def verify_port_in_use(self, host: str, port: int) -> None:
        script = (
            "import socket, sys, time\n"
            f"host = {host!r}\n"
            f"port = {port}\n"
            "for trial in range(3):\n"
            "    sock = socket.socket()\n"
            "    sock.settimeout(2.0)\n"
            "    try:\n"
            "        sock.connect((host, port))\n"
            "    except Exception as exc:\n"
            "        if trial < 2:\n"
            "            time.sleep(2)\n"
            "            continue\n"
            "        print(str(exc), file=sys.stderr)\n"
            "        sys.exit(1)\n"
            "    finally:\n"
            "        sock.close()\n"
        )
        self.executor.run_python(script, conda=bool(self.cfg.conda_env), silent=True)

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
        self.executor.run_python(script, conda=bool(self.cfg.conda_env))

    def process_exists(self, pid: int) -> bool:
        return self.executor.process_exists(pid)


def _default_client_directory() -> pathlib.Path:
    base = os.environ.get(CONFIG_DIR_ENV)
    if base:
        return pathlib.Path(base).expanduser()
    return pathlib.Path.home() / DEFAULT_DIRNAME / "client"


def _default_server_directory() -> pathlib.Path:
    base = os.environ.get(CONFIG_DIR_ENV)
    if base:
        return pathlib.Path(base)
    return pathlib.Path(f"~/{DEFAULT_DIRNAME}/server")


def _evaluate_template(template: str, namespace: Dict[str, Any]) -> Any:
    expression = f"f{template!r}"
    try:
        code = compile(expression, "<config>", "eval")
        return eval(code, {"__builtins__": {}}, namespace)
    except Exception as exc:
        raise LauncherError(
            f"Failed to evaluate template {expression}: {exc}"
        ) from None


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
    host: str, port: int, handshake: HandshakeConfig | None, trials: int
) -> None:
    url = build_handshake_url(host, port, handshake)
    for trial in range(trials):
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                status = getattr(response, "status", 200)
                if not (200 <= status < 300):
                    raise LauncherError(f"Handshake failed with HTTP status {status}.")
        except urllib.error.URLError as exc:
            if trial < trials - 1:
                time.sleep(3)
                continue
            raise LauncherError(f"Handshake failed: {exc}") from exc


def validate_remote_running_state(data: Dict[str, Any]) -> None:
    port = data.get("port")
    if not isinstance(port, int):
        raise LauncherError("Remote JSON must provide integer port.")
    if port < 1 or port > 65535:
        raise LauncherError("Remote port is out of range.")
    if not isinstance(data.get("command"), str) or not data["command"]:
        raise LauncherError("Remote JSON must include a plausible command string.")
    if "network_interface" in data and not isinstance(data["network_interface"], str):
        raise LauncherError(
            "Remote JSON network_interface must be a string when present."
        )
    if "parameters" in data and not isinstance(data["parameters"], dict):
        raise LauncherError("Remote JSON parameters must be a mapping when present.")
    if not isinstance(data.get("uid"), int):
        raise LauncherError("Remote JSON must include uid.")
    if not isinstance(data.get("pid"), int):
        raise LauncherError("Remote JSON must include pid.")


def allocate_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        _, port = sock.getsockname()
        return int(port)


TUNNEL_MONITOR_SCRIPT = textwrap.dedent(
    """
    import argparse
    import json
    import os
    import signal
    import subprocess
    import sys
    import time


    def write_status(path, state, message=None):
        tmp_path = f"{path}.tmp"
        payload = {"status": state}
        if message:
            payload["message"] = message
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, path)


    def remote_pid_exists(ssh_host, remote_pid):
        result = subprocess.run(
            ["ssh", ssh_host, "kill", "-0", str(remote_pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--ssh-host", required=True)
        parser.add_argument("--remote-host", required=True)
        parser.add_argument("--remote-port", type=int, required=True)
        parser.add_argument("--local-port", type=int, required=True)
        parser.add_argument("--remote-pid", type=int, required=True)
        parser.add_argument("--status-path", required=True)
        args = parser.parse_args()

        try:
            os.unlink(args.status_path)
        except FileNotFoundError:
            pass

        ssh_cmd = [
            "ssh",
            "-N",
            "-o",
            "ExitOnForwardFailure=yes",
            "-L",
            f"{args.local_port}:{args.remote_host}:{args.remote_port}",
            args.ssh_host,
        ]
        try:
            tunnel_proc = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:  # pragma: no cover - platform dependent
            write_status(args.status_path, "error", f"Unable to launch ssh: {exc}")
            return 1

        time.sleep(0.5)
        if tunnel_proc.poll() is not None:
            err = ""
            if tunnel_proc.stderr:
                try:
                    err = tunnel_proc.stderr.read().strip()
                except Exception:  # pragma: no cover - best effort
                    err = ""
            write_status(
                args.status_path,
                "error",
                err or "ssh exited before establishing the tunnel.",
            )
            return 1

        if tunnel_proc.stderr:
            tunnel_proc.stderr.close()

        write_status(args.status_path, "ready", None)

        stop = False

        def request_stop(_signum, _frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

        try:
            while not stop:
                time.sleep(20.0)
                if tunnel_proc.poll() is not None:
                    return 0
                if not remote_pid_exists(args.ssh_host, args.remote_pid):
                    break
        finally:
            if tunnel_proc.poll() is None:
                tunnel_proc.terminate()
                try:
                    tunnel_proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    tunnel_proc.kill()
        return 0


    if __name__ == "__main__":  # pragma: no cover - executed in subprocess
        sys.exit(main())
    """
)


def _wait_for_tunnel_ready(process: subprocess.Popen, status_path: str) -> None:
    start_time = time.time()
    while time.time() - start_time < 15.0:
        if process.poll() is not None:
            raise LauncherError(
                "Tunnel monitor exited before reporting readiness. "
                "Check SSH connectivity or credentials."
            )
        if os.path.exists(status_path):
            try:
                with open(status_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except json.JSONDecodeError:
                time.sleep(0.1)
                continue
            status = payload.get("status")
            if status == "ready":
                return
            if status == "error":
                message = payload.get("message", "Unknown error")
                raise LauncherError(f"Failed to establish SSH tunnel: {message}")
        time.sleep(0.1)
    raise LauncherError("Timed out while waiting for SSH tunnel to become ready.")


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def establish_tunnel(
    ssh_host: str,
    remote_host: str,
    remote_port: int,
    local_port: int,
    remote_pid: int,
) -> None:
    status_handle = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
    status_path = status_handle.name
    status_handle.close()
    try:
        os.unlink(status_path)
    except FileNotFoundError:
        pass
    cmd = [
        sys.executable,
        "-c",
        TUNNEL_MONITOR_SCRIPT,
        "--ssh-host",
        ssh_host,
        "--remote-host",
        remote_host,
        "--remote-port",
        str(remote_port),
        "--local-port",
        str(local_port),
        "--remote-pid",
        str(remote_pid),
        "--status-path",
        status_path,
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        _wait_for_tunnel_ready(process, status_path)
    except Exception:
        _terminate_process(process)
        raise
    finally:
        try:
            os.unlink(status_path)
        except FileNotFoundError:
            pass


def _fetch_conda_base(executor: Executor) -> Optional[str]:
    try:
        base_result = executor.run_shell("conda info --base", check=False, conda=True)
    except LauncherError:
        return None
    if base_result.returncode != 0:
        return None
    base_path = base_result.stdout.strip()
    if not base_path:
        raise LauncherError("Unable to discover conda base path.")
    return base_path


def _conda_env_exists(executor: Executor, env: str) -> bool:
    result = executor.run_shell("conda env list --json", check=True, conda=True)
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
                pid = data["pid"]
                target_host = data.get("network_interface", cfg.network_interface)
                try:
                    remote.verify_port_in_use(target_host, data["port"])
                except LauncherError as exc:
                    traceback.print_exc()
                    remote.kill_process(pid)
                    remote.remove()
                    continue

                for trial in range(5):
                    try:
                        remote.handshake(target_host, data["port"], cfg.handshake)
                    except Exception as exc:
                        if trial < 4:
                            time.sleep(3)
                            continue
                        traceback.print_exc()
                        remote.kill_process(pid)
                        remote.remove()
                        continue

                return data
            if status == "starting" and isinstance(data.get("pid"), int):
                monitored = monitor_remote_start(cfg, remote, data)
                if monitored is not None:
                    return monitored
                log = remote.read_log()
                if log is not None:
                    LOGGER.error(
                        """Remote process FAILED.
                                Log contents:
                                """
                        + log
                    )
                remote.remove()
                if attempted_launch:
                    raise LauncherError(
                        "Remote process did not finish starting after relaunch."
                    )
                continue
            remote.remove()
            attempted_launch = True
            continue
        if attempted_launch:
            raise LauncherError("Remote launch already attempted and failed.")
        conda_base = None
        if cfg.conda_env:
            conda_base = _fetch_conda_base(remote.executor)
            if not conda_base:
                raise LauncherError(
                    "Conda environment requested but conda is not installed remotely."
                )
            if not _conda_env_exists(remote.executor, cfg.conda_env):
                raise LauncherError(
                    f"Remote conda environment '{cfg.conda_env}' does not exist."
                )
        remote.launch_process(conda_base)
        attempted_launch = True
        for _ in range(10):
            if remote.exists():
                break
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
            target_host = data.get("network_interface", cfg.network_interface)
            remote.verify_port_in_use(target_host, data["port"])
            for trial in range(3):
                try:
                    remote.handshake(target_host, data["port"], cfg.handshake)
                except Exception as exc:
                    if trial < 2:
                        time.sleep(2)
                        continue
                    raise exc from None
            return data
        mtime = float(payload["mtime"])
        if mtime != last_mtime:
            last_mtime = mtime
            last_change = time.time()
            continue
        t = time.time()
        max_elapse = max(t - last_change, t - last_mtime)
        if max_elapse > 60:
            return None
        elif max_elapse > 10 and not remote.process_exists(pid):
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
        perform_local_handshake(hostname, port, cfg.handshake, trials=1)
        return existing
    except LauncherError:
        local_state.delete()
        return None


def create_local_file(
    cfg: Configuration,
    local_state: LocalState,
    remote_data: Dict[str, Any],
) -> Dict[str, Any]:
    port = int(remote_data["port"])
    if cfg.tunnel:
        if cfg.hostname is None or cfg.ssh_hostname is None:
            raise LauncherError(
                "SSH tunneling requires hostname and ssh_hostname to be configured."
            )
        local_port = allocate_local_port()
        establish_tunnel(
            cfg.ssh_hostname, cfg.hostname, port, local_port, int(remote_data["pid"])
        )
        payload: Dict[str, Any] = {
            "hostname": "localhost",
            "port": local_port,
            "tunneled-host": cfg.ssh_hostname,
            "tunneled-port": port,
            "tunneled-network-interface": remote_data.get(
                "network_interface", cfg.network_interface
            ),
        }
    else:
        target_host = (
            cfg.hostname
            if cfg.hostname
            else remote_data.get("network_interface", cfg.network_interface)
        )
        if not isinstance(target_host, str) or not target_host:
            raise LauncherError("Unable to determine hostname for local connection.")
        payload = {"hostname": target_host, "port": port}
        if cfg.hostname and cfg.ssh_hostname and cfg.ssh_hostname != cfg.hostname:
            payload["ssh_hostname"] = cfg.ssh_hostname
    local_state.write(payload)
    perform_local_handshake(
        payload["hostname"], payload["port"], cfg.handshake, trials=5
    )
    return payload


def _execute(
    cfg: Configuration, connection_dir: Optional[pathlib.Path]
) -> Dict[str, Any]:
    local_state = LocalState.build(cfg, connection_dir)

    try:
        local_result = handle_local(cfg, local_state)
        if local_result:
            return local_result
    except LauncherError:
        pass

    remote_dir = _default_server_directory()
    if cfg.hostname:
        if not cfg.ssh_hostname:
            raise LauncherError(
                "ssh_hostname must be provided when hostname is configured."
            )
        executor: Executor = SSHExecutor(cfg.ssh_hostname)
    else:
        executor = LocalExecutor()
    remote = RemoteState(executor, cfg, remote_dir.as_posix())
    remote_result = handle_remote(cfg, remote)

    return create_local_file(cfg, local_state, remote_result)


def run(
    config: Dict[str, Any],
    *,
    connection_dir: pathlib.Path | str | None = None,
) -> Dict[str, Any]:
    """Programmatic API returning the local connection payload.

    Connection dir defaults to ~/.remote-http-launcher/client
    """
    _configure_module_logger()
    cfg = Configuration.from_mapping(config)
    override_dir: Optional[pathlib.Path]
    if connection_dir is None:
        override_dir = None
    else:
        override_dir = pathlib.Path(connection_dir).expanduser()
    return _execute(cfg, override_dir)


def main(argv: list[str] | None = None) -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    parser = argparse.ArgumentParser(description="Remote HTTP launcher.")
    parser.add_argument(
        "config", type=pathlib.Path, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--connection-dir",
        type=pathlib.Path,
        help="Override the connection directory. Defaults to ~/.remote-http-launcher/client",
    )
    args = parser.parse_args(argv)

    cfg = Configuration.from_yaml(args.config)
    _execute(cfg, args.connection_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
