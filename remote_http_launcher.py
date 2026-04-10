#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import enum
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
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable, Dict, Optional, Protocol, TextIO, Tuple

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


class Phase(enum.Enum):
    LOCAL_CHECK = "local_check"
    CONNECT = "connect"
    REMOTE_CHECK = "remote_check"
    REMOTE_VALIDATE = "remote_validate"
    STALE_CLEANUP = "stale_cleanup"
    CONDA_SETUP = "conda_setup"
    LAUNCH = "launch"
    WAIT_FOR_START = "wait_for_start"
    TUNNEL_SETUP = "tunnel_setup"
    LOCAL_VALIDATE = "local_validate"
    DONE = "done"


class LauncherObserver(Protocol):
    def on_phase(self, phase: Phase, result: str) -> None:
        ...

    def on_detail(self, message: str) -> None:
        ...

    def on_command(self, host: str, command: str, full_command: str) -> None:
        ...

    def on_command_done(self, host: str, command: str) -> None:
        ...

    def on_script(self, host: str, script: str) -> None:
        ...

    def on_error(self, message: str) -> None:
        ...


class NullObserver:
    def on_phase(self, phase: Phase, result: str) -> None:
        return

    def on_detail(self, message: str) -> None:
        return

    def on_command(self, host: str, command: str, full_command: str) -> None:
        return

    def on_command_done(self, host: str, command: str) -> None:
        return

    def on_script(self, host: str, script: str) -> None:
        return

    def on_error(self, message: str) -> None:
        return


class CompositeObserver:
    def __init__(self, observers: list[LauncherObserver]):
        self._observers = observers

    def on_phase(self, phase: Phase, result: str) -> None:
        for observer in self._observers:
            observer.on_phase(phase, result)

    def on_detail(self, message: str) -> None:
        for observer in self._observers:
            observer.on_detail(message)

    def on_command(self, host: str, command: str, full_command: str) -> None:
        for observer in self._observers:
            observer.on_command(host, command, full_command)

    def on_command_done(self, host: str, command: str) -> None:
        for observer in self._observers:
            observer.on_command_done(host, command)

    def on_script(self, host: str, script: str) -> None:
        for observer in self._observers:
            observer.on_script(host, script)

    def on_error(self, message: str) -> None:
        for observer in self._observers:
            observer.on_error(message)


class BasicObserver:
    def __init__(self, stream: TextIO):
        self.stream = stream

    def on_phase(self, phase: Phase, result: str) -> None:
        print(f"{phase.value}: {result}", file=self.stream, flush=True)

    def on_detail(self, message: str) -> None:
        return

    def on_command(self, host: str, command: str, full_command: str) -> None:
        return

    def on_command_done(self, host: str, command: str) -> None:
        return

    def on_script(self, host: str, script: str) -> None:
        return

    def on_error(self, message: str) -> None:
        print(f"error: {message}", file=self.stream, flush=True)


class DebugObserver(BasicObserver):
    def on_detail(self, message: str) -> None:
        print(f"detail: {message}", file=self.stream, flush=True)

    def on_command(self, host: str, command: str, full_command: str) -> None:
        print(f"command[{host}]: {command}", file=self.stream, flush=True)
        if full_command != command:
            print(
                f"full-command[{host}]: {full_command}",
                file=self.stream,
                flush=True,
            )

    def on_command_done(self, host: str, command: str) -> None:
        print(f"command-done[{host}]: {command}", file=self.stream, flush=True)

    def on_script(self, host: str, script: str) -> None:
        print(f"script[{host}]:", file=self.stream, flush=True)
        print(script, file=self.stream, flush=True)


class MinimalObserver:
    def __init__(
        self,
        key: str,
        expected: list[Phase],
        *,
        stream: TextIO,
        clock: Callable[[], float] = time.monotonic,
        threshold: float = 1.5,
    ):
        self.key = key
        self.expected = expected
        self._stream = stream
        self._clock = clock
        self._threshold = threshold
        self._start = clock()
        self._rendered = False
        self._index = {phase: idx for idx, phase in enumerate(expected)}
        self._max_seen_index = -1
        self._max_seen_phase: Phase | None = None
        isatty = getattr(stream, "isatty", None)
        self._tty = bool(isatty and isatty())

    def on_phase(self, phase: Phase, result: str) -> None:
        if phase is not Phase.DONE:
            current_index = self._index.get(phase, len(self.expected) - 1)
            if current_index >= self._max_seen_index:
                self._max_seen_index = current_index
                self._max_seen_phase = phase
        elapsed = self._clock() - self._start
        if not self._rendered:
            if phase is Phase.DONE and elapsed < self._threshold:
                return
            if elapsed < self._threshold:
                return
            self._rendered = True
        if phase is Phase.DONE:
            self._finish(elapsed)
            return
        if self._max_seen_phase is None:
            return
        if phase is not self._max_seen_phase:
            return
        self._render(self._max_seen_phase)

    def on_detail(self, message: str) -> None:
        return

    def on_command(self, host: str, command: str, full_command: str) -> None:
        return

    def on_command_done(self, host: str, command: str) -> None:
        return

    def on_script(self, host: str, script: str) -> None:
        return

    def on_error(self, message: str) -> None:
        if self._rendered and self._tty:
            print("\r" + " " * 120 + "\r", end="", file=self._stream, flush=True)
        print(f"  {self.key}: error: {message}", file=self._stream, flush=True)

    def _render(self, phase: Phase) -> None:
        current = self._index.get(phase, len(self.expected) - 1)
        total = max(1, len(self.expected))
        width = 20
        filled = max(1, min(width, int(round(width * (current + 1) / total))))
        bar = "#" * filled + "·" * (width - filled)
        line = f"  {self.key} [{bar}] {phase.value}"
        if self._tty:
            print("\r" + line.ljust(120), end="", file=self._stream, flush=True)
        else:
            print(line, file=self._stream, flush=True)

    def _finish(self, elapsed: float) -> None:
        if self._tty:
            print("\r" + " " * 120 + "\r", end="", file=self._stream, flush=True)
        print(
            f"  {self.key}: ready ({elapsed:.1f}s)",
            file=self._stream,
            flush=True,
        )


@dataclasses.dataclass
class ObserverBundle:
    observer: LauncherObserver
    close: Callable[[], None]


@dataclasses.dataclass
class LoggingConfig:
    level: str = "basic"
    log_file: Optional[str] = None
    debug_log_file: Optional[str] = None


def _command_snippet(command: str) -> str:
    snippet = command[:30]
    if len(command) > 30:
        snippet += " [...]"
        if len(command) > 65:
            snippet += command[-30:]
    return snippet


def _build_expected_phases(cfg: "Configuration") -> list[Phase]:
    phases = [
        Phase.LOCAL_CHECK,
        Phase.CONNECT,
        Phase.REMOTE_CHECK,
        Phase.REMOTE_VALIDATE,
        Phase.STALE_CLEANUP,
    ]
    if cfg.conda_env:
        phases.append(Phase.CONDA_SETUP)
    phases.extend([Phase.LAUNCH, Phase.WAIT_FOR_START])
    if cfg.tunnel:
        phases.append(Phase.TUNNEL_SETUP)
    phases.extend([Phase.LOCAL_VALIDATE, Phase.DONE])
    return phases


def _open_log_stream(path: str) -> TextIO:
    log_path = pathlib.Path(path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path.open("a", encoding="utf-8", buffering=1)


def _build_observer_bundle(
    key: str,
    cfg: "Configuration",
    logging_config: LoggingConfig,
    *,
    stream: TextIO | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ObserverBundle:
    target_stream = stream or sys.stderr
    opened: list[TextIO] = []
    observers: list[LauncherObserver] = []

    def _maybe_open(path: Optional[str]) -> TextIO | None:
        if path is None:
            return None
        handle = _open_log_stream(path)
        opened.append(handle)
        return handle

    if logging_config.level == "minimal":
        observers.append(
            MinimalObserver(
                key,
                _build_expected_phases(cfg),
                stream=target_stream,
                clock=clock,
            )
        )
        basic_stream = _maybe_open(logging_config.log_file)
        if basic_stream is not None:
            observers.append(BasicObserver(basic_stream))
        debug_stream = _maybe_open(logging_config.debug_log_file)
        if debug_stream is not None:
            observers.append(DebugObserver(debug_stream))
    elif logging_config.level == "basic":
        observers.append(BasicObserver(target_stream))
        debug_stream = _maybe_open(logging_config.debug_log_file)
        if debug_stream is not None:
            observers.append(DebugObserver(debug_stream))
    else:
        observers.append(DebugObserver(target_stream))

    observer: LauncherObserver
    if not observers:
        observer = NullObserver()
    elif len(observers) == 1:
        observer = observers[0]
    else:
        observer = CompositeObserver(observers)

    def _close() -> None:
        for handle in opened:
            handle.close()

    return ObserverBundle(observer=observer, close=_close)


def _extract_logging_config(
    config: Dict[str, Any],
    *,
    log_level: str | None = None,
    log_file: str | None = None,
    debug_log_file: str | None = None,
) -> tuple[Dict[str, Any], LoggingConfig]:
    data = dict(config)
    config_level = data.pop("log_level", "basic")
    config_log_file = data.pop("log_file", None)
    config_debug_log_file = data.pop("debug_log_file", None)
    level_value = log_level if log_level is not None else config_level
    log_file_value = log_file if log_file is not None else config_log_file
    debug_log_file_value = (
        debug_log_file if debug_log_file is not None else config_debug_log_file
    )
    if level_value not in ("minimal", "basic", "debug"):
        raise LauncherError("log_level must be one of: minimal, basic, debug.")
    if log_file_value is not None and not isinstance(log_file_value, str):
        raise LauncherError("log_file must be a string when provided.")
    if debug_log_file_value is not None and not isinstance(debug_log_file_value, str):
        raise LauncherError("debug_log_file must be a string when provided.")
    return data, LoggingConfig(
        level=level_value,
        log_file=log_file_value,
        debug_log_file=debug_log_file_value,
    )


def _load_yaml_mapping(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise LauncherError("Configuration mapping must be a dictionary.")
    return data


@dataclasses.dataclass
class HandshakeConfig:
    path: str = ""
    parameters: Dict[str, str] = dataclasses.field(default_factory=dict)
    port_name: str | None = None


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
        return Configuration.from_mapping(_load_yaml_mapping(path))

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
            port_name = value.get("port_name")
            if port_name is not None and not isinstance(port_name, str):
                raise LauncherError("handshake.port_name must be a string.")
            return HandshakeConfig(
                path=path_value, parameters=normalized, port_name=port_name
            )
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
        try:
            with self.json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise LauncherError(
                f"Malformed local connection file: {self.json_path}"
            ) from exc
        return data

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
    def __init__(self, host: str, observer: LauncherObserver):
        self.host = host
        self.observer = observer
        self._conda_checked = False
        self._conda_source: Optional[str] = None
        self._conda_unavailable = False

    def run_shell(
        self, command: str, *, check: bool = True, conda: bool = False
    ) -> subprocess.CompletedProcess:
        snippet = _command_snippet(command)
        final_command = command
        if conda:
            conda_source = self._ensure_conda_setup()
            if conda_source:
                final_command = f"source {shlex.quote(conda_source)} && {final_command}"
        self.observer.on_command(self.host, snippet, final_command)
        result = self._run_bash(final_command, snippet=snippet)
        self.observer.on_command_done(self.host, snippet)
        if check and result.returncode != 0:
            raise LauncherError(
                f"{self.host} command failed: {final_command}\n{result.stderr.strip()}"
            )
        return result

    def _run_bash(self, command: str, *, snippet=None) -> subprocess.CompletedProcess:
        raise NotImplementedError

    def ensure_conda_setup(self) -> Optional[str]:
        return self._ensure_conda_setup()

    def _ensure_conda_setup(self) -> Optional[str]:
        if self._conda_unavailable:
            raise LauncherError(
                "Conda is not installed on the target host or could not be initialized."
            )
        if self._conda_checked:
            return self._conda_source
        self._conda_checked = True
        self.observer.on_detail(f"{self.host} probing for conda on PATH")
        probe = self._run_bash("command -v conda >/dev/null 2>&1")
        if probe.returncode == 0:
            self.observer.on_detail(f"{self.host} found conda on PATH")
            self._conda_source = None
            return None
        conda_source = self._discover_conda_from_bashrc()
        if not conda_source:
            self._conda_unavailable = True
            raise LauncherError(
                "Conda is not available on the target host and ~/.bashrc does not "
                "contain a conda initialize block."
            )
        self.observer.on_detail(
            f"{self.host} discovered conda.sh via ~/.bashrc at {conda_source}"
        )
        self._conda_source = conda_source
        return self._conda_source

    def _discover_conda_from_bashrc(self) -> Optional[str]:
        self.observer.on_detail(
            f"{self.host} scraping ~/.bashrc for conda initialize block"
        )
        bashrc = self._run_bash("cat ~/.bashrc")
        if bashrc.returncode != 0:
            self.observer.on_detail(
                f"{self.host} unable to read ~/.bashrc (exit {bashrc.returncode})"
            )
            return None
        text = bashrc.stdout
        block_match = CONDA_INIT_BLOCK_RE.search(text)
        if not block_match:
            self.observer.on_detail(
                f"{self.host} no conda initialize block found in ~/.bashrc"
            )
            return None
        block = block_match.group(1)
        path_match = CONDA_PATH_RE.search(block)
        if not path_match:
            self.observer.on_detail(
                f"{self.host} conda initialize block missing conda.sh path"
            )
            return None
        return path_match.group(1)

    def run_python(
        self, script: str, *, check: bool = True, conda: bool = False
    ) -> subprocess.CompletedProcess:
        raise NotImplementedError

    def process_exists(self, pid: int) -> bool:
        raise NotImplementedError


class SSHExecutor(Executor):
    def __init__(self, host: str, observer: LauncherObserver):
        super().__init__(f"SSHExecutor[{host}]", observer)
        self._host = host

    def _run_bash(self, command: str, *, snippet=None) -> subprocess.CompletedProcess:
        if snippet is None:
            snippet = command
        remote_command = f"bash -lc {shlex.quote(command)}"
        result = subprocess.run(
            ["ssh", self._host, remote_command],
            text=True,
            capture_output=True,
        )
        return result

    def run_python(
        self, script: str, *, check: bool = True, conda: bool = False
    ) -> subprocess.CompletedProcess:
        self.observer.on_script(self.host, script)
        heredoc_tag = "__RHL_REMOTE_SCRIPT__"
        if heredoc_tag in script:
            raise LauncherError(
                "Generated Python script unexpectedly contained remote heredoc sentinel."
            )
        command = f"python3 - <<'{heredoc_tag}'\n{script}\n{heredoc_tag}"
        return self.run_shell(command, check=check, conda=conda)

    def process_exists(self, pid: int) -> bool:
        result = self.run_shell(f"ps -p {pid} -o pid=", check=False)
        return result.returncode == 0 and result.stdout.strip() != ""


class LocalExecutor(Executor):
    def __init__(self, observer: LauncherObserver):
        super().__init__("LocalExecutor", observer)

    def _run_bash(self, command: str, *, snippet=None) -> subprocess.CompletedProcess:
        if snippet is None:
            snippet = command
        result = subprocess.run(
            ["bash", "-lc", command],
            text=True,
            capture_output=True,
        )
        return result

    def run_python(
        self, script: str, *, check: bool = True, conda: bool = False
    ) -> subprocess.CompletedProcess:
        self.observer.on_script(self.host, script)
        heredoc_tag = "__RHL_LOCAL_SCRIPT__"
        if heredoc_tag in script:
            raise LauncherError(
                "Generated Python script unexpectedly contained heredoc sentinel."
            )
        command = f"python3 - <<'{heredoc_tag}'\n{script}\n{heredoc_tag}"
        return self.run_shell(command, check=check, conda=conda)

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
    def observer(self) -> LauncherObserver:
        return self.executor.observer

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
            "import sys, os, json\n"
            "if not path.exists():\n"
            "    sys.exit(1)\n"
            "with path.open('r', encoding='utf-8') as handle:\n"
            "    data = json.load(handle)\n"
            "if isinstance(data, dict) and data.get('dry_run'):\n"
            "    os.remove(path)\n"
            "    sys.exit(1)\n"
            "sys.exit(0)\n"
        )
        result = self.executor.run_python(script, check=False, conda=bool(self.cfg.conda_env))
        return result.returncode == 0

    def kill_process(self, pid) -> None:
        script = f"kill -1 {pid}"
        self.executor.run_shell(script, check=False, conda=False)

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
        self.executor.run_python(script, check=True, conda=bool(self.cfg.conda_env))

    def read(self) -> Dict[str, Any]:
        script = (
            "import json, pathlib, sys\n"
            f"path = pathlib.Path({self.json_path!r}).expanduser()\n"
            "with path.open('r', encoding='utf-8') as handle:\n"
            "    data = json.load(handle)\n"
            "import json\n"
            "json.dump(data, sys.stdout)\n"
        )
        result = self.executor.run_python(script, conda=bool(self.cfg.conda_env))
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
        result = self.executor.run_python(script, conda=bool(self.cfg.conda_env))
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
        result = self.executor.run_python(script, conda=bool(self.cfg.conda_env))
        stdout = result.stdout.strip()
        if not stdout:
            return None
        return stdout

    def evaluate_command(self) -> str:
        command_template = self.cfg.command_template
        namespace = self.cfg.namespace.copy()
        namespace["status_file"] = self.json_path
        evaluated_command = _evaluate_template(command_template, namespace)
        if not isinstance(evaluated_command, str):
            raise LauncherError("The evaluated command must be a string.")
        if not evaluated_command.strip() or "\n" in evaluated_command:
            raise LauncherError("The command must be a plausible bash command.")
        return evaluated_command

    def launch_process(self, conda_base: Optional[str]) -> None:
        evaluated_command = self.evaluate_command()
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
        self.executor.run_python(script, conda=bool(self.cfg.conda_env))

    def write_dry_run_metadata(self, evaluated_command: str) -> None:
        remote_dir = pathlib.Path(self.remote_dir).expanduser()
        remote_dir.mkdir(parents=True, exist_ok=True)
        json_path = pathlib.Path(self.json_path).expanduser()
        log_path = pathlib.Path(self.log_path).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_path.exists():
            log_path.touch()
        parameters_literal = (
            json.dumps(self.cfg.file_parameters)
            if self.cfg.file_parameters is not None
            else "null"
        )
        parameters = json.loads(parameters_literal)
        data = {
            "workdir": self.cfg.workdir,
            "log": log_path.as_posix(),
            "command": evaluated_command,
            "network_interface": self.cfg.network_interface,
            "parameters": parameters,
            "dry_run": True,
        }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle)

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
        self.executor.run_python(script, conda=bool(self.cfg.conda_env))

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
    host: str,
    port: int,
    handshake: HandshakeConfig | None,
    trials: int,
    *,
    observer: LauncherObserver | None = None,
) -> None:
    url = build_handshake_url(host, port, handshake)
    for trial in range(trials):
        if observer is not None:
            observer.on_detail(
                f"local handshake attempt {trial + 1}/{trials}: {url}"
            )
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                status = getattr(response, "status", 200)
                if not (200 <= status < 300):
                    raise LauncherError(f"Handshake failed with HTTP status {status}.")
            break
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


def _remote_handshake_port(cfg: Configuration, data: Dict[str, Any]) -> int:
    if cfg.handshake is None:
        return int(data["port"])
    port_name = cfg.handshake.port_name
    if port_name is not None:
        return int(data[port_name])
    return int(data["port"])


def _cleanup_remote_state(
    remote: RemoteState, *, result: str, pid: int | None = None
) -> None:
    if pid is not None:
        remote.kill_process(pid)
    remote.remove()
    remote.observer.on_phase(Phase.STALE_CLEANUP, result)


def _validate_existing_remote(
    cfg: Configuration,
    remote: RemoteState,
    data: Dict[str, Any],
    *,
    emit_phase: bool = True,
) -> Optional[Dict[str, Any]]:
    validate_remote_running_state(data)
    pid = data["pid"]
    target_host = data.get("network_interface", cfg.network_interface)
    remote.observer.on_detail(f"verify remote port {target_host}:{data['port']}")
    try:
        remote.verify_port_in_use(target_host, data["port"])
    except LauncherError:
        if emit_phase:
            remote.observer.on_phase(Phase.REMOTE_VALIDATE, "port-failed")
        else:
            remote.observer.on_detail("post-launch remote port validation failed")
        _cleanup_remote_state(remote, result="kill+remove", pid=pid)
        return None

    if cfg.handshake is not None:
        handshake_port = _remote_handshake_port(cfg, data)
        for trial in range(15):
            remote.observer.on_detail(
                f"remote handshake attempt {trial + 1}/15: {target_host}:{handshake_port}"
            )
            try:
                remote.handshake(target_host, handshake_port, cfg.handshake)
                break
            except Exception:
                if trial < 14:
                    time.sleep(1)
                    continue
                if emit_phase:
                    remote.observer.on_phase(Phase.REMOTE_VALIDATE, "handshake-failed")
                else:
                    remote.observer.on_detail("post-launch remote handshake failed")
                _cleanup_remote_state(remote, result="kill+remove", pid=pid)
                return None

    if emit_phase:
        remote.observer.on_phase(Phase.REMOTE_VALIDATE, "ok")
    return data


def handle_remote(
    cfg: Configuration, remote: RemoteState, *, dry_run: bool = False
) -> Tuple[Dict[str, Any], str]:
    attempted_launch = False
    while True:
        if remote.exists():
            data = remote.read()
            status = data.get("status")
            remote.observer.on_phase(Phase.REMOTE_CHECK, f"found:{status}")
            if status == "running":
                validated = _validate_existing_remote(cfg, remote, data)
                if validated is None:
                    continue
                return validated, "reused-remote"
            if status == "starting" and isinstance(data.get("pid"), int):
                remote.observer.on_phase(Phase.WAIT_FOR_START, "waiting")
                monitored = monitor_remote_start(cfg, remote, data)
                if monitored is not None:
                    return monitored, "reused-remote"
                log = remote.read_log()
                if log is not None:
                    remote.observer.on_detail(f"remote process failed log:\n{log}")
                _cleanup_remote_state(
                    remote,
                    result="remove+log" if log is not None else "remove",
                )
                if attempted_launch:
                    raise LauncherError(
                        "Remote process did not finish starting after relaunch."
                    )
                continue
            _cleanup_remote_state(remote, result="remove")
            attempted_launch = True
            continue
        remote.observer.on_phase(Phase.REMOTE_CHECK, "missing")
        if attempted_launch:
            raise LauncherError("Remote launch already attempted and failed.")
        conda_base = None
        if cfg.conda_env and not dry_run:
            remote.observer.on_detail("discover remote conda base and environment")
            conda_base = _fetch_conda_base(remote.executor)
            if not conda_base:
                raise LauncherError(
                    "Conda environment requested but conda is not installed remotely."
                )
            if not _conda_env_exists(remote.executor, cfg.conda_env):
                raise LauncherError(
                    f"Remote conda environment '{cfg.conda_env}' does not exist."
                )
            remote.observer.on_phase(Phase.CONDA_SETUP, "ok")
        if dry_run:
            evaluated_command = remote.evaluate_command()
            remote.write_dry_run_metadata(evaluated_command)
            print(evaluated_command)
            raise SystemExit(0)
        remote.observer.on_phase(Phase.LAUNCH, "started")
        remote.launch_process(conda_base)
        attempted_launch = True
        remote.observer.on_phase(Phase.WAIT_FOR_START, "waiting")
        for poll in range(10):
            remote.observer.on_detail(f"wait_for_start exists poll {poll + 1}/10")
            if remote.exists():
                data = remote.read()
                status = data.get("status")
                remote.observer.on_detail(f"wait_for_start status={status}")
                if status == "starting" and isinstance(data.get("pid"), int):
                    monitored = monitor_remote_start(cfg, remote, data)
                    if monitored is not None:
                        return monitored, "launched"
                    log = remote.read_log()
                    if log is not None:
                        remote.observer.on_detail(f"remote process failed log:\n{log}")
                    _cleanup_remote_state(
                        remote,
                        result="remove+log" if log is not None else "remove",
                    )
                    raise LauncherError(
                        "Remote process did not finish starting after relaunch."
                    )
                if status == "running":
                    validated = _validate_existing_remote(
                        cfg,
                        remote,
                        data,
                        emit_phase=False,
                    )
                    if validated is not None:
                        return validated, "launched"
                    raise LauncherError("Remote launch already attempted and failed.")
                _cleanup_remote_state(remote, result="remove")
                raise LauncherError("Remote launch already attempted and failed.")
            time.sleep(2.0)
        raise LauncherError("Remote launch already attempted and failed.")


def monitor_remote_start(
    cfg: Configuration, remote: RemoteState, initial: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    payload = remote.stat_and_read()
    last_mtime = float(payload["mtime"])
    last_change = time.time()
    pid = int(initial["pid"])
    while True:
        time.sleep(1.0)
        payload = remote.stat_and_read()
        data = payload["data"]
        status = data.get("status")
        remote.observer.on_detail(
            f"monitor_remote_start status={status} mtime={float(payload['mtime']):.3f}"
        )
        if status == "failed":
            return None
        if status == "running":
            validate_remote_running_state(data)
            target_host = data.get("network_interface", cfg.network_interface)
            remote.observer.on_detail(
                f"verify remote port {target_host}:{data['port']} while starting"
            )
            remote.verify_port_in_use(target_host, data["port"])
            if cfg.handshake is not None:
                handshake_port = _remote_handshake_port(cfg, data)
                for trial in range(6):
                    remote.observer.on_detail(
                        f"startup handshake attempt {trial + 1}/6: {target_host}:{handshake_port}"
                    )
                    try:
                        remote.handshake(target_host, handshake_port, cfg.handshake)
                        break
                    except Exception:
                        if trial < 5:
                            time.sleep(1)
                            continue
                        raise
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
    cfg: Configuration, local_state: LocalState, observer: LauncherObserver
) -> Optional[Dict[str, Any]]:
    try:
        existing = local_state.read()
    except LauncherError:
        observer.on_phase(Phase.LOCAL_CHECK, "malformed")
        raise
    if existing is None:
        observer.on_phase(Phase.LOCAL_CHECK, "missing")
        return None
    if cfg.handshake is None:
        observer.on_phase(Phase.LOCAL_CHECK, "reused")
        return existing

    port_name = cfg.handshake.port_name
    if port_name is None:
        port_name = "port"
    port = existing.get(port_name)
    if not isinstance(port, int):
        raise LauncherError(f"Local JSON must contain integer {port_name}.")
    hostname = existing.get("hostname")
    if not isinstance(hostname, str):
        raise LauncherError("Local JSON must contain hostname.")
    try:
        perform_local_handshake(
            hostname,
            port,
            cfg.handshake,
            trials=1,
            observer=observer,
        )
        observer.on_phase(Phase.LOCAL_CHECK, "reused")
        return existing
    except LauncherError:
        local_state.delete()
        observer.on_phase(Phase.LOCAL_CHECK, "stale")
        return None


def _gather_port_parameters(parameters: Dict[str, Any]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for name, value in parameters.items():
        if not isinstance(name, str) or (
            not name.endswith("_port") and not name.endswith("-port")
        ):
            continue
        if isinstance(value, bool):
            continue
        try:
            port = int(value)
        except (TypeError, ValueError):
            continue
        if port < 1 or port > 65535:
            continue
        result[name] = port
    return result


def create_local_file(
    cfg: Configuration,
    local_state: LocalState,
    remote_data: Dict[str, Any],
    observer: LauncherObserver,
) -> Dict[str, Any]:
    port = int(remote_data["port"])
    remote_pid = int(remote_data["pid"])
    remote_parameters = remote_data.get("parameters")
    parameters_payload: Dict[str, Any] | None = None
    if isinstance(remote_parameters, dict):
        parameters_payload = dict(remote_parameters)
    if cfg.tunnel:
        if cfg.hostname is None or cfg.ssh_hostname is None:
            raise LauncherError(
                "SSH tunneling requires hostname and ssh_hostname to be configured."
            )
        local_port = allocate_local_port()
        observer.on_phase(
            Phase.TUNNEL_SETUP,
            f"localhost:{local_port} -> {cfg.hostname}:{port}",
        )
        establish_tunnel(cfg.ssh_hostname, cfg.hostname, port, local_port, remote_pid)
        payload: Dict[str, Any] = {
            "hostname": "localhost",
            "port": local_port,
            "tunneled-host": cfg.ssh_hostname,
            "tunneled-port": port,
            "tunneled-network-interface": remote_data.get(
                "network_interface", cfg.network_interface
            ),
        }
        tunnel_ports = _gather_port_parameters(remote_data)
        for name, remote_param_port in tunnel_ports.items():
            local_param_port = allocate_local_port()
            observer.on_detail(
                f"extra tunnel {name}: localhost:{local_param_port} -> {cfg.hostname}:{remote_param_port}"
            )
            establish_tunnel(
                cfg.ssh_hostname,
                cfg.hostname,
                remote_param_port,
                local_param_port,
                remote_pid,
            )
            payload[name] = local_param_port
            payload[f"tunneled-{name}"] = remote_param_port
    else:
        port_parameters = _gather_port_parameters(remote_data)
        target_host = (
            cfg.hostname
            if cfg.hostname
            else remote_data.get("network_interface", cfg.network_interface)
        )
        if not isinstance(target_host, str) or not target_host:
            raise LauncherError("Unable to determine hostname for local connection.")
        payload = {"hostname": target_host, "port": port}
        payload.update(port_parameters)
        if cfg.hostname and cfg.ssh_hostname and cfg.ssh_hostname != cfg.hostname:
            payload["ssh_hostname"] = cfg.ssh_hostname
    if parameters_payload is not None:
        payload["parameters"] = parameters_payload
    local_state.write(payload)
    port_name = "port"
    if cfg.handshake is not None and cfg.handshake.port_name is not None:
        port_name = cfg.handshake.port_name
    perform_local_handshake(
        payload["hostname"],
        payload[port_name],
        cfg.handshake,
        trials=5,
        observer=observer,
    )
    observer.on_phase(Phase.LOCAL_VALIDATE, "ok")
    return payload


def _execute(
    cfg: Configuration,
    connection_dir: Optional[pathlib.Path],
    *,
    observer: LauncherObserver,
    dry_run: bool = False,
) -> Dict[str, Any]:
    local_state = LocalState.build(cfg, connection_dir)

    try:
        local_result = handle_local(cfg, local_state, observer)
        if local_result:
            observer.on_phase(Phase.DONE, "reused-local")
            return local_result
    except LauncherError:
        pass

    remote_dir = _default_server_directory()
    if cfg.hostname:
        if not cfg.ssh_hostname:
            raise LauncherError(
                "ssh_hostname must be provided when hostname is configured."
            )
        executor = SSHExecutor(cfg.ssh_hostname, observer)
    else:
        executor = LocalExecutor(observer)
    observer.on_phase(Phase.CONNECT, executor.host)
    if cfg.conda_env:
        executor.ensure_conda_setup()
    remote = RemoteState(executor, cfg, remote_dir.as_posix())
    if dry_run and cfg.hostname is not None:
        raise LauncherError("--dry-run requires hostname to be unset in the config.")
    remote_result, done_result = handle_remote(cfg, remote, dry_run=dry_run)

    local_payload = create_local_file(cfg, local_state, remote_result, observer)
    observer.on_phase(Phase.DONE, done_result)
    return local_payload


def run(
    config: Dict[str, Any],
    *,
    connection_dir: pathlib.Path | str | None = None,
) -> Dict[str, Any]:
    """Programmatic API returning the local connection payload.

    Connection dir defaults to ~/.remote-http-launcher/client
    """
    config_data, logging_config = _extract_logging_config(config)
    cfg = Configuration.from_mapping(config_data)
    bundle = _build_observer_bundle(cfg.key, cfg, logging_config)
    override_dir: Optional[pathlib.Path]
    if connection_dir is None:
        override_dir = None
    else:
        override_dir = pathlib.Path(connection_dir).expanduser()
    try:
        return _execute(cfg, override_dir, observer=bundle.observer)
    except LauncherError as exc:
        bundle.observer.on_error(str(exc))
        raise
    finally:
        bundle.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remote HTTP launcher.")
    parser.add_argument(
        "config", type=pathlib.Path, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--connection-dir",
        type=pathlib.Path,
        help="Override the connection directory. Defaults to ~/.remote-http-launcher/client",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluated remote command without launching it.",
    )
    parser.add_argument(
        "--log-level",
        choices=["minimal", "basic", "debug"],
        default=None,
        help="Override log level from config.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional file for basic logs.",
    )
    parser.add_argument(
        "--debug-log-file",
        type=str,
        default=None,
        help="Optional file for debug logs.",
    )
    args = parser.parse_args(argv)

    raw_config = _load_yaml_mapping(args.config)
    config_data, logging_config = _extract_logging_config(
        raw_config,
        log_level=args.log_level,
        log_file=args.log_file,
        debug_log_file=args.debug_log_file,
    )
    cfg = Configuration.from_mapping(config_data)
    bundle = _build_observer_bundle(cfg.key, cfg, logging_config)
    try:
        _execute(cfg, args.connection_dir, observer=bundle.observer, dry_run=args.dry_run)
        return 0
    except LauncherError as exc:
        bundle.observer.on_error(str(exc))
        return 1
    finally:
        bundle.close()


if __name__ == "__main__":
    sys.exit(main())
