import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import remote_http_launcher as rhl


def make_cfg(**overrides):
    data = {
        "workdir": "/tmp/rhl-test",
        "key": "svc",
        "command": "echo hello",
        "network_interface": "127.0.0.1",
    }
    data.update(overrides)
    return rhl.Configuration.from_mapping(data)


class RecordingObserver:
    def __init__(self):
        self.phases = []
        self.details = []
        self.errors = []
        self.commands = []
        self.scripts = []

    def on_phase(self, phase, result):
        self.phases.append((phase, result))

    def on_detail(self, message):
        self.details.append(message)

    def on_command(self, host, command, full_command):
        self.commands.append((host, command, full_command))

    def on_command_done(self, host, command):
        self.commands.append((host, f"done:{command}", None))

    def on_script(self, host, script):
        self.scripts.append((host, script))

    def on_error(self, message):
        self.errors.append(message)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class FakeRemote:
    def __init__(self, observer, exists_values, read_values):
        self.observer = observer
        self.executor = mock.Mock()
        self.exists_values = list(exists_values)
        self.read_values = list(read_values)
        self.killed = []
        self.removed = 0
        self.handshake_calls = 0
        self.verify_calls = 0
        self.launch_called = False
        self.wrote_dry_run = False
        self.logged = None

    def exists(self):
        if self.exists_values:
            return self.exists_values.pop(0)
        return False

    def read(self):
        if not self.read_values:
            raise AssertionError("read() called without prepared payload")
        return self.read_values.pop(0)

    def verify_port_in_use(self, host, port):
        self.verify_calls += 1
        return None

    def handshake(self, host, port, handshake):
        self.handshake_calls += 1
        return None

    def kill_process(self, pid):
        self.killed.append(pid)

    def remove(self):
        self.removed += 1

    def stat_and_read(self):
        raise AssertionError("stat_and_read should be mocked in specific tests")

    def process_exists(self, pid):
        return False

    def read_log(self):
        return self.logged

    def launch_process(self, conda_base):
        self.launch_called = True

    def evaluate_command(self):
        return "echo hello"

    def write_dry_run_metadata(self, evaluated_command):
        self.wrote_dry_run = True


class LoggingObserverTests(unittest.TestCase):
    def test_build_expected_phases_with_and_without_optional_steps(self):
        basic_cfg = make_cfg()
        self.assertEqual(
            rhl._build_expected_phases(basic_cfg),
            [
                rhl.Phase.LOCAL_CHECK,
                rhl.Phase.CONNECT,
                rhl.Phase.REMOTE_CHECK,
                rhl.Phase.REMOTE_VALIDATE,
                rhl.Phase.STALE_CLEANUP,
                rhl.Phase.LAUNCH,
                rhl.Phase.WAIT_FOR_START,
                rhl.Phase.LOCAL_VALIDATE,
                rhl.Phase.DONE,
            ],
        )

        rich_cfg = make_cfg(conda="env", tunnel=True, hostname="host", ssh_hostname="ssh")
        self.assertEqual(
            rhl._build_expected_phases(rich_cfg),
            [
                rhl.Phase.LOCAL_CHECK,
                rhl.Phase.CONNECT,
                rhl.Phase.REMOTE_CHECK,
                rhl.Phase.REMOTE_VALIDATE,
                rhl.Phase.STALE_CLEANUP,
                rhl.Phase.CONDA_SETUP,
                rhl.Phase.LAUNCH,
                rhl.Phase.WAIT_FOR_START,
                rhl.Phase.TUNNEL_SETUP,
                rhl.Phase.LOCAL_VALIDATE,
                rhl.Phase.DONE,
            ],
        )

    def test_basic_observer_formats_phase_and_error_lines(self):
        stream = io.StringIO()
        clock = FakeClock()
        observer = rhl.BasicObserver(stream, clock=clock)
        observer.on_phase(rhl.Phase.LOCAL_CHECK, "missing")
        clock.now = 1.2
        observer.on_error("boom")
        self.assertEqual(
            stream.getvalue().splitlines(),
            ["   0.0s local_check: missing", "   1.2s error: boom"],
        )

    def test_minimal_observer_fast_path_is_silent(self):
        stream = io.StringIO()
        clock = FakeClock()
        observer = rhl.MinimalObserver(
            "svc",
            [rhl.Phase.LOCAL_CHECK, rhl.Phase.DONE],
            stream=stream,
            clock=clock,
        )
        observer.on_phase(rhl.Phase.LOCAL_CHECK, "reused")
        observer.on_phase(rhl.Phase.DONE, "reused-local")
        self.assertEqual(stream.getvalue(), "")

    def test_minimal_observer_renders_after_threshold_and_skips_forward(self):
        stream = io.StringIO()
        clock = FakeClock()
        observer = rhl.MinimalObserver(
            "svc",
            [
                rhl.Phase.LOCAL_CHECK,
                rhl.Phase.CONNECT,
                rhl.Phase.REMOTE_CHECK,
                rhl.Phase.REMOTE_VALIDATE,
                rhl.Phase.STALE_CLEANUP,
                rhl.Phase.LAUNCH,
                rhl.Phase.WAIT_FOR_START,
                rhl.Phase.DONE,
            ],
            stream=stream,
            clock=clock,
        )
        observer.on_phase(rhl.Phase.LOCAL_CHECK, "missing")
        clock.now = 2.0
        observer.on_phase(rhl.Phase.WAIT_FOR_START, "waiting")
        line = stream.getvalue().strip().splitlines()[-1]
        self.assertIn("wait_for_start", line)
        self.assertGreater(line.count("#"), 10)

    def test_minimal_observer_tty_finishes_with_summary(self):
        stream = FakeTTY()
        clock = FakeClock()
        observer = rhl.MinimalObserver(
            "svc",
            [rhl.Phase.LOCAL_CHECK, rhl.Phase.DONE],
            stream=stream,
            clock=clock,
        )
        clock.now = 2.0
        observer.on_phase(rhl.Phase.LOCAL_CHECK, "missing")
        observer.on_phase(rhl.Phase.DONE, "launched")
        self.assertIn("svc: ready (2.0s)", stream.getvalue())

    def test_minimal_observer_does_not_regress_on_retry_phase(self):
        stream = io.StringIO()
        clock = FakeClock()
        observer = rhl.MinimalObserver(
            "svc",
            [
                rhl.Phase.LOCAL_CHECK,
                rhl.Phase.CONNECT,
                rhl.Phase.REMOTE_CHECK,
                rhl.Phase.REMOTE_VALIDATE,
                rhl.Phase.STALE_CLEANUP,
                rhl.Phase.CONDA_SETUP,
                rhl.Phase.LAUNCH,
                rhl.Phase.WAIT_FOR_START,
                rhl.Phase.DONE,
            ],
            stream=stream,
            clock=clock,
        )
        clock.now = 2.0
        observer.on_phase(rhl.Phase.STALE_CLEANUP, "kill+remove")
        first_line = stream.getvalue().strip().splitlines()[-1]
        observer.on_phase(rhl.Phase.REMOTE_CHECK, "missing")
        lines = stream.getvalue().strip().splitlines()
        self.assertEqual(lines[-1], first_line)
        self.assertIn("stale_cleanup", lines[-1])

    def test_debug_observer_emits_full_command_and_script(self):
        stream = io.StringIO()
        observer = rhl.DebugObserver(stream)
        observer.on_command("host", "short", "full command")
        observer.on_script("host", "print('hi')")
        observer.on_command_done("host", "short")
        output = stream.getvalue()
        self.assertIn("command[host]: short", output)
        self.assertIn("full-command[host]: full command", output)
        self.assertIn("script[host]:", output)
        self.assertIn("print('hi')", output)
        self.assertIn("command-done[host]: short", output)

    def test_open_log_stream_truncates_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "launcher.log"
            path.write_text("old contents\n", encoding="utf-8")
            with rhl._open_log_stream(str(path)) as handle:
                handle.write("new contents\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "new contents\n")


class LauncherFlowTests(unittest.TestCase):
    def test_run_strips_logging_keys_before_configuration(self):
        bundle = rhl.ObserverBundle(observer=rhl.NullObserver(), close=lambda: None)
        with mock.patch.object(rhl, "_build_observer_bundle", return_value=bundle):
            with mock.patch.object(rhl, "_execute", return_value={"hostname": "x", "port": 1}) as execute:
                rhl.run(
                    {
                        "workdir": "/tmp/rhl-test",
                        "key": "svc",
                        "command": "echo hello",
                        "log_level": "minimal",
                        "log_file": "~/basic.log",
                        "debug_log_file": "~/debug.log",
                    }
                )
        cfg = execute.call_args.kwargs["observer"]
        self.assertIs(cfg, bundle.observer)
        built_cfg = execute.call_args.args[0]
        self.assertNotIn("log_level", built_cfg.raw)
        self.assertNotIn("log_file", built_cfg.raw)
        self.assertNotIn("debug_log_file", built_cfg.raw)

    def test_handle_remote_reuses_existing_running_service(self):
        observer = RecordingObserver()
        cfg = make_cfg()
        data = {
            "status": "running",
            "pid": 123,
            "port": 8123,
            "uid": 1000,
            "command": "echo hello",
        }
        remote = FakeRemote(observer, [True], [data])
        result, kind = rhl.handle_remote(cfg, remote)
        self.assertEqual(result, data)
        self.assertEqual(kind, "reused-remote")
        self.assertEqual(
            observer.phases,
            [
                (rhl.Phase.REMOTE_CHECK, "found:running"),
                (rhl.Phase.REMOTE_VALIDATE, "ok"),
            ],
        )

    def test_handle_remote_failed_handshake_restarts_outer_loop(self):
        observer = RecordingObserver()
        cfg = make_cfg(handshake="health")
        data = {
            "status": "running",
            "pid": 456,
            "port": 9000,
            "uid": 1000,
            "command": "echo hello",
        }
        remote = FakeRemote(observer, [True, False], [data])

        def fail_handshake(host, port, handshake):
            remote.handshake_calls += 1
            raise RuntimeError("still unhealthy")

        remote.handshake = fail_handshake
        with mock.patch.object(rhl.time, "sleep", return_value=None):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    rhl.handle_remote(cfg, remote, dry_run=True)
        self.assertEqual(exc.exception.code, 0)
        self.assertEqual(remote.handshake_calls, 15)
        self.assertEqual(remote.killed, [456])
        self.assertEqual(remote.removed, 1)
        self.assertTrue(remote.wrote_dry_run)
        self.assertIn((rhl.Phase.REMOTE_VALIDATE, "handshake-failed"), observer.phases)
        self.assertIn((rhl.Phase.STALE_CLEANUP, "kill+remove"), observer.phases)

    def test_handle_remote_launches_and_waits_without_reentering_remote_check(self):
        observer = RecordingObserver()
        cfg = make_cfg()
        data = {
            "status": "running",
            "pid": 789,
            "port": 9100,
            "uid": 1000,
            "command": "echo hello",
        }
        remote = FakeRemote(observer, [False, True], [data])
        result, kind = rhl.handle_remote(cfg, remote)
        self.assertEqual(result, data)
        self.assertEqual(kind, "launched")
        self.assertTrue(remote.launch_called)
        self.assertEqual(
            observer.phases,
            [
                (rhl.Phase.REMOTE_CHECK, "missing"),
                (rhl.Phase.LAUNCH, "started"),
                (rhl.Phase.WAIT_FOR_START, "waiting"),
            ],
        )

    def test_handle_remote_stale_state_cleans_and_launches_same_invocation(self):
        observer = RecordingObserver()
        cfg = make_cfg()
        stale = {"status": "stale", "pid": 222}
        running = {
            "status": "running",
            "pid": 789,
            "port": 9100,
            "uid": 1000,
            "command": "echo hello",
        }
        remote = FakeRemote(observer, [True, False, True], [stale, running])
        result, kind = rhl.handle_remote(cfg, remote)
        self.assertEqual(result, running)
        self.assertEqual(kind, "launched")
        self.assertTrue(remote.launch_called)
        self.assertEqual(remote.removed, 1)
        self.assertEqual(
            observer.phases,
            [
                (rhl.Phase.REMOTE_CHECK, "found:stale"),
                (rhl.Phase.STALE_CLEANUP, "remove"),
                (rhl.Phase.REMOTE_CHECK, "missing"),
                (rhl.Phase.LAUNCH, "started"),
                (rhl.Phase.WAIT_FOR_START, "waiting"),
            ],
        )

    def test_handle_remote_starting_state_cleans_up_and_retries(self):
        observer = RecordingObserver()
        cfg = make_cfg()
        data = {"status": "starting", "pid": 321}
        remote = FakeRemote(observer, [True, False], [data])
        remote.logged = "boom"
        with mock.patch.object(rhl, "monitor_remote_start", return_value=None):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    rhl.handle_remote(cfg, remote, dry_run=True)
        self.assertEqual(exc.exception.code, 0)
        self.assertEqual(remote.removed, 1)
        self.assertIn((rhl.Phase.WAIT_FOR_START, "waiting"), observer.phases)
        self.assertIn((rhl.Phase.STALE_CLEANUP, "remove+log"), observer.phases)

    def test_execute_local_reuse_short_circuits_before_connect(self):
        observer = RecordingObserver()
        cfg = make_cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "svc.json"
            payload = {"hostname": "localhost", "port": 8123}
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = rhl._execute(cfg, pathlib.Path(tmpdir), observer=observer)
        self.assertEqual(result, payload)
        self.assertEqual(
            observer.phases,
            [
                (rhl.Phase.LOCAL_CHECK, "reused"),
                (rhl.Phase.DONE, "reused-local"),
            ],
        )

    def test_handle_local_stale_status_deletes_and_relaunches(self):
        observer = RecordingObserver()
        cfg = make_cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_state = rhl.LocalState.build(cfg, pathlib.Path(tmpdir))
            payload = {"hostname": "localhost", "port": 8123, "status": "stale"}
            local_state.write(payload)
            result = rhl.handle_local(cfg, local_state, observer)
            self.assertIsNone(result)
            self.assertFalse(local_state.json_path.exists())
        self.assertEqual(observer.phases, [(rhl.Phase.LOCAL_CHECK, "stale")])

    def test_create_local_file_tunnel_emits_phase_and_writes_payload(self):
        observer = RecordingObserver()
        cfg = make_cfg(hostname="frontend", ssh_hostname="ssh-front", tunnel=True)
        remote_data = {
            "pid": 111,
            "port": 7000,
            "network_interface": "10.0.0.1",
            "dashboard_port": 7001,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            local_state = rhl.LocalState.build(cfg, pathlib.Path(tmpdir))
            with mock.patch.object(rhl, "allocate_local_port", side_effect=[17000, 17001]):
                with mock.patch.object(rhl, "establish_tunnel", return_value=None):
                    with mock.patch.object(rhl, "perform_local_handshake", return_value=None):
                        payload = rhl.create_local_file(
                            cfg, local_state, remote_data, observer
                        )
        self.assertEqual(payload["hostname"], "localhost")
        self.assertEqual(payload["port"], 17000)
        self.assertEqual(payload["dashboard_port"], 17001)
        self.assertIn(
            (rhl.Phase.TUNNEL_SETUP, "localhost:17000 -> frontend:7000"),
            observer.phases,
        )
        self.assertIn((rhl.Phase.LOCAL_VALIDATE, "ok"), observer.phases)


if __name__ == "__main__":
    unittest.main()
