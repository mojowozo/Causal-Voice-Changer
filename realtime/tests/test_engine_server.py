import asyncio
import os
import tempfile
import threading
import time
import unittest

from engine import protocol
from engine.cli import EngineClient
from engine.server import CommandError, EngineServer


class FakeEngine:
    """Stands in for RealtimeEngine so the protocol tests need no Torch."""

    def __init__(self, emit, ready_event=None):
        if ready_event is not None:
            ready_event.wait(5)
        self.emit = emit
        self.settings = {"pitch": 0.0}
        self.ticks = 0
        self.shut_down = False

    def state(self):
        return {"running": False, "function": "vc", "settings": dict(self.settings)}

    def update_settings(self, changes):
        if "pitch" not in changes:
            raise CommandError("bad_setting", "only pitch is supported", key="x")
        self.settings["pitch"] = changes["pitch"]
        self.emit("state", self.state())

    def start(self, function):
        print(f"starting {function}")
        raise RuntimeError("no audio devices in tests")

    def tick(self):
        self.ticks += 1
        if self.ticks == 1:
            self.emit("meters", {"input": 0.5})

    def shutdown(self):
        self.shut_down = True


class FakeUpdater:
    """Stands in for causalvc.update.Updater: records calls, never touches the network."""

    def __init__(self, emit):
        self.emit, self.calls, self.state = emit, [], {"state": "idle"}

    def launch(self):
        self.calls.append(("launch",))
        self.state = {"state": "available", "version": "v9"}

    def snapshot(self):
        return dict(self.state)

    def check(self, manual):
        self.calls.append(("check", manual))

    def download(self):
        from causalvc.update import UpdateError

        raise UpdateError("no_update", "nothing to download")

    def install(self, pid):
        self.calls.append(("install", pid))

    def skip(self, version):
        self.calls.append(("skip", version))
        self.state = {"state": "idle"}

    def later(self):
        self.calls.append(("later",))

    def set_check_at_launch(self, on):
        self.calls.append(("set_check_at_launch", on))


class HangingShutdownEngine(FakeEngine):
    """Shutdown never returns, like an audio device that will not close."""

    def shutdown(self):
        self.shut_down = True
        threading.Event().wait(2)


class EngineAddressTest(unittest.TestCase):
    def test_each_copy_of_the_app_has_its_own_address(self):
        # The windows compute the same key: ui/src/engine.rs and winui/Engine/EngineConnection.cs.
        self.assertEqual(protocol.app_key("/opt/causalvc/realtime"), "8f89d62bd5c3af82")
        self.assertNotEqual(protocol.default_address("/a/realtime"), protocol.default_address("/b/realtime"))
        self.assertIn(protocol.app_key(protocol.APP_DIR), protocol.default_address())


class EngineServerTest(unittest.TestCase):
    """Over a Unix socket (Linux); the subclass below repeats every test over TCP (Windows)."""

    ADDRESS_PREFIX = ""
    ADDRESS_NAME = "e.sock"

    def start_server(self, engine_factory, exit_when_idle=None, updater_factory=None):
        self.directory = tempfile.mkdtemp(prefix="causalvc-")
        self.path = os.path.join(self.directory, self.ADDRESS_NAME)
        self.address = self.ADDRESS_PREFIX + self.path
        self.server = EngineServer(
            self.address, engine_factory=engine_factory, exit_when_idle=exit_when_idle,
            updater_factory=updater_factory,
        )
        self.thread = threading.Thread(target=asyncio.run, args=(self.server.serve(),))
        self.thread.start()
        for _ in range(200):
            if os.path.exists(self.path):
                return
            time.sleep(0.01)
        self.fail("server did not bind its socket")

    def stop_server(self):
        if self.thread.is_alive():
            client = EngineClient(self.address, timeout=5)
            client.call("shutdown")
            client.close()
        self.thread.join(5)
        self.assertFalse(self.thread.is_alive())
        self.assertFalse(os.path.exists(self.path))
        os.rmdir(self.directory)

    def connect_ready(self):
        client = EngineClient(self.address, timeout=5)
        self.addCleanup(client.close)
        if not client.hello["data"]["ready"]:
            while client.read_message().get("event") != "ready":
                pass
        return client

    def test_update_commands_work_before_the_engine_is_ready_and_install_stops_it(self):
        release = threading.Event()
        updaters = []

        def make(emit):
            updaters.append(FakeUpdater(emit))
            return updaters[-1]

        self.start_server(lambda emit: FakeEngine(emit, release), updater_factory=make)
        client = EngineClient(self.address, timeout=5)
        self.assertEqual(client.hello["data"]["update"]["state"], "available")  # from the check at launch
        self.assertEqual(client.call("skip_update", {"version": "v9"})["result"]["state"], "idle")
        self.assertEqual(updaters[0].calls[-1], ("skip", "v9"))
        self.assertEqual(client.call("set_update_check", {"on": "yes"})["error"]["code"], "bad_request")
        client.call("set_update_check", {"on": False})
        self.assertEqual(updaters[0].calls[-1], ("set_check_at_launch", False))
        self.assertEqual(client.call("download_update")["error"]["code"], "no_update")
        self.assertTrue(client.call("install_update", {"pid": 4321})["ok"])
        self.assertEqual(updaters[0].calls[-1], ("install", 4321))
        client.close()
        release.set()
        self.thread.join(5)  # the engine stops by itself after starting the installer
        self.assertFalse(self.thread.is_alive())
        os.rmdir(self.directory)

    def test_socket_accepts_commands_before_engine_is_ready(self):
        release = threading.Event()
        self.start_server(lambda emit: FakeEngine(emit, release))
        client = EngineClient(self.address, timeout=5)

        self.assertFalse(client.hello["data"]["ready"])
        self.assertTrue(client.call("ping")["result"]["protocol"] >= 1)
        self.assertEqual(client.call("get_state")["error"]["code"], "not_ready")
        release.set()
        while client.read_message().get("event") != "ready":
            pass
        self.assertTrue(client.call("get_state")["ok"])
        client.close()
        self.stop_server()

    def test_commands_results_errors_and_events(self):
        self.start_server(FakeEngine)
        client = self.connect_ready()
        events = []

        response = client.call(
            "update_settings", {"settings": {"pitch": 3}}, on_event=events.append
        )
        self.assertTrue(response["ok"])
        while not any(event["event"] == "state" for event in events):
            events.append(client.read_message())
        self.assertEqual(
            client.call("get_state")["result"]["settings"]["pitch"], 3
        )

        error = client.call("update_settings", {"settings": {"x": 1}})["error"]
        self.assertEqual((error["code"], error["details"]), ("bad_setting", {"key": "x"}))
        self.assertEqual(client.call("update_settings")["error"]["code"], "bad_request")
        self.assertEqual(client.call("nope")["error"]["code"], "unknown_command")
        self.assertEqual(client.call("start")["error"]["code"], "internal_error")

        log = client.call("get_log")["result"]["text"]
        self.assertIsInstance(log, str)
        self.stop_server()
        self.assertTrue(self.server.engine.shut_down)

    def test_malformed_lines_get_an_error_response(self):
        self.start_server(FakeEngine)
        client = self.connect_ready()

        client.sock.sendall(b"not json\n")
        message = client.read_message()
        while "ok" not in message:
            message = client.read_message()

        self.assertEqual(message["error"]["code"], "bad_request")
        self.stop_server()

    def test_init_failure_is_reported(self):
        def broken(emit):
            raise RuntimeError("CUDA exploded")

        self.start_server(broken)
        client = EngineClient(self.address, timeout=5)
        message = client.hello
        while message.get("event") not in ("fatal",) and not message["data"].get("init_error"):
            message = client.read_message()
        response = client.call("get_state")
        self.assertEqual(response["error"]["code"], "init_failed")
        self.assertIn("CUDA exploded", response["error"]["message"])
        client.close()
        self.stop_server()

    def test_a_failure_to_load_amds_runtime_says_what_to_do(self):
        def broken(emit):
            # What Windows raises when ROCm's runtime DLL cannot start (an old AMD driver).
            raise OSError("[WinError 1114] A dynamic link library (DLL) initialization routine failed: rocm_sdk")

        self.start_server(broken)
        client = EngineClient(self.address, timeout=5)
        message = client.hello
        while message.get("event") not in ("fatal",) and not message["data"].get("init_error"):
            message = client.read_message()
        text = client.call("get_state")["error"]["message"]
        self.assertTrue(text.startswith("AMD's GPU runtime (ROCm) could not start"))
        self.assertIn("Adrenalin Edition 26.2.2", text)
        self.assertIn("WinError 1114", text)  # the details stay below
        client.close()
        self.stop_server()

    def test_exits_when_idle(self):
        self.start_server(FakeEngine, exit_when_idle=0.2)
        client = self.connect_ready()
        client.close()

        self.thread.join(5)
        self.assertFalse(self.thread.is_alive())
        self.assertFalse(os.path.exists(self.path))
        os.rmdir(self.directory)

    def test_exits_even_if_the_engine_shutdown_hangs(self):
        from unittest import mock

        from engine import server as server_module

        with mock.patch.object(server_module, "SHUTDOWN_TIMEOUT_SECONDS", 0.3):
            self.start_server(HangingShutdownEngine, exit_when_idle=0.2)
            client = self.connect_ready()
            client.close()
            started = time.monotonic()
            self.thread.join(5)
        self.assertFalse(self.thread.is_alive())
        self.assertLess(time.monotonic() - started, 3)
        self.assertTrue(self.server.engine.shut_down)
        self.assertFalse(os.path.exists(self.path))
        os.rmdir(self.directory)


class TcpEngineServerTest(EngineServerTest):
    """The Windows transport: loopback TCP, with the port and token in an endpoint file."""

    ADDRESS_PREFIX = "tcp:"
    ADDRESS_NAME = "endpoint.json"

    def test_endpoint_file_is_private_and_clients_need_the_token(self):
        import json
        import socket
        import stat

        self.start_server(FakeEngine)
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        with open(self.path, encoding="utf-8") as handle:
            port = json.load(handle)["port"]
        for first_line in (b'{"auth": "wrong"}\n', b'{"id": 1, "cmd": "get_state"}\n'):
            with socket.create_connection(("127.0.0.1", port), timeout=5) as intruder:
                intruder.sendall(first_line)
                self.assertEqual(intruder.recv(4096), b"")  # closed without a hello
        client = self.connect_ready()
        self.assertTrue(client.call("get_state")["ok"])
        self.stop_server()

    def test_a_second_engine_refuses_to_start(self):
        self.start_server(FakeEngine)
        second = EngineServer(self.address, engine_factory=FakeEngine)
        with self.assertRaises(SystemExit):
            asyncio.run(second.serve())
        self.assertTrue(os.path.exists(self.path))  # the first engine's file is left alone
        self.stop_server()


if __name__ == "__main__":
    unittest.main()
