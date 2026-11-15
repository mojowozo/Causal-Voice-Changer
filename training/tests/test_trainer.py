"""The training windows' backend (causalvc.trainer): checking the form, the queue, stopping, progress.

    PYTHONPATH=. .venv/bin/python -m unittest discover -s training/tests
"""

import json
import queue
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from causalvc import trainer

ROOT = Path(__file__).resolve().parents[2]


class FakeJob:
    """Stands in for the build process: the test feeds its output."""

    def __init__(self):
        self.lines = queue.Queue()
        self.started = []
        self.stopping = False
        self.alive = False
        self.proc = mock.Mock(returncode=None)
        self.stops = 0

    def running(self):
        return self.alive

    def start(self, **config):
        self.started.append(config)
        self.alive, self.stopping = True, False

    def stop(self):
        self.stops += 1
        self.stopping = True

    def finish(self, code=0):
        self.alive = False
        self.proc.returncode = code
        self.lines.put(None)


class ServiceTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.audio = self.root / "speech"
        self.audio.mkdir()
        patcher = mock.patch.object(trainer, "SETTINGS", self.root / "gui.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.events = []
        self.job = FakeJob()
        self.service = trainer.Service(lambda event, data: self.events.append((event, data)), bases=["Mine"], job=self.job)

    def config(self, **changes):
        return {"name": "Kira", "audio": str(self.audio), "batch": 8, "steps": 100, "save_every": 50, "base": "default",
                "device": "cuda:0", "restart": False, "fixes": True, "app_models": "", **changes}

    def state(self):
        return [data for event, data in self.events if event == "state"][-1]

    def errors(self):
        return [data for event, data in self.events if event == "error"]

    def test_hello_offers_the_saved_form_and_the_bases(self):
        trainer.save_form(self.config(name="Saved", steps=777))
        self.service.hello()
        event, data = self.events[0]
        self.assertEqual(event, "hello")
        self.assertEqual(data["form"]["name"], "Saved")
        self.assertEqual(data["form"]["steps"], 777)
        self.assertEqual(data["bases"], ["default", "Mine"])
        self.assertEqual(self.state()["phase"], "idle")

    def test_a_damaged_settings_file_falls_back_to_defaults(self):
        (self.root / "gui.json").write_text('{"steps": "lots", "fixes": 1, "name": 5')
        self.assertEqual(trainer.load_form(), trainer.DEFAULT_FORM)
        (self.root / "gui.json").write_text(json.dumps({"steps": True, "fixes": 1, "name": "Ok"}))
        form = trainer.load_form()
        self.assertEqual((form["steps"], form["fixes"], form["name"]), (10000, True, "Ok"))

    def test_bad_forms_are_refused_naming_the_field(self):
        for changes, code, field in (
            ({"name": "two words"}, "invalid_name", "name"),
            ({"name": ""}, "invalid_name", "name"),
            ({"audio": str(self.root / "nowhere")}, "missing_audio", "audio"),
            ({"steps": 0}, "invalid_numbers", "steps"),
            ({"batch": "eight"}, "invalid_numbers", "batch"),
            ({"save_every": True}, "invalid_numbers", "save_every"),
        ):
            self.service.command("start", self.config(**changes))
            self.assertEqual((self.errors()[-1]["code"], self.errors()[-1]["field"]), (code, field), changes)
        self.assertEqual(self.job.started, [])

    def test_numbers_typed_as_text_and_unknown_bases_are_accepted(self):
        self.service.command("start", self.config(batch=" 4 ", base="Gone"))
        self.assertEqual(self.job.started[0]["batch"], 4)
        self.assertEqual(self.job.started[0]["base"], "default")
        self.assertEqual(json.loads((self.root / "gui.json").read_text())["batch"], 4)

    def test_a_pasted_path_keeps_working_with_quotes_and_spaces(self):
        self.service.command("start", self.config(audio=f' "{self.audio}" \n'))
        self.assertEqual(self.errors(), [])
        self.assertEqual(self.job.started[0]["audio"], str(self.audio))

    def test_progress_moves_the_phase_and_noise_stays_out_of_the_log(self):
        self.service.command("start", self.config())
        self.assertEqual(self.state()["phase"], "starting")
        for line in ("UserWarning: something", "loading base", "step 10/100 (10%)  elapsed 0:05  remaining 0:45  2.00 it/s"):
            self.job.lines.put(line)
        self.service.pump()
        self.assertEqual([data["text"] for event, data in self.events if event == "log"], ["loading base"])
        self.assertEqual(self.state()["phase"], "training")
        self.assertEqual(self.state()["progress"], {"step": 10, "total": 100, "percent": 10, "elapsed": "0:05",
                                                    "remaining": "0:45", "rate": 2.0})
        self.job.finish(0)
        self.service.pump()
        self.assertEqual((self.state()["phase"], self.state()["exit_code"]), ("finished", 0))

    def test_queued_builds_run_in_order_and_can_be_removed(self):
        self.service.command("queue", self.config(name="A"))  # idle: starts at once
        self.service.command("queue", self.config(name="B"))
        self.service.command("queue", self.config(name="C"))
        self.assertEqual([c["name"] for c in self.state()["queue"]], ["B", "C"])
        self.service.command("start", self.config(name="D"))
        self.assertEqual(self.errors()[-1]["code"], "busy")
        self.service.command("remove", {"index": 0})
        self.service.command("remove", {"index": 5})
        self.assertEqual(self.errors()[-1]["code"], "bad_request")
        self.job.finish(0)
        self.service.pump()
        self.assertEqual([c["name"] for c in self.job.started], ["A", "C"])
        self.assertEqual(self.state()["current"]["name"], "C")
        self.assertEqual(self.state()["queue"], [])

    def test_stopping_saves_first_and_keeps_the_queue_waiting(self):
        self.service.command("queue", self.config(name="A"))
        self.service.command("queue", self.config(name="B"))
        self.service.command("stop", {})
        self.assertEqual(self.state()["phase"], "stopping")
        self.service.command("stop", {})
        self.assertEqual(self.job.stops, 2)  # the second ends it at once (Job.stop)
        self.job.finish(-9)
        self.service.pump()
        self.assertEqual(self.state()["phase"], "stopped")
        self.assertEqual([c["name"] for c in self.job.started], ["A"])
        self.assertEqual([c["name"] for c in self.state()["queue"]], ["B"])

    def test_a_failed_build_stops_the_queue(self):
        self.service.command("queue", self.config(name="A"))
        self.service.command("queue", self.config(name="B"))
        self.job.finish(1)
        self.service.pump()
        self.assertEqual((self.state()["phase"], self.state()["exit_code"]), ("failed", 1))
        self.assertEqual(len(self.job.started), 1)


class CloseTest(unittest.TestCase):
    def test_closing_the_window_saves_a_checkpoint_instead_of_killing(self):
        job = FakeJob()
        job.alive = True
        job.proc.wait = mock.Mock()
        service = trainer.Service(lambda *_: None, bases=[], job=job)
        service.close()
        self.assertEqual(job.stops, 1)
        job.proc.wait.assert_called_once()
        job.proc.kill.assert_not_called()


class ProcessTest(unittest.TestCase):
    def test_the_service_speaks_json_lines_and_exits_when_its_window_closes(self):
        folder = tempfile.mkdtemp()
        code = (
            "import sys; from pathlib import Path; from unittest import mock; from causalvc import trainer\n"
            f"trainer.SETTINGS = Path({folder!r}) / 'gui.json'\n"
            "trainer.torch_gpus = lambda: [{'id': 'cuda:0', 'label': 'Test GPU'}]\n"
            "sys.stdout.reconfigure(encoding='utf-8'); trainer.main()\n"
        )
        commands = json.dumps({"cmd": "start", "args": {"name": "bad name"}}) + "\nnot json\n"
        result = subprocess.run([sys.executable, "-c", code], input=commands, capture_output=True, text=True,
                                timeout=60, cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": ""})
        events = [json.loads(line) for line in result.stdout.splitlines()]
        names = [event["event"] for event in events]
        self.assertEqual(names[:2], ["hello", "state"])
        self.assertIn({"event": "gpus", "data": {"gpus": [{"id": "cuda:0", "label": "Test GPU"}]}}, events)
        self.assertIn("invalid_name", [event["data"].get("code") for event in events if event["event"] == "error"])
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
