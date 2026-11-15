# Training window

The window for building voices. `run-training.sh` and `run-training.bat` build it on first run and open it.

| Folder | What it is |
|---|---|
| `ui/` | The COSMIC front end (Rust), for Linux |
| `winui/` | The WinUI 3 front end (C#), for Windows: the same pages, on desktop acrylic |
| `tests/` | Tests for the trainer service: `PYTHONPATH=.. ../.venv/bin/python -m unittest discover -s tests` |

Both windows are front ends only. Each one starts `python -m causalvc.trainer` as a child process
and talks to it in JSON lines over stdin and stdout. The trainer runs the builds one at a time,
keeps the queue, and saves the form to `runs/gui.json`. The protocol is described at the top of
[`causalvc/trainer.py`](../causalvc/trainer.py).

Closing the window ends the trainer's stdin. A running build then stops the same way as with
Stop: it saves a checkpoint, and you can continue it later.

## Building by hand

Linux (the compiled libraries are shared with the voice changer, in `realtime/ui/target`):

```bash
(cd training/ui && cargo build --release)
CAUSALVC_DIR="$PWD" realtime/ui/target/release/causalvc-training
```

Windows (the .NET 10 SDK; run `run-training.bat` with `CAUSALVC_REBUILD=1` to rebuild after updating):

```bat
dotnet publish training\winui\CausalVC.Training.csproj -c Release -o training\winui\out
```

If neither can be built, `python -m causalvc.gui` is the small Tk window with the same features.
