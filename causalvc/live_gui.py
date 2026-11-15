"""A small window for using a voice live: microphone in, converted voice out.

    python -m causalvc.live_gui

Works wherever Python's Tk and PortAudio do (Windows, Linux, macOS). On Linux
the fuller app is in realtime/. Pitch, gains and the gate apply while running;
changing the voice, devices, block or GPU takes effect on the next start.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = ROOT / "runs" / "live_gui.json"
SUFFIXES = (".causalvc.json", ".nextvc.json")


def voices() -> dict[str, str]:
    """Voice name -> description file, from the realtime app's folder and from finished builds."""
    found: dict[str, str] = {}
    for folder, pattern in ((ROOT / "realtime" / "models", "*/*.json"), (ROOT / "runs" / "models", "*/voices/*/*.json")):
        if folder.is_dir():
            for path in sorted(folder.glob(pattern)):
                if path.name.endswith(SUFFIXES):
                    name = path.name
                    for suffix in SUFFIXES:
                        name = name.removesuffix(suffix)
                    found.setdefault(name, str(path))
    return found


def devices(kind: str) -> list[str]:
    """Labels of the input or output devices, the system default first."""
    import sounddevice as sd

    apis = sd.query_hostapis()
    labels = ["System default"]
    for index, d in enumerate(sd.query_devices()):
        if d[f"max_{kind}_channels"] > 0:
            labels.append(f"{index}: {d['name']} [{apis[d['hostapi']]['name']}]")
    return labels


def device_index(label: str):
    return None if not label or label == "System default" else int(label.split(":", 1)[0])


def gpu_choices() -> list[str]:
    import torch

    if not torch.cuda.is_available():
        return ["cpu"]
    return [f"cuda:{i}  ({torch.cuda.get_device_name(i)})" for i in range(torch.cuda.device_count())]


def main() -> None:
    import tkinter as tk
    from tkinter import filedialog, ttk

    saved = json.loads(SETTINGS.read_text()) if SETTINGS.is_file() else {}
    root = tk.Tk()
    root.title("CausalVC Realtime")
    root.minsize(560, 420)
    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    known = voices()
    state = {"live": None, "stream": None, "busy": False, "seen": 0}
    v = {
        "voice": tk.StringVar(value=saved.get("voice", next(iter(known), ""))),
        "input": tk.StringVar(value=saved.get("input", "System default")),
        "output": tk.StringVar(value=saved.get("output", "System default")),
        "gpu": tk.StringVar(value=saved.get("gpu", "")),
        "block": tk.StringVar(value=str(saved.get("block", 40))),
        "pitch": tk.DoubleVar(value=saved.get("pitch", 0.0)),
        "in_gain": tk.DoubleVar(value=saved.get("in_gain", 0.0)),
        "out_gain": tk.DoubleVar(value=saved.get("out_gain", 0.0)),
        "gate": tk.DoubleVar(value=saved.get("gate", -60.0)),
        "denoise": tk.BooleanVar(value=saved.get("denoise", False)),
    }
    row = 0

    def label(text: str) -> None:
        ttk.Label(frame, text=text).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 10))

    def pick_voice() -> None:
        path = filedialog.askopenfilename(filetypes=[("CausalVC voice", "*.json")])
        if path:
            known[Path(path).name] = path
            voice_box.configure(values=list(known))
            v["voice"].set(Path(path).name)

    label("Voice")
    voice_box = ttk.Combobox(frame, textvariable=v["voice"], values=list(known), state="readonly")
    voice_box.grid(row=row, column=1, sticky="ew")
    ttk.Button(frame, text="File…", command=pick_voice).grid(row=row, column=2, padx=(6, 0))
    row += 1
    try:
        ins, outs = devices("input"), devices("output")
    except Exception as error:  # no PortAudio, or no audio devices
        ins = outs = ["System default"]
        problem = str(error)
    else:
        problem = ""
    for text, key, choices in (("Microphone", "input", ins), ("Output", "output", outs)):
        label(text)
        if v[key].get() not in choices:
            v[key].set("System default")
        ttk.Combobox(frame, textvariable=v[key], values=choices, state="readonly").grid(row=row, column=1, columnspan=2, sticky="ew")
        row += 1
    label("GPU")
    gpus = gpu_choices()
    if v["gpu"].get() not in gpus:
        v["gpu"].set(gpus[0])
    perf = ttk.Frame(frame)
    perf.grid(row=row, column=1, columnspan=2, sticky="w")
    ttk.Combobox(perf, textvariable=v["gpu"], values=gpus, state="readonly", width=34).pack(side="left")
    ttk.Label(perf, text="Block (ms)").pack(side="left", padx=(14, 4))
    ttk.Combobox(perf, textvariable=v["block"], values=["20", "40", "60", "80", "100"], state="readonly", width=5).pack(side="left")
    row += 1

    def slider(text: str, key: str, low: float, high: float, step: float, unit: str) -> None:
        nonlocal row
        label(text)
        shown = ttk.Label(frame, width=10)

        def moved(_=None) -> None:
            value = round(v[key].get() / step) * step
            v[key].set(value)
            shown.configure(text=f"{value:+.1f} {unit}" if key != "gate" or value > -60 else "off")
            live = state["live"]
            if live is not None:  # these apply while running
                if key == "pitch":
                    live.set_pitch(value)
                else:
                    setattr(live, {"in_gain": "in_gain_db", "out_gain": "out_gain_db", "gate": "gate_db"}[key], value)

        ttk.Scale(frame, variable=v[key], from_=low, to=high, command=moved).grid(row=row, column=1, sticky="ew")
        shown.grid(row=row, column=2, sticky="w", padx=(6, 0))
        moved()
        row += 1

    slider("Pitch", "pitch", -24, 24, 0.5, "st")
    slider("Input gain", "in_gain", -24, 24, 0.5, "dB")
    slider("Output gain", "out_gain", -24, 24, 0.5, "dB")
    slider("Noise gate", "gate", -60, 0, 1.0, "dB")

    from causalvc import rnnoise

    def denoise_changed() -> None:
        if state["live"] is not None:  # applies while running
            state["live"].denoise = v["denoise"].get()

    noise_box = ttk.Checkbutton(frame, text="Noise reduction (RNNoise): removes background noise from the microphone, adds 20 ms",
                                variable=v["denoise"], command=denoise_changed)
    noise_box.grid(row=row, column=1, columnspan=2, sticky="w", pady=(4, 0))
    if not rnnoise.available():
        v["denoise"].set(False)
        noise_box.configure(state="disabled", text="Noise reduction (RNNoise) is not available on this system")
    row += 1

    buttons = ttk.Frame(frame)
    buttons.grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 4))
    start_btn = ttk.Button(buttons, text="Start")
    stop_btn = ttk.Button(buttons, text="Stop", state="disabled")
    start_btn.pack(side="left")
    stop_btn.pack(side="left", padx=(8, 0))
    row += 1
    status = ttk.Label(frame, text=problem or "Idle", wraplength=520, justify="left")
    status.grid(row=row, column=0, columnspan=3, sticky="w")
    row += 1
    timing = ttk.Label(frame, text="", wraplength=520, justify="left")
    timing.grid(row=row, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def remember() -> None:
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps({k: var.get() for k, var in v.items()}, indent=1))

    def running(on: bool) -> None:
        start_btn.configure(state="disabled" if on else "normal")
        stop_btn.configure(state="normal" if on else "disabled")

    def start() -> None:
        path = known.get(v["voice"].get())
        if not path or not Path(path).is_file():
            status.configure(text="Choose a voice first. Voices appear here after one is built.")
            return
        remember()
        running(True)
        stop_btn.configure(state="disabled")
        status.configure(text="Loading the voice (a few seconds)…")
        state["busy"] = True
        args = (path, v["gpu"].get().split()[0], int(v["block"].get()), v["pitch"].get(), v["in_gain"].get(), v["out_gain"].get(), v["gate"].get(), v["denoise"].get())
        route = (device_index(v["input"].get()), device_index(v["output"].get()))

        def work() -> None:
            try:
                from causalvc.live import Live

                live = Live(*args)
                state["stream"], state["live"], state["seen"] = live.stream(*route), live, 0
                state["error"] = None
            except Exception as error:
                state["error"] = f"Could not start: {error}"
            state["busy"] = False

        threading.Thread(target=work, daemon=True).start()

    def stop() -> None:
        stream, state["stream"], state["live"] = state["stream"], None, None
        if stream is not None:
            stream.stop()
            stream.close()
        running(False)
        status.configure(text="Stopped")

    def tick() -> None:
        if state.get("error"):
            status.configure(text=state.pop("error"))
            running(False)
        live, stream = state["live"], state["stream"]
        if live is not None and stream is not None and not state["busy"]:
            stop_btn.configure(state="normal")
            if not stream.active:
                stop()
                status.configure(text="The audio device stopped")
            else:
                from causalvc.live import SR

                block_ms = live.block / SR * 1000
                noise = " (including 20 ms of noise reduction)" if live.denoise and live.rnnoise is not None else ""
                status.configure(text=f"Converting as {live.speaker}: delay {live.model_delay_ms:.0f} ms{noise} + {block_ms:.0f} ms block")
                recent = live.stats(len(live.times) - state["seen"]) if len(live.times) > state["seen"] else {}
                state["seen"] = len(live.times)
                if recent:
                    timing.configure(text=f"compute {recent['median_ms']:.1f} ms per block (worst recent {recent['max_ms']:.1f} ms), late callbacks {recent['late']}")
        root.after(1000, tick)

    def close() -> None:
        remember()
        if state["stream"] is not None:
            stop()
        root.destroy()

    start_btn.configure(command=start)
    stop_btn.configure(command=stop)
    root.protocol("WM_DELETE_WINDOW", close)
    hook = os.environ.get("CAUSALVC_GUI_TEST")  # test hook: start at once, stop and close after N seconds
    if hook:
        root.after(300, start)

        def finish() -> None:
            print("status:", status.cget("text"), "|", timing.cget("text"), flush=True)
            close()

        root.after(int(float(hook) * 1000), finish)
    root.after(1000, tick)
    root.mainloop()


if __name__ == "__main__":
    main()
