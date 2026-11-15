"""A small window for building a voice (the fallback when the COSMIC or WinUI app can't be built).

    PYTHONPATH=$PWD .venv/bin/python -m causalvc.gui

It runs `python -m causalvc.build` as a separate process and shows its progress.
Stop asks the trainer to save a checkpoint and leave (a second click ends it at
once); starting again with the same name resumes from the last checkpoint.
Queued builds start in order after the current one finishes. Uses only the
standard library, and nothing here is specific to Linux or Windows.
"""

from __future__ import annotations

import json
import os
import queue
import re
from pathlib import Path

from causalvc.trainer import NOISE, PROGRESS, ROOT, SETTINGS, Job, custom_bases, gpus

SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
COLOURS = ("#ff5f5f", "#ff875f", "#ffaf5f", "#ffd75f", "#d7ff5f", "#87ff5f", "#5fffaf", "#5fffff", "#5fafff", "#875fff", "#d75fff", "#ff5fd7")


def main() -> None:
    import tkinter as tk
    from tkinter import filedialog, ttk

    saved = json.loads(SETTINGS.read_text()) if SETTINGS.is_file() else {}
    job = Job()
    pending: list[dict] = []
    root = tk.Tk()
    root.title("Causal VC Model Trainer")
    root.minsize(640, 520)
    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    v = {
        "audio": tk.StringVar(value=saved.get("audio", "")),
        "name": tk.StringVar(value=saved.get("name", "")),
        "batch": tk.StringVar(value=str(saved.get("batch", 8))),
        "steps": tk.StringVar(value=str(saved.get("steps", 10000))),
        "save_every": tk.StringVar(value=str(saved.get("save_every", 10000))),
        "custom": tk.BooleanVar(value=False),
        "base": tk.StringVar(value=""),
        "device": tk.StringVar(value=""),
        "restart": tk.BooleanVar(value=False),
        "fixes": tk.BooleanVar(value=saved.get("fixes", True)),
        "app_models": tk.StringVar(value=saved.get("app_models", str(ROOT / "realtime" / "models"))),
    }
    row = 0

    def label(text: str) -> None:
        ttk.Label(frame, text=text).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 10))

    def browse(var: tk.StringVar, folder: bool = True) -> None:
        path = filedialog.askdirectory() if folder else filedialog.askopenfilename()
        if path:
            var.set(path)

    label("Dataset")
    ttk.Entry(frame, textvariable=v["audio"]).grid(row=row, column=1, sticky="ew")
    pick = ttk.Frame(frame)
    pick.grid(row=row, column=2, padx=(6, 0))
    ttk.Button(pick, text="Folder…", command=lambda: browse(v["audio"])).pack(side="left")
    ttk.Button(pick, text="File…", command=lambda: browse(v["audio"], False)).pack(side="left", padx=(4, 0))
    row += 1
    label("Voice name")
    ttk.Entry(frame, textvariable=v["name"]).grid(row=row, column=1, sticky="ew")
    row += 1
    label("Training")
    nums = ttk.Frame(frame)
    nums.grid(row=row, column=1, columnspan=2, sticky="w")
    for text, key, width in (("Batch size", "batch", 5), ("Steps", "steps", 8), ("Save every", "save_every", 8)):
        ttk.Label(nums, text=text).pack(side="left")
        ttk.Entry(nums, textvariable=v[key], width=width).pack(side="left", padx=(4, 14))
    row += 1
    label("Base")
    base_row = ttk.Frame(frame)
    base_row.grid(row=row, column=1, columnspan=2, sticky="w")
    names = custom_bases()
    combo = ttk.Combobox(base_row, textvariable=v["base"], values=names, state="disabled", width=28)

    def toggle_base() -> None:
        combo.configure(state="readonly" if v["custom"].get() else "disabled")
        if v["custom"].get() and names and not v["base"].get():
            v["base"].set(names[0])

    check = ttk.Checkbutton(base_row, text="Custom base", variable=v["custom"], command=toggle_base)
    check.pack(side="left")
    combo.pack(side="left", padx=(8, 0))
    if not names:
        check.configure(state="disabled")
        ttk.Label(base_row, text="(none in custom-base/, using the default)").pack(side="left", padx=(8, 0))
    row += 1
    label("GPU")
    devices = gpus()
    wanted = saved.get("device", "cuda:0")
    v["device"].set(next((d for d in devices if d.split()[0] == wanted), devices[0]))
    ttk.Combobox(frame, textvariable=v["device"], values=devices, state="readonly").grid(row=row, column=1, sticky="w")
    row += 1
    label("Voice changer voices")
    ttk.Entry(frame, textvariable=v["app_models"]).grid(row=row, column=1, sticky="ew")
    ttk.Button(frame, text="Folder…", command=lambda: browse(v["app_models"])).grid(row=row, column=2, padx=(6, 0), sticky="w")
    row += 1
    ttk.Checkbutton(frame, text="Start over (ignores existing checkpoints for this voice name)", variable=v["restart"]).grid(
        row=row, column=1, columnspan=2, sticky="w", pady=(2, 8))
    row += 1
    ttk.Checkbutton(frame, text="Quality fixes: natural breaths and whispers, brighter tone (about 25% slower)", variable=v["fixes"]).grid(
        row=row, column=1, columnspan=2, sticky="w", pady=(0, 8))
    row += 1

    buttons = ttk.Frame(frame)
    buttons.grid(row=row, column=0, columnspan=3, sticky="w")
    start_btn = ttk.Button(buttons, text="Start training")
    queue_btn = ttk.Button(buttons, text="Queue build")
    stop_btn = ttk.Button(buttons, text="Stop", state="disabled")
    start_btn.pack(side="left")
    queue_btn.pack(side="left", padx=(8, 0))
    stop_btn.pack(side="left", padx=(8, 0))
    row += 1

    queue_frame = ttk.LabelFrame(frame, text="Build queue (runs in order)", padding=6)
    queue_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 0))
    queue_frame.columnconfigure(0, weight=1)
    queue_list = tk.Listbox(queue_frame, height=3, exportselection=False)
    queue_list.grid(row=0, column=0, sticky="ew")

    def refresh_queue() -> None:
        queue_list.delete(0, "end")
        for index, config in enumerate(pending, 1):
            queue_list.insert("end", f"{index}. {config['name']} — {config['steps']} steps")

    def remove_queued() -> None:
        selection = queue_list.curselection()
        if selection:
            pending.pop(selection[0])
            refresh_queue()

    ttk.Button(queue_frame, text="Remove selected", command=remove_queued).grid(row=0, column=1, padx=(8, 0), sticky="n")
    row += 1

    status = ttk.Frame(frame)
    status.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(12, 4))
    spinner = tk.Label(status, text=" ", font=("TkFixedFont", 16), width=2)
    spinner.pack(side="left")
    state = ttk.Label(status, text="Idle")
    state.pack(side="left", padx=(6, 0))
    row += 1
    bar = ttk.Progressbar(frame, maximum=100)
    bar.grid(row=row, column=0, columnspan=3, sticky="ew")
    row += 1
    log = tk.Text(frame, height=12, state="disabled", wrap="word")
    log.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
    frame.rowconfigure(row, weight=1)

    def say(text: str) -> None:
        log.configure(state="normal")
        log.insert("end", text + "\n")
        log.see("end")
        log.configure(state="disabled")

    def set_running(on: bool) -> None:
        start_btn.configure(state="disabled" if on else "normal")
        stop_btn.configure(state="normal" if on else "disabled", text="Stop")
        if not on:
            spinner.configure(text=" ")

    def launch(config: dict) -> None:
        job.start(**config)
        bar.configure(value=0)
        state.configure(text=f"Starting {config['name']} (loading the models takes a little while)…")
        set_running(True)

    def start_next() -> None:
        if pending and not job.running() and not job.stopping:
            config = pending.pop(0)
            refresh_queue()
            launch(config)

    def collect_config() -> dict | None:
        try:
            name, audio = v["name"].get().strip(), v["audio"].get().strip()
            batch, steps, save_every = (int(v[k].get()) for k in ("batch", "steps", "save_every"))
            if not name or not re.fullmatch(r"[\w.-]+", name):
                raise ValueError("Give the voice a name using letters, digits, '-', '_' or '.'")
            if not audio or not Path(audio).exists():
                raise ValueError("The dataset path does not exist")
            if min(batch, steps, save_every) < 1:
                raise ValueError("Batch size, steps and save interval must be at least 1")
        except ValueError as err:
            state.configure(text=str(err) if "invalid literal" not in str(err) else "Batch size, steps and save interval must be whole numbers")
            return None
        device = v["device"].get().split()[0]
        app_models = v["app_models"].get().strip()
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps({"audio": audio, "name": name, "batch": batch, "steps": steps, "save_every": save_every,
                                        "device": device, "app_models": app_models, "fixes": v["fixes"].get()}, indent=1))
        base = v["base"].get() if v["custom"].get() and v["base"].get() else "default"
        return {"name": name, "audio": audio, "batch": batch, "steps": steps, "save_every": save_every,
                "base": base, "device": device, "restart": v["restart"].get(), "app_models": app_models,
                "fixes": v["fixes"].get()}

    def start() -> None:
        config = collect_config()
        if config is not None:
            launch(config)
            v["restart"].set(False)

    def queue_build() -> None:
        config = collect_config()
        if config is None:
            return
        pending.append(config)
        refresh_queue()
        v["restart"].set(False)
        if job.running():
            state.configure(text=f"Queued {config['name']} ({len(pending)} waiting)")
        else:
            start_next()

    def stop() -> None:
        if job.stopping:
            state.configure(text="Ending now, without saving")
        else:
            state.configure(text="Stopping: saving a checkpoint first (click again to end at once)")
            stop_btn.configure(text="End now")
        job.stop()

    tick = {"n": 0}

    def pump() -> None:
        try:
            while True:
                line = job.lines.get_nowait()
                if line is None:
                    code = job.proc.returncode
                    state.configure(text="Finished" if code == 0 and not job.stopping else "Stopped" if job.stopping else f"Failed (exit code {code}); see the log below")
                    set_running(False)
                    if not job.stopping and pending:
                        start_next()
                    continue
                m = PROGRESS.match(line)
                if m:
                    step, total, pct, elapsed, left, rate = m.groups()
                    bar.configure(value=100 * int(step) / int(total))
                    if not job.stopping:
                        state.configure(text=f"Step {step} of {total} ({pct}%)   elapsed {elapsed}   remaining {left}   {rate} it/s")
                elif line.strip() and not any(n in line for n in NOISE):
                    say(line)
        except queue.Empty:
            pass
        if job.running():
            tick["n"] += 1
            spinner.configure(text=SPIN[tick["n"] % len(SPIN)], fg=COLOURS[tick["n"] % len(COLOURS)])
        root.after(100, pump)

    def close() -> None:
        if job.running():
            job.proc.kill()
        root.destroy()

    start_btn.configure(command=start)
    queue_btn.configure(command=queue_build)
    stop_btn.configure(command=stop)
    root.protocol("WM_DELETE_WINDOW", close)
    hook = os.environ.get("CAUSALVC_GUI_TEST")  # test hook: start at once, optionally stop after N seconds, close when done
    if hook:
        root.after(500, start)
        if float(hook) > 0:
            root.after(int(float(hook) * 1000), stop)

        def watch() -> None:
            if not job.running() and job.proc is not None and job.lines.empty():
                print("GUI state:", state.cget("text"), "| bar", round(float(bar.cget("value"))), flush=True)
                root.destroy()
            else:
                root.after(500, watch)

        root.after(5000, watch)
    root.after(100, pump)
    root.mainloop()


if __name__ == "__main__":
    main()
