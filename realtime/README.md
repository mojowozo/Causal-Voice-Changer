# CausalVC Realtime

The live voice changer for CausalVC voices: microphone in, converted voice out.
Its front end is native on each system: COSMIC (Rust) on Linux, WinUI 3 with
acrylic on Windows. Both drive the same Python engine.

- **Linux:** `../run-realtime.sh`. The front end is built with `cargo` on first
  launch, and again on the next launch after an update. `scripts/install-desktop-entry.sh` adds it to the application launcher.
- **Windows:** `..\run-realtime.bat`. It installs the Python packages (PyTorch
  for NVIDIA or AMD, picked by `../scripts/gpu-extra.ps1`; set `CAUSALVC_GPU=cuda`,
  `rocm` or `rocm-rdna2` to choose), then builds the front end with the .NET 10 SDK on first
  launch, installing that SDK into `..\.dotnet` when it is missing. Set `CAUSALVC_REBUILD=1` to
  build it again after updating. If the SDK cannot be installed, a smaller window opens instead.

Both use the Python environment of the repository above this folder (`.venv`
on Linux, `.venv-win` on Windows).

### AMD graphics on Windows

AMD's ROCm build of PyTorch (ROCm 7.2.1, torch 2.9.1) runs on Radeon RX 7000
and 9000, Radeon PRO W7000, and Ryzen AI Max and AI 300 graphics, with AMD
Software 26.2.2 or newer. It cannot start on the RX 6000 series, so those cards
(and Radeon PRO W6000) get AMD's newer nightly build instead (the `rocm-rdna2`
extra: TheRock, ROCm 10.3, torch 2.16). Older Radeons
run voices on the processor: set Chunk to 60 ms or more.

## Using it

- **Voice**: pick a voice built with the training window. Voices live in
  [models/](models/README.md).
- **Pitch**: semitones to shift your voice by, to sit in the target's range.
  **Advanced pitch**, a collapsed row under it, holds two ways to tame high
  peaks, both off by default and saved per voice like the pitch:
  - **Pitch offset** (Hz), added after the shift: output pitch = your pitch ×
    2^(semitones/12) + offset, on voiced sounds only. A positive offset raises
    low notes more than high ones, so it narrows your range (and bends sung
    intervals); a negative one widens it.
  - **Pitch ceiling** (Hz) and **Above the ceiling** (%): output pitch above
    the ceiling keeps only that share of its rise, in semitones (0% stops at
    the ceiling). Notes below it keep their intervals, so singing stays in tune.
- **Index** (%; off by default, saved per voice): how much of what you say is
  redrawn from the voice's own recordings, like an RVC index. Each 20 ms of
  what you say is matched to the closest-sounding moments of the voice's
  training audio and blended toward them. Higher can sound more like the
  voice; too high blurs words. Voices built from now on carry their index in
  their .cvc (about 12 MB more); for an older voice the row says it has none,
  and `python -m causalvc.index VOICE.cvc ... --audio FOLDER` adds one from
  its audio.
- **Chunk**: how much audio is converted at a time, in steps of 20 ms. Delay is
  the model's fixed 60 ms plus one chunk, plus the sound card's buffers.
- **Fast GPU replay** (Performance, on by default): the model's GPU work for a
  chunk is recorded once and replayed for every chunk after, about twice as fast
  and much steadier (an RTX 5050 converts a 20 ms chunk in 4.4 ms instead of
  9.7 ms). Turn it off if a graphics card converts wrongly or not at all with it.
- **Input noise reduction**: [Hushwire](https://github.com/mojowozo/Hushwire)
  (Amazon Voice Focus; the default, strongest, adds about 55 ms) or RNNoise
  (lighter, adds 20 ms). Hushwire's model is Amazon's and isn't included: the
  app asks before downloading it once from Amazon, and shares it with the
  Hushwire plugin if that's installed.
- **Input and output gain, noise gate, monitoring, recording, and file input**
  work as in any such app.

There is no pitch method, protect, or context setting: none apply to this
model.

## Layout

| Folder | What it is |
|---|---|
| `engine/` | The audio engine: devices, routing, settings, recording; one process, driven over a local socket ([protocol](docs/engine-protocol.md)) |
| `infer/causalvc_backend.py` | Loads a voice and converts blocks, through the `causalvc` package |
| `tools/` | Audio helpers: FIFO, routing, limiter, loudness meter, recorder, file source, RNNoise, Hushwire |
| `ui/` | The COSMIC front end (Rust), for Linux |
| `winui/` | The WinUI 3 front end (C#), for Windows: the same pages, on acrylic |
| `tests/` | `../.venv/bin/python -m unittest discover -s tests` |

See [NOTICE.md](NOTICE.md) for where this code comes from.
