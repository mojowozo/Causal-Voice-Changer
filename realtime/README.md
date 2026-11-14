# CausalVC Realtime

The live voice changer for CausalVC voices: microphone in, converted voice out,
with a COSMIC desktop front end. Linux only for now.

```bash
./run.sh
```

It uses the Python environment of the repository above it, so run the
repository's `run.sh` once first. The front end is built with `cargo` on first
launch. `scripts/install-desktop-entry.sh` adds it to the application launcher.

## Using it

- **Voice**: pick a voice built with the training window. Voices live in
  [models/](models/README.md).
- **Pitch**: semitones to shift your voice by, to sit in the target's range.
- **Chunk**: how much audio is converted at a time, in steps of 20 ms. Delay is
  the model's fixed 60 ms plus one chunk, plus the sound card's buffers.
- **Input and output gain, noise gate, input noise reduction** (RNNoise, when
  installed), **monitoring, recording, and file input** work as in any such app.

There is no index, pitch method, protect, or context setting: none apply to
this model.

## Layout

| Folder | What it is |
|---|---|
| `engine/` | The audio engine: devices, routing, settings, recording; one process, driven over a Unix socket ([protocol](docs/engine-protocol.md)) |
| `infer/causalvc_backend.py` | Loads a voice and converts blocks, through the `causalvc` package |
| `tools/` | Audio helpers: FIFO, routing, limiter, loudness meter, recorder, file source, RNNoise |
| `ui/` | The COSMIC front end (Rust) |
| `tests/` | `../.venv/bin/python -m unittest discover -s tests` |

See [NOTICE.md](NOTICE.md) for where this code comes from.
