# Engine protocol (version 1)

The real-time engine runs as its own process and is controlled over a local
stream socket: a Unix socket on Linux, loopback TCP on Windows. The COSMIC
front end (`ui/`), the WinUI front end (`winui/`), the `engine.cli` client,
and the tests all use this protocol.

```sh
.venv/bin/python -m engine.server [--socket PATH] [--exit-when-idle SECONDS]
.venv/bin/python -m engine.cli watch          # print events
.venv/bin/python -m engine.cli state          # current state as JSON
```

- **Socket (Linux):** `$XDG_RUNTIME_DIR/causalvc-realtime.sock` by default (mode `0600`).
  The server refuses to start if another engine already listens there.
- **TCP (Windows):** asyncio has no Unix sockets there, so the engine listens on
  `127.0.0.1` on a free port and writes `{"port", "token", "pid"}` to
  `%LOCALAPPDATA%\CausalVC\realtime-engine.json` (per user). A client's first
  line must be `{"auth": "<token>"}`; a client that sends anything else, or
  nothing within 5 s, is disconnected without a `hello`. Pass
  `--socket tcp:<endpoint file>` to use this transport elsewhere (the tests do).
- **Framing:** one UTF-8 JSON object per line (`\n`), in both directions.
- **Startup:** the socket is bound *before* PyTorch and the engine load
  (about 3 s). Clients can connect immediately; engine commands fail with
  `not_ready` until the `ready` event.
- **`--exit-when-idle`:** the server shuts down after no client has been
  connected for that many seconds. A front end that spawns the engine should
  pass this so the engine never outlives it.
- **Ordering:** engine commands run one at a time on a single thread, in the
  order received. A slow command (`start` loads the model and warms up CUDA
  Graphs, which can take seconds) delays later ones; events keep flowing.

## Messages

```jsonc
// client -> engine
{"id": 7, "cmd": "update_settings", "args": {"settings": {"pitch": 12}}}

// engine -> client: response (exactly one per request, matched by "id")
{"id": 7, "ok": true, "result": null}
{"id": 7, "ok": false, "error": {"code": "bad_setting", "message": "pitch must be a number", "details": {"key": "pitch"}}}

// engine -> client: event (any time, interleaved with responses)
{"event": "meters", "data": {...}}
```

`message` is English and meant as a fallback or for logs; front ends should
translate by `code`.

## Commands

| Command | Args | Result |
| --- | --- | --- |
| `ping` | | `{"protocol": 1, "ready": bool}` (works before ready) |
| `get_state` | | [State](#state) |
| `update_settings` | `{"settings": {key: value, ...}}` | `null` |
| `reset_settings` | `{"group": "general" \| "performance"}` | `null` |
| `reload_models` | | `null` |
| `import_model` | `{"paths": ["/abs/Voice-10000s.cvc", …]}` | `{"models": ["Voice-10000s"]}`; copies each `.cvc` into its voice's folder, `models/<voice>/<name>.cvc` (`Voice-10000s` goes into `Voice/`; an older `.causalvc.json` description is packed into one) and selects the first. A `.cvc` placed in `models/` or one folder down by hand is a voice as it is, listed on startup and `reload_models` |
| `download_hushwire` | — | `null`; records that the user accepted Amazon's download notice (the client shows it first; `hushwire-notice-body`) and fetches Hushwire's model in the background into Hushwire's own folder (`~/.local/share/Hushwire/models`, `%APPDATA%\Hushwire\models`), shared with the Hushwire plugin. Ends with status `hushwire_downloaded` or error `hushwire_download_failed` |
| `merge_models` | `{"voices": [{"name": "Kira-1000s", "strength": 60}, {"name": "Mika", "strength": 40}], "name": "Kira-Mika"}` | `{"name": …}`; writes a new voice, `models/<voice>/<name>.cvc`, whose decoder weights are the strength-weighted average of the voices' (strengths are shares; zero leaves a voice out), with the first voice's settings, and selects it. Refused with `merge_incompatible` unless every voice has the same `merge_family` (see `models`), `merge_failed` for fewer than two distinct voices |
| `export_model` | `{"name": "Voice", "path": "/abs/Voice.cvc"}` | `{"path": …}`; writes the voice with its current settings as one `.cvc` file (`.cvc` is appended if missing) |
| `delete_model` | `{"name": "Voice"}` | `null`; moves the voice (its `.cvc`, or an older voice's folder) to the desktop Trash, removing its voice's folder once that is empty, (`gio trash` on Linux, the Recycle Bin on Windows; recoverable) and selects another model if it was selected. Refused with `model_in_use` while converting with it |
| `rename_model` | `{"name": "Voice", "new_name": "Voice v2"}` | `null`; renames the voice's file (or an older voice's folder; settings go with it; the selection follows). Names must be unique (ignoring case), not start with `.`, and avoid `/ \ : * ? " < > \|`. Refused with `model_in_use` while converting with it |
| `reload_devices` | | `null` |
| `asio_control_panel` | — | `null`; opens the selected ASIO device's own control panel (Windows). Refused while running (`asio_panel_busy`); `asio_panel_failed` when the driver can't show it |
| `start` | `{"function": "vc" \| "passthrough"}` (default `vc`) | `null` |
| `stop` | | `null` |
| `file_select` | `{"path": "/abs/file.wav"}` | `null` |
| `file_play` / `file_pause` / `file_stop` | | `null` |
| `file_seek` | `{"seconds": number}` | `null` |
| `record_start` | | `null` |
| `record_stop` | | `{"paths": ["/…/file.wav", …]}` |
| `get_log` | | `{"text": "…"}` (last 40,000 characters) |
| `clear_log` | | `null` |
| `save_log` | | `{"path": "logs/RuntimeLog_….txt"}` |
| `shutdown` | | `null`, then the server stops the engine and exits |

`update_settings` validates every key before applying any. Changing a
*restart* setting while running stops the stream (with status
`settings_changed`); the others apply live.

## Settings

| Key | Type | Notes |
| --- | --- | --- |
| `model_name` | string | restart; must be a model (checkpoint) name from state. Its voice's settings are loaded with it. Another checkpoint of the same voice (state `voices`) keeps the voice's settings and, while running, does not restart: only the decoder is loaded and swapped in (status `loading_model` with `model`, then `checkpoint_switched` with `name`; a failed load keeps the old checkpoint and reports `model_load_failed`). Picking a checkpoint other than the voice's newest is remembered as the voice's `checkpoint` |
| `gpu` | string | restart; `"auto"` or a GPU `id` from state |
| `input_source` | `"microphone"` \| `"file"` | restart |
| `input_device` / `output_device` | string | restart; a device `label` from state |
| `monitor_device` | string \| null | restart; `null` disables the monitor |
| `block_time` | number (s) | restart; min 0.02 |
| `asio_buffer` | 0, 32, 64, 96, 128, 192, 256, 512, 1024 or 2048 | restart; the ASIO buffer to ask the driver for, in samples. 0 (default) uses the driver's current size, read at start. PortAudio rounds to a size the driver accepts (state `asio.sizes`) |
| `cuda_graphs` | bool (default true) | restart: replay each block's recorded GPU work (a CUDA graph), about twice as fast and steadier; the stream then starts from four seconds of recorded silence. Where recording fails the voice converts block by block, and the log says so |
| `input_denoise` | bool | live: noise reduction on the input, with `input_denoise_method` |
| `input_denoise_method` | `"hushwire"` (default) or `"rnnoise"` | live: Hushwire (Amazon Voice Focus, adds ~53–55 ms; needs its model, state `hushwire`) or RNNoise (adds 20 ms, state `rnnoise`). A denoiser that can't run leaves the input alone |
| `loudness_target` | `"voice_chat"` \| `"streaming"` | live (default `"voice_chat"`): the output-level target, -21 or -16 LUFS (see [Output level](#output-level)) |
| `recording_folder` | string | live |
| `recording_mode` | `"separate"` \| `"mix"` \| `"stereo"` | live (next recording) |
| `file_input_volume` | number 0–1 | live |
| `input_gain_db` / `output_gain_db` / `monitor_gain_db` | number | live, per voice |
| `noise_gate_db` | number | live, per voice (−60 = off) |
| `pitch` | number (semitones) | live, per voice |
| `pitch_offset_hz` | number (Hz, −300 to 300) added to the shifted pitch of voiced frames | live, per voice |
| `pitch_ceiling_hz` | number (Hz, 0–1000; 0 = none): output pitch above it keeps only `pitch_ceiling_keep` of its rise | live, per voice |
| `pitch_ceiling_keep` | number 0–1 (default 0.25): share of the rise above the ceiling kept, in semitones | live, per voice |
| `index_rate` | number 0–1 (default 0 = off): how much of each content frame is drawn from the voice's index; does nothing for a voice without one (`has_index`) | live, per voice |

Per-voice settings are shared by every checkpoint of a voice and saved under the voice's
name in `configs/voice_settings.json` (with `"checkpoint"`, the picked checkpoint when it is
not the newest). Entries saved per checkpoint by earlier versions stay in the file; a voice
without its own entry takes the selected checkpoint's, else its newest checkpoint's that has one.

A block is converted as it arrives and the output is continuous from block to
block, so there is no crossfade or context setting. Delay is the model's fixed
60 ms plus one `block_time`, plus the sound card's buffers.

## State

Front ends can re-render entirely from the latest state.

```jsonc
{
  "running": false,
  "function": "vc",                    // or "passthrough"
  "settings": { /* every key above */ },
  "models": [{"name": "MyVoice-10000s", "model_file": "MyVoice-10000s.cvc",
              "size_bytes": 48000000,       // of the .cvc (an older voice folder: its decoder)
              "modified": 1791067045,       // the file's modification time (Unix seconds)
              "merge_family": "fbb46583a204",  // equal for voices that can be merged together; null: cannot be merged
              "has_index": true,            // the file carries an index (index_rate does something)
              "voice": "MyVoice",           // the voice it is a checkpoint of (see voices)
              "step": 10000}],              // the training step in its name, or null
  // Each voice once. A voice's checkpoints are the .cvc files in its folder (models/MyVoice/),
  // or loose files named MyVoice-<step>s; any other model is a voice with one checkpoint.
  "voices": [{"name": "MyVoice",
              "checkpoints": ["MyVoice-1000s", "MyVoice-10000s"],  // by step, then those without one
              "checkpoint": "MyVoice-10000s"}],  // what picking the voice loads: the picked one, else the newest
  "devices": {
    "inputs":  [{"label": "[JACK] Elgato Wave XLR Mono", "api": "JACK",
                 "name": "Elgato Wave XLR Mono", "channels": [1, 1]}],
    "outputs": [{"label": "[ALSA] default", "api": "ALSA", "name": "default", "channels": []}]
    // Windows: "[WASAPI] Microphone (Wave XLR)", "[ASIO] Focusrite USB ASIO — 1 / 2", …
  },
  "gpus": [{"id": "auto", "label": null}, {"id": "0", "label": "GPU 0 — NVIDIA GeForce RTX 4080"}],  // or an AMD GPU, through ROCm
  "samplerate": null,                  // Hz while running
  "delay_ms": null,                    // estimated end-to-end delay while running
  "recording": false,
  "file": {"path": null, "duration": null, "playing": false},
  "ffmpeg": true,
  "rnnoise": true,                    // input noise reduction (RNNoise) is installed
  "hushwire": "ready",               // Hushwire's model: ready, needs_download, downloading, failed, unavailable (no wasmtime)
  "asio": {                          // null unless the route uses an ASIO device
    "sizes": {"min": 32, "max": 2048, "preferred": 256, "granularity": -1},  // null if the driver can't say; granularity 0 = only min, max or preferred, -1 = powers of two
    "in_use": 256                    // samples per driver callback while running, else null
  }
}
```

`channels` lists the selected 1-based JACK or ASIO channels; it is empty for
ALSA and WASAPI devices. On Windows, WASAPI and ASIO devices are listed (ASIO
through sounddevice's ASIO build; set `CAUSALVC_ASIO=0` to leave it out), and
WASAPI converts to the model's 48 kHz when the Windows mixer runs at another rate.

## Events

| Event | Data |
| --- | --- |
| `hello` | `{"protocol": 1, "ready": bool, "init_error": string\|null, "state": State\|null}`, sent once on connect |
| `ready` | State; the engine finished starting |
| `fatal` | `{"code": "init_failed", "message": traceback}`; the engine could not start |
| `state` | State, after anything changes |
| `meters` | about 20 Hz while running or playing a file: `{"input", "output", "monitor"}` (0–1, −60 dB to 0 dB), `"infer_ms"`, `"recording_seconds"`, `"file_position"`, `"queued_ms"` (audio waiting in the input and output buffers right now; a standing backlog is drained by dropping silent blocks), `"underruns"` (output callbacks that found too little audio queued since the stream started: each is an audible gap; also logged), `"voice_pitch"` (as in State, live while running), `"loudness"` (see [Output level](#output-level)) (each may be `null`) |
| `status` | `{"code": …, …}`: progress and results, listed below |
| `error` | `{"code", "message"}`: a problem not tied to a request (e.g. `monitor_failed`, `file_error`) |
| `log` | `{"text": "…"}`: engine stdout/stderr as it is written |

Status codes: `importing_model`, `model_imported` (`names`), `model_exported` (`name`, `path`), `merging_models`, `model_merged` (`name`), `hushwire_downloaded`, `model_deleted` (`name`), `model_renamed` (`name`, `new_name`), `preparing`, `loading_model`, `checkpoint_switched` (`name`), `preparing_inference`,
`starting_audio`, `conversion_started`, `passthrough_started`,
`conversion_stopped`, `passthrough_stopped`, `settings_changed`,
`stream_stopped` (the audio device went away), `file_selected` (`name`),
`playing`, `paused`, `playback_stopped`, `recording_started`,
`recording_saved` (`paths`).

## Error codes

`bad_request`, `unknown_command`, `not_ready`, `init_failed`,
`internal_error`, `bad_setting`, `bad_device`, `gpu_invalid`, `no_model`,
`model_file_missing`, `no_audio_file`, `ffmpeg_missing`,
`no_common_samplerate`, `audio_start_failed`, `record_requires_running`,
`recording_failed`, `samplerate_unsupported`, `model_load_failed`, `import_no_model_file`, `import_failed`, `file_error`, `monitor_failed`, `log_save_failed`,
`model_in_use`, `delete_failed`, `bad_model_name` (details: `reason`),
`model_name_taken`, `rename_failed`, `export_failed`, `merge_incompatible`, `merge_failed`, `hushwire_unavailable`, `hushwire_download_failed`,
`asio_one_device`, `asio_panel_busy`, `asio_panel_failed`.
