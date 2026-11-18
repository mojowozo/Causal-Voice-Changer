# Causal Voice Changer
Causal Voice Changer is a 48Khz voice changer technology which attempts to bridge the voice changer gap between low latency options (ex. Beatrice) and higher-fidelity-but-slower options (ex. RVC) in order to achieve a balance between speed and quality instead of sacrificing one for the other. It uses a distilled version of ContentVec and RMVPE to achieve this. 

<img width="463" height="55" alt="Latency Measurement" src="https://github.com/user-attachments/assets/2d15c1d7-c32b-44bf-953b-d38e5f877341" />

Realtime example: 

https://github.com/user-attachments/assets/cbb8e298-6d2e-4720-b690-b81a42cb1582


## Differences from RVC

RVC takes in audio, converts the chunk, and then stitches them together via a crossfade algorithm. RVC was originally designed to convert files, and as a result its mechanism of voice changing starts to degrade at low block sizes, as both accuracy and audio quality drop. This is easy to notice if you set a low block size, and then attempt to draw out a word. You will hear the seams of the blocks as they stitch together. 

CVC by comparison converts audio as it receives it. It mostly only looks backward, with only a 60ms lookahead buffer, and streams the result out in realtime. In practicality, this means that you maintain nearly all of the quality of RVC, but at lower latency than RVC can reach. This also means that things such as sustained sounds no longer break and warble. 

Running on Linux (CachyOS, JACK audio) my total end-to-end latency often measures at 130ms. 

# Important Note
The base model/pretrain was built using EARS, VCTK, and LibriTTS-R. Because of the inclusion of Ears, you may **not** use the included pretrain for commercial usage, and this includes finetunes. By using this program, you accept that no models built using the pretrain will be used for commercial purposes. 

## Quick start

Each [release](https://github.com/mojowozo/Causal-Voice-Changer/releases) has a download for each
system with the app windows already built, so you need neither the .NET SDK nor Rust:
`Causal-Voice-Changer-<version>-windows-x64.zip` and `Causal-Voice-Changer-<version>-linux-x86_64.tar.gz`
(Linux with glibc 2.35 or newer: Ubuntu 22.04, Debian 12 and anything newer). Unpack it and
carry on below. Python, PyTorch and the pretrained files are not in it: they depend on your
graphics card, so `run-training` still installs them on first run. The windows are not signed, so
Windows may ask whether to run them ("More info", then "Run anyway"). With the source code instead
(a git clone, or the release's source download), the windows are built on first launch as described below,
and they are built again whenever their source files change.

Run `run-training.sh` (Linux) or `run-training.bat` (Windows). On first run, it installs the Python
packages with [uv](https://astral.sh/uv), downloads the pretrained files (the base
model and RMVPE, about 900 MB) into `weights/`, and opens the training window. This will also handle
installing the perquisites for the realtime portion, so please run this first! 

The training window is a COSMIC app on Linux (built with Rust's `cargo` on first run) and a WinUI 3
app on acrylic on Windows (built with the .NET 10 SDK, installed into `.dotnet` when missing). If it
can't be built, a small Tk window opens instead. See [training/README.md](training/README.md).

On Windows it installs the PyTorch build for your graphics card: CUDA for NVIDIA, ROCm for AMD
(Radeon RX 7000/9000 and Ryzen AI graphics; the RX 6000 series gets AMD's newer nightly build).
Set `CAUSALVC_GPU=cuda`, `rocm` or `rocm-rdna2` to choose yourself; when running `uv` commands by
hand on Windows, add `--extra cuda`, `--extra rocm` or `--extra rocm-rdna2`.

## Building a voice

In the window, or from the command line:

```bash
uv run python -m causalvc.build --name NAME --audio /path/to/folder-or-file
```

This fine-tunes the base decoder on that audio (10,000 steps, about 35 minutes on
an RTX 4080). The audio needs no slicing or silence handling and can be at any
sample rate. Every checkpoint is kept as its own voice, one file,
`runs/models/voice-NAME/voices/NAME-STEPs.cvc`; `--save-every` sets the interval.
`--app-models DIR` also installs each voice in a realtime app's models folder.

A `.cvc` file is the whole voice: send it, and on the other machine drag it onto
the realtime window (or put it in `realtime/models/` and press Reload). Nothing in
it refers to the machine that built it. The library's Export button turns an
installed voice back into a `.cvc`, with its pitch and gain settings; voices built
as folders by older versions convert with `python -m causalvc.voice cvc FOLDER`, or with
[scripts/convert_to_cvc.py](scripts/convert_to_cvc.py), a standalone copy to hand to others. `--base` takes `default`, a folder name
under `custom-base/`, or a path.

## Using a voice live
Use `run-realtime.sh` (Linux) or `run-realtime.bat` (Windows) to launch the GUI. Both open the full app in `realtime/`: a COSMIC window on Linux, a WinUI 3 window with acrylic on Windows (built on first launch with the .NET 10 SDK, which is installed into `.dotnet` if missing; if that fails, a smaller window with the basic controls opens instead). You need to ensure you've
run the training GUI first so the prerequisites can be installed. 

Windows GUI:
<img width="1897" height="1515" alt="gui-windows" src="https://github.com/user-attachments/assets/a20cfe23-5b69-4ffb-a4fd-078c1e36dcaf" />

Linux GUI:
<img width="1246" height="980" alt="gui-window-linux" src="https://github.com/user-attachments/assets/98fdf4f8-c2ee-481e-9a1d-7d6b7310b029" />


## Layout

| Module | What it is |
|---|---|
| `causalvc/encoder` | Causal content encoder, 16 kHz audio to 768-d features at 50 Hz, 40 ms lookahead |
| `causalvc/pitch_causal.py` | Causal pitch tracker, 100 frames per second, 52 ms delay |
| `causalvc/decoder` | Excitation source, decoder, discriminators and training |
| `causalvc/fast.py` | Block-by-block runtime that computes only what is new each block |
| `causalvc/stream.py` | The original block-by-block runtime (recomputes its whole window) |
| `causalvc/live.py` | Live audio in and out |
| `causalvc/infer.py` | Whole-file conversion |
| `causalvc/build.py`, `causalvc/trainer.py` | Building a voice: command, and the service the training windows drive |
| `causalvc/gui.py` | The small fallback training window (Tk) |
| `training/` | The training windows: COSMIC (`ui/`) and WinUI 3 (`winui/`) |
| `causalvc/weights.py` | Downloads of the pretrained files |

The harness and its metric models are an optional extra: `uv sync --extra bench`.

## Licence

The code is under the MIT licence ([LICENSE](LICENSE)). Parts come from other projects:

- `realtime/` is derived from RVC-Realtime-GUI and RVC WebUI (MIT; see [realtime/NOTICE.md](realtime/NOTICE.md) and [realtime/LICENSE](realtime/LICENSE)).
- `causalvc/rmvpe_model.py` is RMVPE's network code as carried by Applio (MIT).
- Input noise reduction uses RNNoise (BSD 3-Clause), installed as the `pyrnnoise` package.

The pretrained weights are a separate matter from the code. The base decoder is
trained on VCTK, EARS (CC BY-NC) and LibriTTS-R, so the base and every voice
built from it are for non-commercial use.
