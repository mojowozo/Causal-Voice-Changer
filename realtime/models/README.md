# Voices

Voices travel as single `.cvc` files and are installed here as one folder each.
The folder name is the voice's name in the app, and the folder is
self-contained: its description and its decoder, with nothing that names a
place on the machine that made it.

```text
models/
  MyVoice-10000s/
    MyVoice-10000s.causalvc.json
    decoder.pt
```

To add a voice, drag its `.cvc` file onto the app's window, or put the file
here and use **Reload**: it is unpacked into a folder and the file removed. To
share one, use **Export** in the library, which writes a `.cvc`
(`python -m causalvc.voice cvc FOLDER` does the same from a terminal). Building
a voice with the training window installs every checkpoint here automatically.

The app saves each voice's pitch and gain settings beside it as
`realtime_settings.json`; Export puts them in the `.cvc`, and they come along
when it is installed.

A `.cvc` holds only its own voice: a decoder that still contains the other
speakers of the base it was trained on (older folders can point at one) is cut
down to the one speaker when exported. A voice built on its own content encoder
or pitch tracker, instead of the shared ones, carries that file too
(`encoder.pt` or `pitch.pt` in its folder), so it sounds the same elsewhere.

To convert voices on a computer without this project's tools at hand, or to
hand someone a way to convert theirs, give them
[scripts/convert_to_cvc.py](../../scripts/convert_to_cvc.py): it is a single
file that needs only Python and PyTorch, and converts a voice folder, or every
voice in a folder such as `realtime/models`, into `.cvc` files in
`./cvc-voices` without changing the originals.

A voice made by an older version (a lone `.causalvc.json` or `.nextvc.json` that
points at weights elsewhere) can be converted with
`python -m causalvc.voice pack PATH`.
