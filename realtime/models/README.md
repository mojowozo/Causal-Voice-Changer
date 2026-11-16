# Voices

Each voice is one `.cvc` file. A voice's checkpoints share a folder named
after it. The file name is the voice's name in the app, and the file is
self-contained: its description and its decoder, with nothing that names a
place on the machine that made it.

```text
models/
  MyVoice/
    MyVoice-5000s.cvc
    MyVoice-10000s.cvc
```

To add a voice, drag its `.cvc` file onto the app's window. It goes into its
voice's folder (`MyVoice-15000s.cvc` into `MyVoice/`). You can also put the
file here, in a folder or straight in `models/`, and use **Reload**. To share
one, use **Export** in the library, which writes a copy with your pitch and
gain settings inside. Building a voice with the training window, with **Add to
the voice changer** on, puts every checkpoint here. Deleting a voice's last
checkpoint removes its folder too.

The **Merge** page blends voices into a new one: each voice's strength sets
its share, and the result is written here as a voice of its own. Voices built
in the training window can be merged with each other; one built from another
base or with other settings cannot, and the page says so.
[causalvc/merge.py](../../causalvc/merge.py) does the same from the command
line.

The app keeps each voice's pitch and gain settings in
`configs/voice_settings.json`, by name, so the voice files are never changed.
A voice with none saved there starts from the settings inside its file.

The app reads a voice through an unpacked copy in your cache folder
(`~/.cache/causalvc/voices`, or `%LOCALAPPDATA%\causalvc\voices` on Windows),
made the first time it is used and removed once the file is renamed,
replaced or deleted.

A `.cvc` holds only its own voice: a decoder that still contains the other
speakers of the base it was trained on is cut down to the one speaker when
written. A voice built on its own content encoder or pitch tracker, instead of
the shared ones, carries that file too, so it sounds the same elsewhere.

Older versions unpacked each voice into a folder here
(`MyVoice-10000s/MyVoice-10000s.causalvc.json` and `decoder.pt`). A folder
holding a `.causalvc.json` is one of those, not a voice's folder. They still
work; Export turns one into a `.cvc`, and
[scripts/convert_to_cvc.py](../../scripts/convert_to_cvc.py) converts all of
them at once. That script is a single file that needs only Python and PyTorch,
so it can also be handed to someone to convert their voices; it writes the
`.cvc` files to `./cvc-voices` without changing the originals.

A voice made by an older version still (a lone `.causalvc.json` or
`.nextvc.json` that points at weights elsewhere) is packed into a `.cvc` when
it is dropped onto the window, or with `python -m causalvc.voice pack PATH`.
