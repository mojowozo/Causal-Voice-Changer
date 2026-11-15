# Voices

One folder per voice. The folder name is the voice's name in the app, and the
folder is self-contained: its description and its decoder, with nothing that
names a place on the machine that made it.

```text
models/
  MyVoice-10000s/
    MyVoice-10000s.causalvc.json
    decoder.pt
```

To share a voice, copy or zip its folder. To add one, put the folder here and
use **Reload** in the app. Building a voice with the training window puts every
checkpoint here automatically.

The app saves each voice's pitch and gain settings beside it as
`realtime_settings.json`; that file can be shared along with the voice or left out.

A voice made by an older version (a lone `.causalvc.json` or `.nextvc.json` that
points at weights elsewhere) can be converted with
`python -m causalvc.voice pack PATH`.
