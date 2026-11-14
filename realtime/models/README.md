# Voices

One folder per voice. The folder name is the voice's name in the app, and the
folder holds the voice's `.causalvc.json` file, which is written when the voice
is built:

```text
models/
  MyVoice-10000s/
    MyVoice-10000s.causalvc.json
```

Building a voice with the training window (or `causalvc.build --app-models`)
puts every checkpoint here. A `.causalvc.json` file can also be dragged onto
the app's window. After adding a folder by hand, use **Reload** in the app.

The app saves each voice's pitch and gain settings beside it as
`realtime_settings.json`.
