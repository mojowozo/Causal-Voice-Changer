# Windows front end (WinUI 3)

The realtime voice changer's window on Windows: the same pages and control bar as the
COSMIC front end in `../ui`, 

Windows 10), with the control bar, drop overlay and toasts in in-app acrylic. It drives
the same Python engine (`../engine`), which on Windows listens on loopback TCP with a token
([protocol](../docs/engine-protocol.md)).

`..\..\run-realtime.bat` builds and opens it. By hand:

```bat
dotnet publish CausalVC.Realtime.csproj -c Release -o out
out\CausalVC.Realtime.exe
```

It needs the .NET 10 SDK to build (`winget install Microsoft.DotNet.SDK.10`) and nothing
to run: the build is unpackaged and carries the Windows App SDK runtime beside the exe.
The app starts the engine with the repository's `.venv-win` Python (made by
`run-realtime.bat`) and the engine exits 10 s after the window closes.

| File | What it is |
|---|---|
| `MainWindow.xaml(.cs)` | Window, navigation, connection screen, toasts, drag and drop; the app state (`app.rs`) |
| `Engine/` | The engine connection and its messages (`engine.rs`) |
| `Ui/Pages/` | One class per page (`pages.rs`, `library.rs`) |
| `Ui/Controls.cs` | Settings sections, rows and sliders with typed values |
| `Ui/Strings.cs` | English text, from `../ui/i18n/en` (Japanese is not ported yet) |
