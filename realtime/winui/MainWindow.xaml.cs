// The window and the app's state, as in ../ui/src/app.rs: engine events come in, the pages
// re-render from the latest state, and the pages' changes go out as engine commands.

using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using CausalVC.Realtime.Engine;
using CausalVC.Realtime.Ui;
using CausalVC.Realtime.Ui.Pages;
using Microsoft.UI.Composition.SystemBackdrops;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;
using Windows.ApplicationModel.DataTransfer;
using Windows.Graphics;
using Windows.Storage;

namespace CausalVC.Realtime;

public sealed partial class MainWindow : Window, IHost
{
    const int MaxLogChars = 40_000;
    const int ToastSeconds = 6;

    enum Pending { Start, Import, SaveLog }

    readonly DispatcherQueue _ui;
    readonly EngineConnection _engine;
    readonly Dictionary<long, Pending> _pending = [];
    readonly Dictionary<PageKind, UIElement> _pages = [];
    readonly ModelPage _modelPage;
    readonly LibraryPage _libraryPage;
    readonly AudioPage _audioPage;
    readonly PerformancePage _performancePage;
    readonly RecordingPage _recordingPage;
    readonly LogPage _logPage;
    readonly ControlBar _controlBar;
    readonly System.Text.StringBuilder _log = new();
    readonly DispatcherQueueTimer _logFlush;
    long? _logRequest;
    bool _connected;
    bool _ready;
    bool _starting;
    /// Why the engine is unavailable, or null while connecting or connected.
    string? _failure;
    PageKind _page = PageKind.Model;

    public EngineState? State { get; private set; }
    public Meters Meters { get; private set; } = Meters.Empty;
    public XamlRoot? XamlRoot => Content.XamlRoot;
    public nint Hwnd => WinRT.Interop.WindowNative.GetWindowHandle(this);

    public MainWindow()
    {
        InitializeComponent();
        _ui = DispatcherQueue;
        SetUpWindow();

        _modelPage = new ModelPage(this);
        _libraryPage = new LibraryPage(this);
        _audioPage = new AudioPage(this);
        _performancePage = new PerformancePage(this);
        _recordingPage = new RecordingPage(this);
        _logPage = new LogPage(SaveLog, ClearLog);
        _pages[PageKind.Model] = _modelPage;
        _pages[PageKind.Library] = _libraryPage;
        _pages[PageKind.Audio] = _audioPage;
        _pages[PageKind.Performance] = _performancePage;
        _pages[PageKind.Recording] = _recordingPage;
        _pages[PageKind.Log] = _logPage;
        _controlBar = new ControlBar(Run);
        ControlBarHost.Child = _controlBar;
        DropTitle.Text = S.DropTitle;
        DropDetail.Text = S.DropDetail;

        // Log lines arrive one event each; the text box is refreshed at most 10 times a second.
        _logFlush = _ui.CreateTimer();
        _logFlush.Interval = TimeSpan.FromMilliseconds(100);
        _logFlush.IsRepeating = false;
        _logFlush.Tick += (_, _) => _logPage.SetText(_log.ToString());

        _engine = new EngineConnection(engineEvent => _ui.TryEnqueue(() => HandleEngine(engineEvent)));
        Nav.SelectedItem = NavModel;
        ShowConnection();
        _engine.Start();
        Closed += async (_, _) => await _engine.DisposeAsync();
    }

    // -- window ---------------------------------------------------------------

    void SetUpWindow()
    {
        // Desktop acrylic: a blur of whatever is behind the window (Mica only tints by the wallpaper).
        if (DesktopAcrylicController.IsSupported()) SystemBackdrop = new DesktopAcrylicBackdrop();

        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
        AppWindow.TitleBar.PreferredHeightOption = TitleBarHeightOption.Tall;

        var icon = Path.Combine(AppContext.BaseDirectory, "Assets", "app.ico");
        if (File.Exists(icon))
        {
            AppWindow.SetIcon(icon);
            TitleBarIcon.Source = new BitmapImage(new Uri(icon));
        }

        // 960 x 760 like the Linux front end, at least 640 x 480; AppWindow works in pixels.
        double scale = GetDpiForWindow(Hwnd) / 96.0;
        AppWindow.Resize(new SizeInt32((int)(960 * scale), (int)(760 * scale)));
        if (AppWindow.Presenter is OverlappedPresenter presenter)
        {
            presenter.PreferredMinimumWidth = (int)(640 * scale);
            presenter.PreferredMinimumHeight = (int)(480 * scale);
        }
    }

    [DllImport("user32.dll")]
    static extern uint GetDpiForWindow(nint hwnd);

    void OnNavigationChanged(NavigationView sender, NavigationViewSelectionChangedEventArgs args)
    {
        if (args.SelectedItem is NavigationViewItem { Tag: string tag } && Enum.TryParse(tag, out PageKind page))
        {
            _page = page;
            ShowPage();
        }
    }

    public void ShowPage(PageKind page)
    {
        Nav.SelectedItem = Nav.MenuItems.OfType<NavigationViewItem>().First(item => (string)item.Tag == page.ToString());
    }

    void ShowPage()
    {
        var title = _page switch
        {
            PageKind.Model => S.PageModel,
            PageKind.Library => S.PageLibrary,
            PageKind.Audio => S.PageAudio,
            PageKind.Performance => S.PagePerformance,
            PageKind.Recording => S.PageRecording,
            _ => S.PageLog,
        };
        PageTitle.Text = title;
        Title = $"{title} — {S.AppTitle}";
        PageHost.Content = _pages[_page];
        if (State is { } state && _pages[_page] is PageBase page) page.Update(state);
        if (_page == PageKind.Log) _logPage.SetText(_log.ToString());
        ShowConnection();
    }

    void ShowConnection()
    {
        string? failure = _failure;
        bool ready = _ready && State is not null && failure is null;
        ConnectionView.Visibility = ready ? Visibility.Collapsed : Visibility.Visible;
        PageHost.Visibility = ready || _page == PageKind.Log ? Visibility.Visible : Visibility.Collapsed;
        if (_page == PageKind.Log) ConnectionView.Visibility = Visibility.Collapsed; // the log explains failures
        ControlBarHost.Visibility = ready ? Visibility.Visible : Visibility.Collapsed;
        ConnectionRing.Visibility = failure is null ? Visibility.Visible : Visibility.Collapsed;
        ConnectionRing.IsActive = failure is null;
        ConnectionErrorIcon.Visibility = failure is null ? Visibility.Collapsed : Visibility.Visible;
        ConnectionTitle.Text = failure is null ? S.EngineStarting : S.EngineUnavailable;
        ConnectionDetail.Text = failure is null ? S.EngineStartingDetail : S.EngineRetrying;
        ConnectionReason.Text = failure ?? "";
        ConnectionReasonCard.Visibility = failure is null ? Visibility.Collapsed : Visibility.Visible;
    }

    // -- commands -------------------------------------------------------------

    public long? Send(string cmd, JsonObject? args = null) => _engine.Send(cmd, args);

    public void SendSetting(string key, JsonNode? value) =>
        Send("update_settings", new JsonObject { ["settings"] = new JsonObject { [key] = value } });

    void Run(string function)
    {
        if (State is { Running: true } state && state.Function == function)
        {
            Send("stop");
            return;
        }
        _starting = true;
        _controlBar.Status = S.StatusPreparing;
        if (Send("start", new JsonObject { ["function"] = function }) is long id) _pending[id] = Pending.Start;
        RenderControlBar();
    }

    public void ApplyLoudness()
    {
        double adjust = Meters.Loudness?.AdjustDb ?? 0;
        if (adjust == 0 || State is not { } state) return;
        var (min, max, _) = Num.OutputGain.Range();
        double value = Math.Clamp(state.Settings.OutputGainDb + adjust, min, max);
        state.Settings.OutputGainDb = value;
        SendSetting(Num.OutputGain.Key(), value);
        // The next meters event brings the reading at the new gain; until then nothing can be applied twice.
        Meters = Meters with { Loudness = null };
        _modelPage.Update(state);
    }

    void SaveLog()
    {
        if (Send("save_log") is long id) _pending[id] = Pending.SaveLog;
    }

    void ClearLog()
    {
        _log.Clear();
        _logPage.SetText("");
        Send("clear_log");
    }

    // -- engine events --------------------------------------------------------

    void HandleEngine(EngineEvent engineEvent)
    {
        switch (engineEvent)
        {
            case EngineEvent.Connected:
                _connected = true;
                _failure = null;
                ShowConnection();
                _pending.Clear();
                _starting = false;
                break;
            case EngineEvent.Disconnected(var reason):
                _connected = false;
                _ready = false;
                State = null;
                _failure = reason;
                ShowConnection();
                break;
            case EngineEvent.Hello(var state, var initError):
                if (initError is not null)
                {
                    _failure = initError;
                    ShowConnection();
                }
                else if (state is not null) ApplyState(state, ready: true);
                // The log history arrives as a plain response.
                _logRequest = Send("get_log");
                break;
            case EngineEvent.Ready(var state):
                ApplyState(state, ready: true);
                break;
            case EngineEvent.Fatal(var message):
                _ready = false;
                _failure = message;
                ShowConnection();
                break;
            case EngineEvent.StateChanged(var state):
                ApplyState(state, ready: _ready);
                break;
            case EngineEvent.MetersChanged(var meters):
                Meters = meters;
                if (State is { } current)
                {
                    _controlBar.UpdateMeters(current, meters);
                    if (_pages[_page] is PageBase page) page.UpdateMeters(current, meters);
                }
                break;
            case EngineEvent.Status(var code, var data):
                var text = S.StatusText(code, data);
                if (text.Length > 0) _controlBar.Status = text;
                if (code is "recording_saved" or "stream_stopped" or "model_deleted" or "model_renamed" or "model_exported" or "hushwire_downloaded") Toast(text);
                break;
            case EngineEvent.Error(var error):
                var errorText = S.ErrorText(error);
                _controlBar.Status = errorText;
                Toast(errorText);
                break;
            case EngineEvent.Log(var logText):
                AppendLog(logText);
                break;
            case EngineEvent.Response response:
                HandleResponse(response);
                break;
        }
    }

    void ApplyState(EngineState state, bool ready)
    {
        bool wasReady = _ready;
        _ready = ready;
        if (!state.Running)
            Meters = Meters.Empty with { FilePosition = state.FilePlaying ? Meters.FilePosition : null };
        State = state;
        if (!wasReady && ready) ShowConnection();
        if (_pages[_page] is PageBase page) page.Update(state);
        RenderControlBar();
    }

    void RenderControlBar()
    {
        if (State is not { } state) return;
        _controlBar.Update(state, _starting);
        _controlBar.UpdateMeters(state, Meters);
    }

    void HandleResponse(EngineEvent.Response response)
    {
        if (response.Id == _logRequest)
        {
            _logRequest = null;
            if (response.Result?["text"] is JsonValue value && value.TryGetValue(out string? history))
            {
                // Keep lines that arrived as events after the request.
                var recent = _log.ToString();
                _log.Clear().Append(history);
                if (!history.EndsWith(recent, StringComparison.Ordinal)) _log.Append(recent);
                TrimLog();
                _logFlush.Start();
            }
            return;
        }
        bool known = _pending.Remove(response.Id, out var pending);
        if (known && pending == Pending.Start)
        {
            _starting = false;
            RenderControlBar();
        }
        if (response.Failure is { } error)
        {
            var text = S.ErrorText(error);
            _controlBar.Status = text;
            Toast(text);
            return;
        }
        if (!known) return; // only checked for errors
        switch (pending)
        {
            case Pending.Import:
                var names = (response.Result?["models"] as JsonArray ?? []).Select(name => name?.GetValue<string>() ?? "");
                var imported = S.ModelImported(string.Join(", ", names));
                _controlBar.Status = imported;
                Toast(imported);
                break;
            case Pending.SaveLog:
                Toast(S.LogSaved(response.Result?["path"]?.GetValue<string>() ?? ""));
                break;
        }
    }

    void AppendLog(string text)
    {
        _log.Append(text);
        TrimLog();
        if (_page == PageKind.Log && !_logFlush.IsRunning) _logFlush.Start();
    }

    void TrimLog()
    {
        if (_log.Length <= MaxLogChars) return;
        int cut = _log.Length - MaxLogChars;
        if (char.IsLowSurrogate(_log[cut])) cut++;
        _log.Remove(0, cut);
    }

    // -- toasts ---------------------------------------------------------------

    public void Toast(string text)
    {
        var bar = new InfoBar { Message = text, IsOpen = true, IsClosable = true, Severity = InfoBarSeverity.Informational };
        bar.Closed += (_, _) => Toasts.Children.Remove(bar);
        Toasts.Children.Add(bar);
        while (Toasts.Children.Count > 4) Toasts.Children.RemoveAt(0);
        var timer = _ui.CreateTimer();
        timer.Interval = TimeSpan.FromSeconds(ToastSeconds);
        timer.IsRepeating = false;
        timer.Tick += (_, _) => Toasts.Children.Remove(bar);
        timer.Start();
    }

    // -- drag and drop: voice files dropped anywhere are imported ---------------

    void OnDragOver(object sender, DragEventArgs args)
    {
        if (!args.DataView.Contains(StandardDataFormats.StorageItems)) return;
        args.AcceptedOperation = DataPackageOperation.Copy;
        args.DragUIOverride.Caption = S.DropTitle;
        DropOverlay.Visibility = Visibility.Visible;
    }

    void OnDragLeave(object sender, DragEventArgs args) => DropOverlay.Visibility = Visibility.Collapsed;

    async void OnDrop(object sender, DragEventArgs args)
    {
        DropOverlay.Visibility = Visibility.Collapsed;
        if (!args.DataView.Contains(StandardDataFormats.StorageItems)) return;
        var deferral = args.GetDeferral();
        List<string> paths;
        try
        {
            var items = await args.DataView.GetStorageItemsAsync();
            paths = items.OfType<IStorageItem>().Select(item => item.Path).Where(path => !string.IsNullOrEmpty(path)).ToList();
        }
        finally
        {
            deferral.Complete();
        }
        if (paths.Count == 0) return;
        if (!_connected || State is null)
        {
            Toast(S.ErrorText(new ErrorInfo("not_ready", "", new JsonObject())));
            return;
        }
        // Voice files are imported; anything else is not something this app takes.
        if (!paths.Any(Library.IsModelFile))
        {
            Toast(S.ErrorText(new ErrorInfo("import_no_model_file", "", new JsonObject())));
            return;
        }
        _controlBar.Status = S.StatusImportingModel;
        var list = new JsonArray(paths.Select(path => (JsonNode?)JsonValue.Create(path)).ToArray());
        if (Send("import_model", new JsonObject { ["paths"] = list }) is long id) _pending[id] = Pending.Import;
    }
}
