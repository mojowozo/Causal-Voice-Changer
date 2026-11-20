// The window and the app's state, as in ../ui/src/app.rs: trainer events come in, the pages
// re-render from the latest state, and the buttons go out as trainer commands.

using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using CausalVC.Training.Trainer;
using CausalVC.Training.Ui;
using CausalVC.Training.Ui.Pages;
using Microsoft.UI.Composition.SystemBackdrops;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;
using Windows.Graphics;

namespace CausalVC.Training;

public sealed partial class MainWindow : Window, IHost
{
    const int MaxLogChars = 60_000;
    const int ToastSeconds = 6;

    readonly DispatcherQueue _ui;
    readonly TrainerConnection _trainer;
    readonly VoicePage _voicePage;
    readonly BuildsPage _buildsPage;
    readonly LogPage _logPage;
    readonly AboutPage _aboutPage;
    readonly Updates _updates;
    readonly ControlBar _controlBar;
    readonly System.Text.StringBuilder _log = new();
    readonly DispatcherQueueTimer _logFlush;
    bool _ready;
    /// Why the trainer is unavailable, or null while it starts or runs.
    string? _failure;
    PageKind _page = PageKind.Voice;

    public TrainerState State { get; private set; } = new();
    public XamlRoot? XamlRoot => Content.XamlRoot;
    public nint Hwnd => WinRT.Interop.WindowNative.GetWindowHandle(this);

    public MainWindow()
    {
        InitializeComponent();
        _ui = DispatcherQueue;
        SetUpWindow();

        _voicePage = new VoicePage(this);
        _buildsPage = new BuildsPage(this);
        _logPage = new LogPage(ClearLog);
        // The installer waits for this window to close (the trainer ends with it).
        _updates = new Updates((cmd, args) => { Send(cmd, args); return null; }, () => XamlRoot, Toast, Close);
        _aboutPage = new AboutPage(this, _updates.Section);
        _controlBar = new ControlBar(
            start: () => Send("start", _voicePage.Config()),
            queue: () => Send("queue", _voicePage.Config()),
            stop: () => Send("stop"));
        ControlBarHost.Child = _controlBar;
        _controlBar.Update(State);

        // Log lines arrive one event each; the text box is refreshed at most 10 times a second.
        _logFlush = _ui.CreateTimer();
        _logFlush.Interval = TimeSpan.FromMilliseconds(100);
        _logFlush.IsRepeating = false;
        _logFlush.Tick += (_, _) => _logPage.SetText(_log.ToString());

        _trainer = new TrainerConnection(trainerEvent => _ui.TryEnqueue(() => HandleTrainer(trainerEvent)));
        Nav.SelectedItem = NavVoice;
        ShowConnection();
        _trainer.Start();
        // Closing the trainer's stdin makes it save and stop a running build.
        Closed += async (_, _) => await _trainer.DisposeAsync();
    }

    // -- window ---------------------------------------------------------------

    void SetUpWindow()
    {
        // Desktop acrylic, as in the voice changer: a blur of whatever is behind the window.
        if (DesktopAcrylicController.IsSupported()) SystemBackdrop = new DesktopAcrylicBackdrop();
        else if (MicaController.IsSupported()) SystemBackdrop = new MicaBackdrop();

        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
        AppWindow.TitleBar.PreferredHeightOption = TitleBarHeightOption.Tall;

        var icon = Path.Combine(AppContext.BaseDirectory, "Assets", "app.ico");
        if (File.Exists(icon))
        {
            AppWindow.SetIcon(icon);
            TitleBarIcon.Source = new BitmapImage(new Uri(icon));
        }

        // 900 x 780 like the Linux front end, at least 600 x 480; AppWindow works in pixels.
        double scale = GetDpiForWindow(Hwnd) / 96.0;
        AppWindow.Resize(new SizeInt32((int)(900 * scale), (int)(780 * scale)));
        if (AppWindow.Presenter is OverlappedPresenter presenter)
        {
            presenter.PreferredMinimumWidth = (int)(600 * scale);
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

    void ShowPage()
    {
        var title = _page switch
        {
            PageKind.Voice => S.PageVoice,
            PageKind.Builds => S.PageBuilds,
            PageKind.About => S.PageAbout,
            _ => S.PageLog,
        };
        PageTitle.Text = title;
        Title = $"{title} — {S.AppTitle}";
        PageHost.Content = _page switch
        {
            PageKind.Voice => _voicePage,
            PageKind.Builds => _buildsPage,
            PageKind.About => _aboutPage,
            _ => _logPage,
        };
        if (_page == PageKind.Builds) _buildsPage.Update(State);
        if (_page == PageKind.Log) _logPage.SetText(_log.ToString());
        ShowConnection();
    }

    void ShowConnection()
    {
        string? failure = _failure;
        bool ready = _ready && failure is null;
        // The log stays readable while the trainer is down: it often says why.
        bool showPage = ready || _page == PageKind.Log;
        ConnectionView.Visibility = showPage ? Visibility.Collapsed : Visibility.Visible;
        PageHost.Visibility = showPage ? Visibility.Visible : Visibility.Collapsed;
        ControlBarHost.Visibility = ready ? Visibility.Visible : Visibility.Collapsed;
        ConnectionRing.Visibility = failure is null ? Visibility.Visible : Visibility.Collapsed;
        ConnectionRing.IsActive = failure is null;
        ConnectionErrorIcon.Visibility = failure is null ? Visibility.Collapsed : Visibility.Visible;
        ConnectionTitle.Text = failure is null ? S.ServiceStarting : S.ServiceUnavailable;
        ConnectionDetail.Text = failure is null ? S.ServiceStartingDetail : S.ServiceRetrying;
        ConnectionReason.Text = failure ?? "";
        ConnectionReasonCard.Visibility = failure is null ? Visibility.Collapsed : Visibility.Visible;
    }

    // -- trainer --------------------------------------------------------------

    public void Send(string cmd, JsonObject? args = null) => _trainer.Send(cmd, args);

    void HandleTrainer(TrainerEvent trainerEvent)
    {
        switch (trainerEvent)
        {
            case TrainerEvent.Started:
                _failure = null;
                ShowConnection();
                break;
            case TrainerEvent.Ended ended:
                _ready = false;
                _failure = ended.Reason;
                ShowConnection();
                break;
            case TrainerEvent.Hello hello:
                _voicePage.ApplyForm(hello.Form, hello.Bases);
                if (hello.Update is { } update) _updates.Apply(update);
                _ready = true;
                ShowConnection();
                break;
            case TrainerEvent.Gpus gpus:
                _voicePage.SetGpus(gpus.List);
                break;
            case TrainerEvent.StateChanged changed:
                bool finished = State.Running && !changed.State.Running;
                State = changed.State;
                _controlBar.Update(State);
                if (_page == PageKind.Builds) _buildsPage.Update(State);
                if (finished) Toast(S.Phase(State));
                break;
            case TrainerEvent.UpdateChanged changed:
                _updates.Apply(changed.Info);
                break;
            case TrainerEvent.Queued queued:
                Toast(S.BuildQueued(queued.Name, queued.Position));
                break;
            case TrainerEvent.Log log:
                AppendLog(log.Text);
                break;
            case TrainerEvent.Error error:
                var text = S.Error(error.Code, error.Message);
                if (error.Field is { } field) _voicePage.ShowError(field, text);
                Toast(text);
                break;
        }
    }

    void AppendLog(string text)
    {
        _log.Append(text).Append('\n');
        if (_log.Length > MaxLogChars) _log.Remove(0, _log.Length - MaxLogChars);
        if (_page == PageKind.Log && !_logFlush.IsRunning) _logFlush.Start();
    }

    void ClearLog()
    {
        _log.Clear();
        _logPage.SetText("");
    }

    public void Toast(string text)
    {
        var bar = new InfoBar { Message = text, IsOpen = true, IsClosable = true, Severity = InfoBarSeverity.Informational };
        // An InfoBar's own background is a light tint meant for a solid page: floating over the
        // page, the text below shows through it. A solid card under it keeps the toast readable.
        var card = new Border
        {
            Child = bar,
            CornerRadius = new CornerRadius(4),
            Background = (Brush)Application.Current.Resources["SolidBackgroundFillColorBaseBrush"],
            Translation = new System.Numerics.Vector3(0, 0, 16),
            Shadow = new ThemeShadow(),
        };
        bar.Closed += (_, _) => Toasts.Children.Remove(card);
        Toasts.Children.Add(card);
        while (Toasts.Children.Count > 4) Toasts.Children.RemoveAt(0);
        var timer = _ui.CreateTimer();
        timer.Interval = TimeSpan.FromSeconds(ToastSeconds);
        timer.IsRepeating = false;
        timer.Tick += (_, _) => Toasts.Children.Remove(card);
        timer.Start();
    }
}
