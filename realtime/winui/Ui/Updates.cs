// New releases: the update window and the update settings, as in ../ui/src/update.rs.
// The same file in the realtime app (realtime/winui/Ui/Updates.cs) and the training app
// (training/winui/Ui/Updates.cs) apart from the namespace: keep both copies equal.
//
// The Python side (causalvc/update.py, behind the engine or the trainer) checks GitHub, downloads
// and installs; it reports every change as an "update" event. This shows that and sends commands.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CausalVC.Realtime.Ui;

/// The backend's update state (see causalvc.update.Updater).
public sealed record UpdateInfo(
    string State, string Current, bool Git, bool CheckAtLaunch, string Version, string Notes,
    string Url, double Progress, string Error, bool Manual)
{
    public static UpdateInfo From(JsonObject json) => new(
        Text(json, "state"), Text(json, "current"), Flag(json, "git"), Flag(json, "check_at_launch"),
        Text(json, "version"), Text(json, "notes"), Text(json, "url"),
        json["progress"] is JsonValue p && p.TryGetValue(out double progress) ? progress : 0,
        Text(json, "error"), Flag(json, "manual"));

    static string Text(JsonObject json, string key) =>
        json[key] is JsonValue value && value.TryGetValue(out string? text) ? text : "";

    static bool Flag(JsonObject json, string key) =>
        json[key] is JsonValue value && value.TryGetValue(out bool flag) && flag;
}

/// English text, from ../ui/i18n/en/*.ftl.
static class UpdateText
{
    public const string SectionUpdates = "Updates";
    public const string Version = "Version";
    public const string CheckAtLaunch = "Check for updates at launch";
    public const string CheckAtLaunchDetail = "Look for a new release on GitHub each time the app starts.";
    public const string Check = "Check for updates";
    public const string CheckButton = "Check now";
    public const string StatusUnknown = "Available once the app has finished starting.";
    public const string StatusIdle = "Not checked yet.";
    public const string StatusChecking = "Checking…";
    public static string StatusAvailable(string version) => $"{version} is available.";
    public static string StatusDownloading(int percent) => $"Downloading… {percent}%";
    public static string None(string version) => $"You have the latest version ({version}).";
    public static string AvailableTitle(string version) => $"{version} is available";
    public static string AvailableBody(string current) =>
        $"You have {current}. Update downloads the new release, closes the app, installs it and opens the app again. Your voices, builds and settings are kept.";
    public static string AvailableGit(string current) =>
        $"You have {current}. This copy is a git checkout: update it with git pull.";
    public const string Now = "Update";
    public const string OpenPage = "Open release page";
    public const string Later = "Ask me next time";
    public const string Skip = "Skip this version";
    public static string Downloading(string version) => $"Updating to {version}";
    public const string DownloadingBody = "Downloading the new release…";
    public const string Restarting = "Installing. The app will close and open again.";
    public const string Hide = "Hide";
    public const string FailedTitle = "The update did not finish";
    public const string TryAgain = "Try again";
    public const string Close = "Close";
}

/// Shows the backend's update state: the window when an update is found, its progress, and the
/// settings section. `send` sends a command, `root` gives the window's XamlRoot once it has one,
/// and `exit` closes the window (the installer waits for it).
public sealed class Updates
{
    readonly Func<string, JsonObject?, long?> _send;
    readonly Func<XamlRoot?> _root;
    readonly Action<string> _toast;
    readonly Action _exit;
    readonly DispatcherQueue _ui = DispatcherQueue.GetForCurrentThread();
    readonly SettingRow _versionRow;
    readonly ToggleSwitch _checkAtLaunch = new() { OnContent = "", OffContent = "", MinWidth = 0 };
    readonly SettingRow _checkRow;
    readonly Button _checkButton;
    readonly HyperlinkButton _pageButton = new() { Content = UpdateText.OpenPage, Visibility = Visibility.Collapsed };
    ContentDialog? _dialog;
    ProgressBar? _progress;
    TextBlock? _progressText;
    bool _updatingToggle;

    public UpdateInfo? Info { get; private set; }

    /// The settings: this version, checking at launch, and checking now.
    public SettingsSection Section { get; }

    public Updates(Func<string, JsonObject?, long?> send, Func<XamlRoot?> root, Action<string> toast, Action exit)
    {
        _send = send;
        _root = root;
        _toast = toast;
        _exit = exit;
        _checkAtLaunch.Toggled += (_, _) =>
        {
            if (!_updatingToggle) _send("set_update_check", new JsonObject { ["on"] = _checkAtLaunch.IsOn });
        };
        _checkButton = Build.Button(UpdateText.CheckButton, (_, _) => _send("check_update", new JsonObject { ["manual"] = true }));
        _pageButton.Click += (_, _) => Open(Info?.Url);
        _versionRow = new SettingRow(UpdateText.Version, null, Build.Text("—"));
        _checkRow = new SettingRow(UpdateText.Check, UpdateText.StatusUnknown,
            new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8, Children = { _pageButton, _checkButton } });
        Section = new SettingsSection(UpdateText.SectionUpdates)
            .Add(_versionRow)
            .Add(new SettingRow(UpdateText.CheckAtLaunch, UpdateText.CheckAtLaunchDetail, _checkAtLaunch))
            .Add(_checkRow);
        Render();
    }

    /// A new state from the backend (in its hello, or an "update" event).
    public void Apply(UpdateInfo info)
    {
        bool changed = Info?.State != info.State;
        Info = info;
        Render();
        if (!changed)
        {
            if (info.State == "downloading") ShowProgress(info);
            return;
        }
        switch (info.State)
        {
            case "available":
                ShowWhenReady(ShowAvailable);
                break;
            case "failed":
                ShowWhenReady(ShowFailed);
                break;
            case "downloading":
                ShowProgress(info);
                break;
            case "ready":
                ShowProgress(info);
                _send("install_update", new JsonObject { ["pid"] = Environment.ProcessId });
                break;
            case "installing":
                _exit();
                break;
            case "none" when info.Manual:
                _toast(UpdateText.None(info.Current));
                break;
            case "error":
                _dialog?.Hide();
                _toast(info.Error);
                break;
        }
    }

    void Render()
    {
        var info = Info;
        _versionRow.Control = Build.Text(info?.Current is { Length: > 0 } current ? current : "—");
        _updatingToggle = true;
        _checkAtLaunch.IsOn = info?.CheckAtLaunch ?? true;
        _checkAtLaunch.IsEnabled = info is not null;
        _updatingToggle = false;
        bool busy = info?.State is "checking" or "downloading" or "ready" or "installing";
        _checkButton.IsEnabled = info is not null && !busy;
        _pageButton.Visibility = string.IsNullOrEmpty(info?.Url) ? Visibility.Collapsed : Visibility.Visible;
        _checkRow.Description = info switch
        {
            null => UpdateText.StatusUnknown,
            { State: "checking" } => UpdateText.StatusChecking,
            { State: "available" } a => UpdateText.StatusAvailable(a.Version),
            { State: "downloading" } d => UpdateText.StatusDownloading((int)Math.Round(d.Progress * 100)),
            { State: "none" } n => UpdateText.None(n.Current),
            { State: "error" or "failed" } e => e.Error,
            _ => UpdateText.StatusIdle,
        };
    }

    /// A ContentDialog needs the window's XamlRoot, which it has only once its content has loaded.
    void ShowWhenReady(Func<XamlRoot, Task> show)
    {
        if (_root() is { } root)
        {
            _ = Show(show, root);
            return;
        }
        var timer = _ui.CreateTimer();
        timer.Interval = TimeSpan.FromMilliseconds(300);
        timer.Tick += (_, _) =>
        {
            if (_root() is not { } ready) return;
            timer.Stop();
            _ = Show(show, ready);
        };
        timer.Start();
    }

    async Task Show(Func<XamlRoot, Task> show, XamlRoot root)
    {
        if (_dialog is not null) return;
        try
        {
            await show(root);
        }
        catch (Exception error) when (error is InvalidOperationException or System.Runtime.InteropServices.COMException)
        {
            // Another dialog is open (only one can be): the About page still shows the update.
        }
    }

    async Task ShowAvailable(XamlRoot root)
    {
        if (Info is not { State: "available" } info) return;
        var content = new StackPanel { Spacing = 12, MaxWidth = 480 };
        content.Children.Add(Build.Text(info.Git ? UpdateText.AvailableGit(info.Current) : UpdateText.AvailableBody(info.Current)));
        if (info.Notes.Length > 0)
            content.Children.Add(new ScrollViewer
            {
                MaxHeight = 240,
                Content = new TextBlock { Text = info.Notes, TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true },
            });
        _dialog = new ContentDialog
        {
            XamlRoot = root,
            Title = UpdateText.AvailableTitle(info.Version),
            Content = content,
            PrimaryButtonText = info.Git ? UpdateText.OpenPage : UpdateText.Now,
            SecondaryButtonText = UpdateText.Skip,
            CloseButtonText = UpdateText.Later,
            DefaultButton = ContentDialogButton.Primary,
        };
        var result = await _dialog.ShowAsync();
        _dialog = null;
        switch (result)
        {
            case ContentDialogResult.Primary when info.Git:
                Open(info.Url);
                break;
            case ContentDialogResult.Primary:
                _send("download_update", null);
                break;
            case ContentDialogResult.Secondary:
                _send("skip_update", new JsonObject { ["version"] = info.Version });
                break;
            default:
                _send("later_update", null);
                break;
        }
    }

    void ShowProgress(UpdateInfo info)
    {
        if (_progress is not null)
        {
            _progress.Value = info.Progress * 100;
            if (_progressText is not null)
                _progressText.Text = info.State == "downloading" ? UpdateText.DownloadingBody : UpdateText.Restarting;
            return;
        }
        if (_dialog is not null) return; // opened from the update window: shown once that has closed
        ShowWhenReady(async root =>
        {
            _progress = new ProgressBar { Minimum = 0, Maximum = 100, Value = info.Progress * 100 };
            _progressText = Build.Text(UpdateText.DownloadingBody);
            _dialog = new ContentDialog
            {
                XamlRoot = root,
                Title = UpdateText.Downloading(info.Version),
                Content = new StackPanel { Spacing = 12, MinWidth = 360, Children = { _progressText, _progress } },
                CloseButtonText = UpdateText.Hide,
            };
            await _dialog.ShowAsync();
            _dialog = null;
            _progress = null;
            _progressText = null;
        });
    }

    async Task ShowFailed(XamlRoot root)
    {
        if (Info is not { State: "failed" } info) return;
        _dialog = new ContentDialog
        {
            XamlRoot = root,
            Title = UpdateText.FailedTitle,
            Content = Build.Text(info.Error),
            PrimaryButtonText = UpdateText.TryAgain,
            CloseButtonText = UpdateText.Close,
            DefaultButton = ContentDialogButton.Primary,
        };
        var result = await _dialog.ShowAsync();
        _dialog = null;
        if (result == ContentDialogResult.Primary) _send("check_update", new JsonObject { ["manual"] = true });
    }

    void Open(string? url)
    {
        if (string.IsNullOrEmpty(url)) return;
        try
        {
            Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
        }
        catch (Exception error)
        {
            _toast(error.Message);
        }
    }
}
