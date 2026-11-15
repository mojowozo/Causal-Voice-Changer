using System.Diagnostics;
using CausalVC.Realtime.Engine;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Windows.Storage.Pickers;

namespace CausalVC.Realtime.Ui.Pages;

/// Recording mode, folder and the record button (pages.rs: recording_page).
public sealed class RecordingPage : PageBase
{
    public static readonly string[] RecordingModes = ["separate", "mix", "stereo"];

    readonly Action<IReadOnlyList<string>, int> _setMode;
    readonly TextBlock _folder = Build.Caption("");
    readonly TextBlock _elapsed = Build.Text("");
    readonly Button _record;
    readonly Button _stopRecording;

    public RecordingPage(IHost host) : base(host)
    {
        var section = new SettingsSection(S.SectionRecording)
            .Add(new SettingRow(S.RecordingMode, S.RecordingModeDetail,
                Choice(index => Host.SendSetting("recording_mode", RecordingModes[index]), out _setMode)))
            .Add(new SettingRow(S.SaveFolder, control: Build.Row(
                _folder,
                Build.Button(S.Change, async (_, _) => await PickFolder()),
                Build.Button(S.OpenFolder, (_, _) => OpenFolder()))));
        _folder.MaxWidth = 320;
        _folder.VerticalAlignment = VerticalAlignment.Center;
        _record = Build.Button(S.Record, (_, _) => Host.Send("record_start"), "\uE7C8", "AccentButtonStyle");
        // record_stop answers with the saved paths; the recording_saved status shows them.
        _stopRecording = Build.CriticalButton(S.StopRecording, (_, _) => Host.Send("record_stop"), "\uE71A");

        Column.Children.Add(section);
        Column.Children.Add(Build.Bar(_elapsed, _record, _stopRecording));
    }

    public override void Update(EngineState state)
    {
        _setMode([S.RecordingSeparate, S.RecordingMix, S.RecordingStereo], Array.IndexOf(RecordingModes, state.Settings.RecordingMode));
        _folder.Text = state.Settings.RecordingFolder;
        _record.Visibility = state.Recording ? Visibility.Collapsed : Visibility.Visible;
        _stopRecording.Visibility = state.Recording ? Visibility.Visible : Visibility.Collapsed;
        _record.IsEnabled = state.Running;
        UpdateMeters(state, Host.Meters);
    }

    public override void UpdateMeters(EngineState state, Meters meters)
    {
        if (state.Recording && meters.RecordingSeconds is double seconds)
        {
            long total = (long)seconds;
            _elapsed.Text = $"● REC {total / 3600:00}:{total / 60 % 60:00}:{total % 60:00}";
        }
        else
        {
            _elapsed.Text = state.Running ? "" : S.RecordHint;
        }
    }

    async Task PickFolder()
    {
        var picker = new FolderPicker();
        picker.FileTypeFilter.Add("*");
        WinRT.Interop.InitializeWithWindow.Initialize(picker, Host.Hwnd);
        var folder = await picker.PickSingleFolderAsync();
        if (folder is not null) Host.SendSetting("recording_folder", folder.Path);
    }

    void OpenFolder()
    {
        if (Host.State is not { } state) return;
        try
        {
            Directory.CreateDirectory(state.Settings.RecordingFolder);
            Process.Start(new ProcessStartInfo("explorer.exe") { ArgumentList = { state.Settings.RecordingFolder } });
        }
        catch (Exception error)
        {
            Host.Toast(error.Message);
        }
    }
}
