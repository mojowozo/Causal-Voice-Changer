using System.Text.Json.Nodes;
using CausalVC.Realtime.Engine;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Input;
using Windows.Storage.Pickers;

namespace CausalVC.Realtime.Ui.Pages;

/// Input source, devices and file playback (pages.rs: audio_page).
public sealed class AudioPage : PageBase
{
    public static readonly string[] InputSources = ["microphone", "file"];

    readonly SettingsSection _input = new(S.SectionInput);
    readonly SettingRow _sourceRow;
    readonly Action<IReadOnlyList<string>, int> _setSource;
    readonly TextBlock _ffmpegMissing = Build.Text("");
    readonly SettingRow _fileRow;
    readonly TextBlock _fileName = Build.Text(S.NoFile);
    readonly Grid _player = new() { ColumnSpacing = 8 };
    readonly Button _playPause;
    readonly Button _stop;
    readonly Slider _seek = new() { Minimum = 0, Maximum = 1, StepFrequency = 0.1, IsThumbToolTipEnabled = false };
    readonly TextBlock _time = Build.Text("00:00 / 00:00");
    readonly SliderSetting _fileVolume;
    readonly SettingRow _inputRow;
    readonly Action<IReadOnlyList<string>, int> _setInput;
    readonly Action<IReadOnlyList<string>, int> _setOutput;
    readonly Action<IReadOnlyList<string>, int> _setMonitor;
    readonly TextBlock _rate = Build.Caption("");
    List<string> _inputs = [];
    List<string> _outputs = [];
    string _layout = "";
    bool _seeking;
    bool _updatingSeek;

    public AudioPage(IHost host) : base(host)
    {
        _sourceRow = new SettingRow(S.InputSource, S.InputSourceDetail,
            Choice(index => Host.SendSetting("input_source", InputSources[index]), out _setSource));
        _fileRow = new SettingRow(S.AudioFile, control: Build.Row(_fileName, Build.Button(S.Browse, async (_, _) => await PickFile())));

        _playPause = Build.IconButton("\uE768", "", (_, _) =>
            Host.Send(Host.State?.FilePlaying == true ? "file_pause" : "file_play"));
        _stop = Build.IconButton("\uE71A", "", (_, _) => Host.Send("file_stop"));
        ToolTipService.SetToolTip(_playPause, null);
        ToolTipService.SetToolTip(_stop, null);
        foreach (var column in new[] { GridLength.Auto, GridLength.Auto, new GridLength(1, GridUnitType.Star), GridLength.Auto })
            _player.ColumnDefinitions.Add(new ColumnDefinition { Width = column });
        _player.Children.Add(_playPause);
        Grid.SetColumn(_stop, 1);
        _player.Children.Add(_stop);
        _seek.VerticalAlignment = VerticalAlignment.Center;
        Grid.SetColumn(_seek, 2);
        _player.Children.Add(_seek);
        _time.VerticalAlignment = VerticalAlignment.Center;
        Grid.SetColumn(_time, 3);
        _player.Children.Add(_time);
        _seek.AddHandler(PointerPressedEvent, new PointerEventHandler((_, _) => _seeking = true), true);
        _seek.AddHandler(PointerCaptureLostEvent, new PointerEventHandler((_, _) => EndSeek()), true);
        _seek.AddHandler(PointerReleasedEvent, new PointerEventHandler((_, _) => EndSeek()), true);
        _seek.ValueChanged += OnSeekChanged;
        _fileVolume = new SliderSetting(S.FileVolume, Num.FileVolume, SendNum);

        _inputRow = new SettingRow(S.InputDevice, S.InputDeviceDetail,
            Choice(index => Host.SendSetting("input_device", _inputs[index]), out _setInput));
        var output = new SettingsSection(S.SectionOutput)
            .Add(new SettingRow(S.OutputDevice, S.OutputDeviceDetail,
                Choice(index => Host.SendSetting("output_device", _outputs[index]), out _setOutput)))
            .Add(new SettingRow(S.MonitorDevice, S.MonitorDeviceDetail,
                // Index 0 is "Disabled"; the rest mirror the output list.
                Choice(index => Host.SendSetting("monitor_device", index == 0 ? null : _outputs[index - 1]), out _setMonitor)));

        Column.Children.Add(_input);
        Column.Children.Add(output);
        Column.Children.Add(Build.Bar(_rate, Build.Button(S.ReloadDevices, (_, _) => Host.Send("reload_devices"), "\uE72C")));
    }

    public override void Update(EngineState state)
    {
        var settings = state.Settings;
        bool fileSource = settings.InputSource == "file";
        string layout = $"{fileSource}/{state.Ffmpeg}";
        if (layout != _layout)
        {
            // Rows come and go with the source; the section is rebuilt so dividers stay right.
            _layout = layout;
            _input.Clear();
            _input.Add(_sourceRow);
            if (fileSource && !state.Ffmpeg)
            {
                _ffmpegMissing.Text = S.ErrorText(new ErrorInfo("ffmpeg_missing", "", new JsonObject()));
                _input.Add(_ffmpegMissing);
            }
            if (fileSource)
            {
                _input.Add(_fileRow);
                _input.Add(_player);
                _input.Add(_fileVolume.Row);
            }
            else
            {
                _input.Add(_inputRow);
            }
        }
        _setSource([S.SourceMicrophone, S.SourceFile], Array.IndexOf(InputSources, settings.InputSource));
        _fileName.Text = state.FilePath is null ? S.NoFile : Path.GetFileName(state.FilePath);
        _fileVolume.Update(settings.FileInputVolume);

        _inputs = state.Inputs;
        _outputs = state.Outputs;
        _setInput(_inputs, _inputs.IndexOf(settings.InputDevice));
        _setOutput(_outputs, _outputs.IndexOf(settings.OutputDevice));
        int monitor = settings.MonitorDevice is null ? -1 : _outputs.IndexOf(settings.MonitorDevice);
        // Index 0 is "Disabled"; an output that is gone shows nothing selected.
        _setMonitor([S.MonitorDisabled, .. _outputs],
            settings.MonitorDevice is null ? 0 : monitor >= 0 ? monitor + 1 : -1);

        _rate.Text = (state.SampleRate is int rate ? S.SampleRateValue(rate) : S.SampleRateIdle) + "\n" + S.DeviceHint;
        UpdateMeters(state, Host.Meters);
    }

    public override void UpdateMeters(EngineState state, Meters meters)
    {
        bool hasFile = state.FilePath is not null;
        ((FontIcon)_playPause.Content).Glyph = state.FilePlaying ? "\uE769" : "\uE768";
        _playPause.IsEnabled = state.FilePlaying || hasFile;
        _stop.IsEnabled = hasFile;
        double duration = Math.Max(state.FileDuration ?? 0, 1);
        if (!_seeking)
        {
            _updatingSeek = true;
            _seek.Maximum = duration;
            _seek.Value = Math.Min(meters.FilePosition ?? 0, duration);
            _updatingSeek = false;
        }
        _time.Text = $"{MmSs(_seek.Value)} / {MmSs(state.FileDuration ?? 0)}";
    }

    void OnSeekChanged(object sender, RangeBaseValueChangedEventArgs args)
    {
        if (_updatingSeek) return;
        _time.Text = $"{MmSs(args.NewValue)} / {MmSs(Host.State?.FileDuration ?? 0)}";
        if (!_seeking) Host.Send("file_seek", new JsonObject { ["seconds"] = args.NewValue }); // keyboard
    }

    void EndSeek()
    {
        if (!_seeking) return;
        _seeking = false;
        Host.Send("file_seek", new JsonObject { ["seconds"] = _seek.Value });
    }

    public static string MmSs(double seconds)
    {
        long total = (long)Math.Max(seconds, 0);
        return $"{total / 60:00}:{total % 60:00}";
    }

    async Task PickFile()
    {
        var picker = new FileOpenPicker { SuggestedStartLocation = PickerLocationId.MusicLibrary };
        foreach (var type in new[] { ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac", ".mp4", ".mkv", "*" })
            picker.FileTypeFilter.Add(type);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, Host.Hwnd);
        var file = await picker.PickSingleFileAsync();
        if (file is not null) Host.Send("file_select", new JsonObject { ["path"] = file.Path });
    }
}
