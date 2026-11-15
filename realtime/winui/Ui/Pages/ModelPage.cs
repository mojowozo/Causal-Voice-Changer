using System.Globalization;
using System.Text.Json.Nodes;
using CausalVC.Realtime.Engine;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CausalVC.Realtime.Ui.Pages;

/// The voice, its pitch and the levels (pages.rs: model_page).
public sealed class ModelPage : PageBase
{
    public static readonly string[] LoudnessTargets = ["voice_chat", "streaming"];
    static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;

    readonly SettingRow _modelRow;
    readonly ComboBox _model;
    readonly Action<IReadOnlyList<string>, int> _setModels;
    readonly TextBlock _modelsEmpty = Build.Text(S.ModelsEmpty);
    readonly TextBlock _files = Build.Caption("");
    readonly List<SliderSetting> _sliders = [];
    readonly SettingRow _levelRow;
    readonly TextBlock _adjust = Build.Text("");
    readonly Button _apply;
    readonly Action<IReadOnlyList<string>, int> _setTarget;
    List<string> _modelNames = [];

    public ModelPage(IHost host) : base(host)
    {
        _model = Choice(index => Host.SendSetting("model_name", _modelNames[index]), out _setModels);
        _modelRow = new SettingRow(S.Model, control: _model);
        var models = new SettingsSection(S.SectionModels)
            .Add(_modelRow)
            .Add(Build.Bar(
                _files,
                Build.Button(S.PageLibrary, (_, _) => Host.ShowPage(PageKind.Library), "\uE8F1"),
                Build.Button(S.Reload, (_, _) => Host.Send("reload_models"), "\uE72C")));

        var voice = new SettingsSection(S.SectionVoice).Add(Slider(S.Pitch, Num.Pitch));

        _apply = Build.Button(S.OutputLevelApply, (_, _) => Host.ApplyLoudness(), style: "AccentButtonStyle");
        var target = Choice(
            index =>
            {
                Host.SendSetting("loudness_target", LoudnessTargets[index]);
            },
            out _setTarget);
        target.MinWidth = 160;
        _levelRow = new SettingRow(S.OutputLevel, S.OutputLevelIdle, Build.Row(_adjust, _apply, target));
        var levels = new SettingsSection(S.SectionLevels)
            .Add(Slider(S.InputGain, Num.InputGain))
            .Add(Slider(S.OutputGain, Num.OutputGain))
            .Add(_levelRow)
            .Add(Slider(S.MonitorGain, Num.MonitorGain))
            .Add(Slider(S.NoiseGate, Num.NoiseGate));

        Column.Children.Add(models);
        Column.Children.Add(voice);
        Column.Children.Add(levels);
        Column.Children.Add(ResetRow(S.ResetModelSettings, "general"));
    }

    SettingRow Slider(string title, Num num)
    {
        var slider = new SliderSetting(title, num, SendNum);
        _sliders.Add(slider);
        return slider.Row;
    }

    public override void Update(EngineState state)
    {
        var settings = state.Settings;
        _modelNames = state.Models.Select(model => model.Name).ToList();
        bool empty = _modelNames.Count == 0;
        _modelRow.Control = empty ? _modelsEmpty : _model;
        _setModels(_modelNames, _modelNames.IndexOf(settings.ModelName));
        var selected = state.Models.FirstOrDefault(model => model.Name == settings.ModelName);
        _files.Text = selected is null ? "" : S.ModelFiles(selected.ModelFile);

        foreach (var slider in _sliders) slider.Update(slider.Num.Get(settings));
        _setTarget([S.LoudnessTargetVoiceChat, S.LoudnessTargetStreaming], Array.IndexOf(LoudnessTargets, settings.LoudnessTarget));
        UpdateMeters(state, Host.Meters);
    }

    /// How loud the output sounds to others, with a one-click gain fix.
    public override void UpdateMeters(EngineState state, Meters meters)
    {
        var reading = state.Running ? meters.Loudness : null;
        _levelRow.Description = Describe(reading);
        bool canApply = reading is { Verdict: not null } && reading.AdjustDb != 0;
        _adjust.Text = canApply ? $"{reading!.AdjustDb.ToString("+0.0;-0.0", Invariant)} dB" : "";
        _adjust.Visibility = canApply ? Visibility.Visible : Visibility.Collapsed;
        _apply.Visibility = canApply ? Visibility.Visible : Visibility.Collapsed;
    }

    static string Describe(Loudness? reading)
    {
        if (reading is null) return S.OutputLevelIdle;
        if (reading.Lufs is not double lufs) return S.OutputLevelListening;
        string level = lufs.ToString("0.0", Invariant);
        if (reading.Verdict is not string verdict)
            return S.OutputLevelMeasuring(level, reading.SpeechSeconds.ToString("0", Invariant));
        string text = verdict switch
        {
            "quiet" when reading.AdjustDb > 0 => S.OutputLevelQuiet,
            // Peaks already reach full scale: raising would squash them.
            "quiet" => S.OutputLevelQuietPeaky,
            "loud" when reading.OverPercent > 1 => S.OutputLevelSquashed,
            "loud" => S.OutputLevelLoud,
            _ => S.OutputLevelGood,
        };
        return S.OutputLevelReading(level, reading.TargetLufs.ToString("0", Invariant), text);
    }
}
