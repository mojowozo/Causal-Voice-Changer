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
    // The selected voice's checkpoints, shown only when it has more than one.
    readonly SettingsSection _models = new(S.SectionModels);
    readonly SettingRow _checkpointRow;
    readonly Action<IReadOnlyList<string>, int> _setCheckpoints;
    readonly UIElement _modelBar;
    List<string> _voiceCheckpoints = [];
    List<string> _checkpointNames = [];
    bool? _checkpointShown;
    readonly TextBlock _modelsEmpty = Build.Text(S.ModelsEmpty);
    readonly TextBlock _files = Build.Caption("");
    readonly List<SliderSetting> _sliders = [];
    readonly SettingRow _levelRow;
    readonly TextBlock _adjust = Build.Text("");
    readonly Button _apply;
    readonly Action<IReadOnlyList<string>, int> _setTarget;
    List<string> _modelNames = [];
    // The Voice section: the pitch, and behind a disclosure row the pitch offset, floor and ceiling, which few voices need.
    readonly SettingsSection _voice = new(S.SectionVoice);
    readonly SettingRow _pitchRow;
    readonly SettingRow _advancedRow;
    readonly SettingRow _offsetRow;
    readonly SettingRow _floorRow;
    readonly SettingRow _floorKeepRow;
    readonly SettingRow _ceilingRow;
    readonly SettingRow _keepRow;
    // The Index slider, or for a voice without an index a note saying so.
    readonly SettingRow _indexRow;
    readonly SettingRow _noIndexRow = new(S.IndexRate, S.IndexRateNone, Build.Text(S.Off));
    bool _hasIndex;
    readonly Button _advancedToggle;
    bool _advancedOpen;

    public ModelPage(IHost host) : base(host)
    {
        // The selector lists each voice once; picking one loads its picked (else newest) checkpoint.
        _model = Choice(index => Host.SendSetting("model_name", _voiceCheckpoints[index]), out _setModels);
        _modelRow = new SettingRow(S.Model, control: _model);
        var checkpoint = Choice(index => Host.SendSetting("model_name", _checkpointNames[index]), out _setCheckpoints);
        _checkpointRow = new SettingRow(S.Checkpoint, S.CheckpointDescription, checkpoint);
        _modelBar = Build.Bar(
            _files,
            Build.Button(S.PageLibrary, (_, _) => Host.ShowPage(PageKind.Library), "\uE8F1"),
            Build.Button(S.Reload, (_, _) => Host.Send("reload_models"), "\uE72C"));
        RenderModels(false);

        _pitchRow = Slider(S.Pitch, Num.Pitch);
        _offsetRow = Slider(S.PitchOffset, Num.PitchOffset);
        _floorRow = Slider(S.PitchFloor, Num.PitchFloor);
        _floorKeepRow = Slider(S.PitchFloorKeep, Num.PitchFloorKeep);
        _ceilingRow = Slider(S.PitchCeiling, Num.PitchCeiling);
        _keepRow = Slider(S.PitchCeilingKeep, Num.PitchCeilingKeep);
        _indexRow = Slider(S.IndexRate, Num.IndexRate);
        _advancedToggle = Build.IconButton("\uE70D", S.PitchAdvancedShow, (_, _) =>
        {
            _advancedOpen = !_advancedOpen;
            RenderVoice();
        });
        _advancedRow = new SettingRow(S.PitchAdvanced, S.PitchAdvancedDetail(NumInfo.Hertz(0), NumInfo.Ceiling(0), NumInfo.Ceiling(0)), _advancedToggle);
        RenderVoice();

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
            .Add(Slider(S.MonitorGain, Num.MonitorGain));

        Column.Children.Add(_models);
        Column.Children.Add(_voice);
        Column.Children.Add(levels);
        Column.Children.Add(ResetRow(S.ResetModelSettings, "general"));
    }

    void RenderModels(bool showCheckpoint)
    {
        if (_checkpointShown == showCheckpoint) return;
        _checkpointShown = showCheckpoint;
        _models.Clear();
        _models.Add(_modelRow);
        if (showCheckpoint) _models.Add(_checkpointRow);
        _models.Add(_modelBar);
    }

    void RenderVoice()
    {
        _voice.Clear();
        _voice.Add(_pitchRow).Add(_hasIndex ? _indexRow : _noIndexRow).Add(_advancedRow);
        if (_advancedOpen) _voice.Add(_offsetRow).Add(_floorRow).Add(_floorKeepRow).Add(_ceilingRow).Add(_keepRow);
        ((FontIcon)_advancedToggle.Content).Glyph = _advancedOpen ? "\uE70E" : "\uE70D";  // chevron up / down
        ToolTipService.SetToolTip(_advancedToggle, _advancedOpen ? S.PitchAdvancedHide : S.PitchAdvancedShow);
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
        // An older engine sends no voices: every model is a voice of its own.
        var voices = state.Voices.Count > 0 || state.Models.Count == 0
            ? state.Voices
            : state.Models.Select(model => new Voice(model.Name, [model.Name], model.Name)).ToList();
        var selectedVoice = voices.FirstOrDefault(voice => voice.Checkpoints.Contains(settings.ModelName));
        _modelNames = voices.Select(voice => voice.Name).ToList();
        _voiceCheckpoints = voices.Select(voice => voice.Checkpoint).ToList();
        bool empty = state.Models.Count == 0;
        _modelRow.Control = empty ? _modelsEmpty : _model;
        _setModels(_modelNames, selectedVoice is null ? -1 : _modelNames.IndexOf(selectedVoice.Name));
        _checkpointNames = selectedVoice?.Checkpoints ?? [];
        var checkpoints = _checkpointNames
            .Select(name => state.Models.FirstOrDefault(model => model.Name == name))
            .ToList();
        string? newest = checkpoints.Where(model => model?.Step is not null).MaxBy(model => model!.Step)?.Name;
        _setCheckpoints(
            _checkpointNames.Zip(checkpoints, (name, model) =>
            {
                string label = model is null ? name : Library.CheckpointLabel(model);
                return name == newest ? S.CheckpointNewest(label) : label;
            }).ToList(),
            _checkpointNames.IndexOf(settings.ModelName));
        RenderModels(_checkpointNames.Count > 1);
        var selected = state.Models.FirstOrDefault(model => model.Name == settings.ModelName);
        _files.Text = selected is null ? "" : S.ModelFiles(selected.ModelFile);
        if ((selected?.HasIndex ?? false) != _hasIndex)
        {
            _hasIndex = selected?.HasIndex ?? false;
            RenderVoice();
        }

        foreach (var slider in _sliders) slider.Update(slider.Num.Get(settings));
        _advancedRow.Description = S.PitchAdvancedDetail(
            NumInfo.Hertz(settings.PitchOffsetHz),
            NumInfo.Ceiling(settings.PitchFloorHz),
            NumInfo.Ceiling(settings.PitchCeilingHz));
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
