using System.Text.Json.Nodes;
using CausalVC.Training.Trainer;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Windows.Storage.Pickers;

namespace CausalVC.Training.Ui.Pages;

/// The form for a new voice (../ui/src/pages.rs: voice_page). The trainer checks it; a field it
/// refuses shows the reason in place of its description until it is edited.
public sealed class VoicePage : PageBase
{
    const string WarningGlyph = "";

    static readonly string[] AudioExtensions = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac", ".wma"];

    /// Typed or pasted, or filled in by the pickers.
    readonly TextBox _audio = new() { Width = 320, PlaceholderText = S.AudioPlaceholder };
    readonly TextBox _name = new() { Width = 240 };
    readonly NumberBox _steps = Number(1000);
    readonly NumberBox _saveEvery = Number(1000);
    readonly NumberBox _batch = Number(1);
    readonly ToggleSwitch _fixes = Toggle();
    readonly ToggleSwitch _restart = Toggle();
    readonly ComboBox _base = new() { MinWidth = 220 };
    readonly ComboBox _gpu = new() { MinWidth = 220, ItemsSource = new[] { S.GpuLoading }, SelectedIndex = 0, IsEnabled = false };
    readonly ToggleSwitch _install = Toggle();
    readonly TextBlock _appModels = Build.Text("");
    readonly SettingRow _appModelsRow;
    /// Rows by the trainer's field names, with their usual descriptions.
    readonly Dictionary<string, (SettingRow Row, string Description)> _rows = [];
    List<string> _bases = ["default"];
    List<Gpu> _gpus = [];
    string _wantedGpu = "cuda:0";
    bool _formApplied;

    public VoicePage(IHost host) : base(host)
    {
        _audio.VerticalAlignment = VerticalAlignment.Center;
        _audio.TextChanged += (_, _) => ClearError("audio");
        _appModels.MaxWidth = 320;
        _appModels.VerticalAlignment = VerticalAlignment.Center;
        _base.ItemsSource = new[] { S.BaseDefault };
        _base.SelectedIndex = 0;
        _gpu.SelectionChanged += (_, _) =>
        {
            if (_gpu.SelectedIndex >= 0 && _gpu.SelectedIndex < _gpus.Count) _wantedGpu = _gpus[_gpu.SelectedIndex].Id;
        };
        _install.Toggled += (_, _) => _appModelsRow!.Visibility = _install.IsOn ? Visibility.Visible : Visibility.Collapsed;
        _name.TextChanged += (_, _) => ClearError("name");
        _steps.ValueChanged += (_, _) => ClearError("steps");
        _saveEvery.ValueChanged += (_, _) => ClearError("save_every");
        _batch.ValueChanged += (_, _) => ClearError("batch");

        var audioRow = Row("audio", S.Audio, S.AudioDetail, Build.Row(
            _audio,
            Build.Button(S.ChooseFolder, async (_, _) => await PickAudioFolder(), ""),
            Build.Button(S.ChooseFile, async (_, _) => await PickAudioFile(), "")));
        Column.Children.Add(new SettingsSection(S.SectionDataset)
            .Add(audioRow)
            .Add(Row("name", S.VoiceName, S.VoiceNameDetail, _name)));

        Column.Children.Add(new SettingsSection(S.SectionTraining)
            .Add(Row("steps", S.Steps, S.StepsDetail, _steps))
            .Add(Row("save_every", S.SaveEvery, S.SaveEveryDetail, _saveEvery))
            .Add(Row("batch", S.Batch, S.BatchDetail, _batch))
            .Add(new SettingRow(S.Fixes, S.FixesDetail, _fixes))
            .Add(new SettingRow(S.Restart, S.RestartDetail, _restart)));

        Column.Children.Add(new SettingsSection(S.SectionHardware)
            .Add(new SettingRow(S.Base, S.BaseDetail, _base))
            .Add(new SettingRow(S.Gpu, S.GpuDetail, _gpu)));

        _appModelsRow = new SettingRow(S.InstallFolder, control: Build.Row(
            _appModels, Build.Button(S.Change, async (_, _) => await PickAppModels())));
        Column.Children.Add(new SettingsSection(S.SectionInstall)
            .Add(new SettingRow(S.Install, S.InstallDetail, _install))
            .Add(_appModelsRow));

        _fixes.IsOn = true;
        _install.IsOn = true;
        _steps.Value = 10000;
        _saveEvery.Value = 10000;
        _batch.Value = 8;
    }

    static NumberBox Number(double step) => new()
    {
        Width = 140,
        Minimum = 1,
        SmallChange = step,
        LargeChange = step,
        SpinButtonPlacementMode = NumberBoxSpinButtonPlacementMode.Compact,
        ValidationMode = NumberBoxValidationMode.InvalidInputOverwritten,
        NumberFormatter = new Windows.Globalization.NumberFormatting.DecimalFormatter { FractionDigits = 0, IsGrouped = false },
    };

    static ToggleSwitch Toggle() => new() { OnContent = "", OffContent = "", MinWidth = 0 };

    SettingRow Row(string field, string title, string description, UIElement control)
    {
        var row = new SettingRow(title, description, control);
        _rows[field] = (row, description);
        return row;
    }

    /// The form as last used, once, when the trainer first says hello (it restarts after a
    /// crash; what was typed since stays).
    public void ApplyForm(Form form, List<string> bases)
    {
        _bases = bases.Count > 0 ? bases : new List<string> { "default" };
        _base.ItemsSource = _bases.Select(name => name == "default" ? S.BaseDefault : name).ToList();
        _base.SelectedIndex = Math.Max(0, _bases.IndexOf(form.Base));
        if (_formApplied) return;
        _formApplied = true;
        SetAudio(form.Audio);
        _name.Text = form.Name;
        _steps.Value = form.Steps;
        _saveEvery.Value = form.SaveEvery;
        _batch.Value = form.Batch;
        _fixes.IsOn = form.Fixes;
        _install.IsOn = form.AppModels.Length > 0;
        _appModels.Text = form.AppModels.Length > 0
            ? form.AppModels
            : TrainerConnection.ProjectDir() is { } root ? Path.Combine(root, "realtime", "models") : "";
        if (form.Device.Length > 0) _wantedGpu = form.Device;
        ClearErrors();
    }

    public void SetGpus(List<Gpu> gpus)
    {
        _gpus = gpus;
        _gpu.ItemsSource = gpus.Select(gpu => gpu.Label is { } label ? $"{label} ({gpu.Id})" : gpu.Id == "cpu" ? S.GpuCpu : gpu.Id).ToList();
        _gpu.SelectedIndex = Math.Max(0, gpus.FindIndex(gpu => gpu.Id == _wantedGpu));
        _gpu.IsEnabled = gpus.Count > 1;
    }

    /// The form as a build config (start or queue). Starting over applies to this build only.
    public JsonObject Config()
    {
        static JsonNode? Whole(NumberBox box) => double.IsNaN(box.Value) ? "" : JsonValue.Create((long)Math.Round(box.Value));
        var config = new JsonObject
        {
            ["name"] = _name.Text.Trim(),
            ["audio"] = _audio.Text.Trim(),
            ["steps"] = Whole(_steps),
            ["save_every"] = Whole(_saveEvery),
            ["batch"] = Whole(_batch),
            ["base"] = _base.SelectedIndex >= 0 && _base.SelectedIndex < _bases.Count ? _bases[_base.SelectedIndex] : "default",
            ["device"] = _wantedGpu,
            ["fixes"] = _fixes.IsOn,
            ["restart"] = _restart.IsOn,
            ["app_models"] = _install.IsOn ? _appModels.Text.Trim() : "",
        };
        _restart.IsOn = false;
        ClearErrors();
        return config;
    }

    public void ShowError(string field, string text)
    {
        if (!_rows.TryGetValue(field, out var entry)) return;
        entry.Row.Description = text;
        entry.Row.Glyph = WarningGlyph;
    }

    void ClearError(string field)
    {
        if (!_rows.TryGetValue(field, out var entry)) return;
        entry.Row.Description = entry.Description;
        entry.Row.Glyph = null;
    }

    void ClearErrors()
    {
        foreach (var field in _rows.Keys) ClearError(field);
    }

    void SetAudio(string path)
    {
        _audio.Text = path;
        _audio.SelectionStart = path.Length;  // the end of a long path, which names the folder
    }

    async Task PickAudioFolder()
    {
        var picker = new FolderPicker();
        picker.FileTypeFilter.Add("*");
        WinRT.Interop.InitializeWithWindow.Initialize(picker, Host.Hwnd);
        if (await picker.PickSingleFolderAsync() is { } folder) SetAudio(folder.Path);
    }

    async Task PickAudioFile()
    {
        var picker = new FileOpenPicker();
        foreach (var extension in AudioExtensions) picker.FileTypeFilter.Add(extension);
        picker.FileTypeFilter.Add("*");
        WinRT.Interop.InitializeWithWindow.Initialize(picker, Host.Hwnd);
        if (await picker.PickSingleFileAsync() is { } file) SetAudio(file.Path);
    }

    async Task PickAppModels()
    {
        var picker = new FolderPicker();
        picker.FileTypeFilter.Add("*");
        WinRT.Interop.InitializeWithWindow.Initialize(picker, Host.Hwnd);
        if (await picker.PickSingleFolderAsync() is { } folder) _appModels.Text = folder.Path;
    }
}
