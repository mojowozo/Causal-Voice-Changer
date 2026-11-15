using CausalVC.Realtime.Engine;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CausalVC.Realtime.Ui.Pages;

/// Chunk size, GPU and noise reduction (pages.rs: performance_page).
public sealed class PerformancePage : PageBase
{
    public static readonly string[] DenoiseMethods = ["hushwire", "rnnoise"];

    readonly SliderSetting _chunk;
    readonly Action<IReadOnlyList<string>, int> _setGpu;
    readonly ToggleSwitch _denoise = new() { OnContent = "", OffContent = "", MinWidth = 0 };
    readonly SettingRow _methodRow;
    readonly Action<IReadOnlyList<string>, int> _setMethod;
    readonly Button _download;
    List<Gpu> _gpus = [];
    bool _updating;

    public PerformancePage(IHost host) : base(host)
    {
        _chunk = new SliderSetting(S.Chunk, Num.BlockTime, SendNum);
        _denoise.Toggled += (_, _) =>
        {
            if (!_updating) Host.SendSetting("input_denoise", _denoise.IsOn);
        };
        var method = Choice(index => Host.SendSetting("input_denoise_method", DenoiseMethods[index]), out _setMethod);
        method.MinWidth = 220;
        _download = Build.Button(S.DenoiseHushwireDownload, async (_, _) => await ConfirmHushwireDownload(), "", "AccentButtonStyle");
        _methodRow = new SettingRow(S.DenoiseMethod, S.DenoiseHushwireReady, Build.Row(_download, method));

        Column.Children.Add(new SettingsSection(S.SectionBuffering).Add(_chunk.Row));
        Column.Children.Add(new SettingsSection(S.SectionInference).Add(new SettingRow(S.Gpu, S.GpuDetail,
            Choice(index => Host.SendSetting("gpu", _gpus[index].Id), out _setGpu))));
        Column.Children.Add(new SettingsSection(S.SectionNoise)
            .Add(new SettingRow(S.InputDenoise, S.InputDenoiseDetail, _denoise))
            .Add(_methodRow));
        Column.Children.Add(ResetRow(S.ResetPerformance, "performance"));
    }

    public override void Update(EngineState state)
    {
        var settings = state.Settings;
        _chunk.Update(settings.BlockTime);
        _gpus = state.Gpus;
        _setGpu(_gpus.Select(gpu => gpu.Label ?? S.GpuAutomatic).ToList(), _gpus.FindIndex(gpu => gpu.Id == settings.Gpu));
        _setMethod([S.DenoiseMethodHushwire, S.DenoiseMethodRnnoise], Array.IndexOf(DenoiseMethods, settings.InputDenoiseMethod));
        var (description, download) = (settings.InputDenoiseMethod, state.Hushwire) switch
        {
            ("rnnoise", _) when state.RnNoise => (S.DenoiseRnnoiseReady, false),
            ("rnnoise", _) => (S.InputDenoiseMissing, false),
            (_, "ready") => (S.DenoiseHushwireReady, false),
            (_, "downloading") => (S.DenoiseHushwireDownloading, false),
            (_, "failed") => (S.DenoiseHushwireFailed, true),
            (_, "unavailable") => (S.DenoiseHushwireUnavailable, false),
            _ => (S.DenoiseHushwireNeedsDownload, true),
        };
        _methodRow.Description = description;
        _download.Visibility = download ? Visibility.Visible : Visibility.Collapsed;
        _updating = true;
        _denoise.IsOn = settings.InputDenoise;
        _updating = false;
    }

    /// Hushwire's model is Amazon's: nothing downloads until the user accepts this notice.
    async Task ConfirmHushwireDownload()
    {
        var dialog = new ContentDialog
        {
            XamlRoot = Host.XamlRoot,
            Title = S.HushwireNoticeTitle,
            Content = Build.Text(S.HushwireNoticeBody),
            PrimaryButtonText = S.HushwireNoticeAccept,
            CloseButtonText = S.Cancel,
            DefaultButton = ContentDialogButton.Close,
        };
        if (await dialog.ShowAsync() == ContentDialogResult.Primary)
            Host.Send("download_hushwire");
    }
}
