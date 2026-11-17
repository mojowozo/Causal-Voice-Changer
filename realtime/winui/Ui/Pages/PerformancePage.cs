using CausalVC.Realtime.Engine;
using Microsoft.UI.Xaml.Controls;

namespace CausalVC.Realtime.Ui.Pages;

/// Chunk size, GPU and fast GPU replay: how fast conversion runs (pages.rs: performance_page).
public sealed class PerformancePage : PageBase
{
    readonly SliderSetting _chunk;
    readonly Action<IReadOnlyList<string>, int> _setGpu;
    readonly ToggleSwitch _graphs = new() { OnContent = "", OffContent = "", MinWidth = 0 };
    List<Gpu> _gpus = [];
    bool _updating;

    public PerformancePage(IHost host) : base(host)
    {
        _chunk = new SliderSetting(S.Chunk, Num.BlockTime, SendNum);
        _graphs.Toggled += (_, _) =>
        {
            if (!_updating) Host.SendSetting("cuda_graphs", _graphs.IsOn);
        };

        Column.Children.Add(new SettingsSection(S.SectionBuffering).Add(_chunk.Row));
        Column.Children.Add(new SettingsSection(S.SectionInference).Add(new SettingRow(S.Gpu, S.GpuDetail,
            Choice(index => Host.SendSetting("gpu", _gpus[index].Id), out _setGpu)))
            .Add(new SettingRow(S.CudaGraphs, S.CudaGraphsDetail, _graphs)));
        Column.Children.Add(ResetRow(S.ResetPerformance, "performance"));
    }

    public override void Update(EngineState state)
    {
        var settings = state.Settings;
        _chunk.Update(settings.BlockTime);
        _gpus = state.Gpus;
        _setGpu(_gpus.Select(gpu => gpu.Label ?? S.GpuAutomatic).ToList(), _gpus.FindIndex(gpu => gpu.Id == settings.Gpu));
        _updating = true;
        _graphs.IsOn = settings.CudaGraphs;
        _updating = false;
    }
}
