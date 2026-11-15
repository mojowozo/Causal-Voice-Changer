using CausalVC.Realtime.Engine;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CausalVC.Realtime.Ui;

/// Start/stop, status, meters and timings, along the bottom (pages.rs: control_bar).
public sealed class ControlBar : Grid
{
    const double MeterWidth = 90;

    readonly Button _start;
    readonly Button _converting;
    readonly Button _passthrough;
    readonly Button _passthroughActive;
    readonly TextBlock _status = Build.Text("");
    readonly ProgressBar _in = Meter(), _out = Meter(), _mon = Meter();
    readonly TextBlock _latency = Build.Caption("");
    readonly TextBlock _inference = Build.Caption("");
    readonly StackPanel _level;
    readonly FontIcon _levelIcon = new() { FontSize = 12 };
    readonly TextBlock _levelText = Build.Caption("");

    /// <param name="run">Called with "vc" or "passthrough": starts it, or stops it if running.</param>
    public ControlBar(Action<string> run)
    {
        ColumnSpacing = 12;
        foreach (var width in new[] { GridLength.Auto, GridLength.Auto, new GridLength(1, GridUnitType.Star), GridLength.Auto, new GridLength(150) })
            ColumnDefinitions.Add(new ColumnDefinition { Width = width });

        _start = Build.Button(S.Start, (_, _) => run("vc"), "\uE768", "AccentButtonStyle");
        _converting = Build.CriticalButton(S.Converting, (_, _) => run("vc"), "\uE71A");
        _passthrough = Build.Button(S.Passthrough, (_, _) => run("passthrough"));
        _passthroughActive = Build.CriticalButton(S.PassthroughActive, (_, _) => run("passthrough"));
        var startHost = new Grid { VerticalAlignment = VerticalAlignment.Center, Children = { _start, _converting } };
        var passthroughHost = new Grid { VerticalAlignment = VerticalAlignment.Center, Children = { _passthrough, _passthroughActive } };
        SetColumn(passthroughHost, 1);

        _status.VerticalAlignment = VerticalAlignment.Center;
        _status.MaxLines = 3;
        _status.TextTrimming = TextTrimming.CharacterEllipsis;
        SetColumn(_status, 2);

        var meters = new StackPanel { Spacing = 2, VerticalAlignment = VerticalAlignment.Center };
        foreach (var (label, bar) in new[] { (S.MeterIn, _in), (S.MeterOut, _out), (S.MeterMon, _mon) })
        {
            var caption = Build.Caption(label);
            caption.Width = 32;
            bar.VerticalAlignment = VerticalAlignment.Center;
            meters.Children.Add(new StackPanel { Orientation = Orientation.Horizontal, Spacing = 4, Children = { caption, bar } });
        }
        SetColumn(meters, 3);

        ToolTipService.SetToolTip(_latency, S.LatencyDetail);
        ToolTipService.SetToolTip(_inference, S.InferenceDetail);
        _level = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 4, Children = { _levelIcon, _levelText } };
        var stats = new StackPanel { Spacing = 2, VerticalAlignment = VerticalAlignment.Center, Children = { _latency, _inference, _level } };
        SetColumn(stats, 4);

        Children.Add(startHost);
        Children.Add(passthroughHost);
        Children.Add(_status);
        Children.Add(meters);
        Children.Add(stats);
    }

    static ProgressBar Meter() => new() { Width = MeterWidth, Minimum = 0, Maximum = 1, Height = 6, MinHeight = 6 };

    public string Status { set => _status.Text = value; }

    public void Update(EngineState state, bool starting)
    {
        bool converting = state.Running && state.Function == "vc";
        bool passthrough = state.Running && state.Function == "passthrough";
        _start.Visibility = converting ? Visibility.Collapsed : Visibility.Visible;
        _converting.Visibility = converting ? Visibility.Visible : Visibility.Collapsed;
        _passthrough.Visibility = passthrough ? Visibility.Collapsed : Visibility.Visible;
        _passthroughActive.Visibility = passthrough ? Visibility.Visible : Visibility.Collapsed;
        ((TextBlock)((StackPanel)_start.Content).Children[1]).Text = starting ? S.Starting : S.Start;
        foreach (var button in new[] { _start, _converting, _passthrough, _passthroughActive })
            button.IsEnabled = !starting;
    }

    public void UpdateMeters(EngineState state, Meters meters)
    {
        _in.Value = meters.Input;
        _out.Value = meters.Output;
        _mon.Value = meters.Monitor;
        _latency.Text = S.Latency(state.DelayMs is long ms ? (ms + (long)(meters.QueuedMs ?? 0)).ToString() : "—");
        _inference.Text = S.Inference(meters.InferMs is double infer ? Math.Round(infer).ToString() : "—");

        var (text, glyph) = meters.Loudness?.Verdict switch
        {
            "good" => (S.LevelGood, "\uE73E"),
            "quiet" => (S.LevelQuiet, "\uE993"),
            "loud" => (S.LevelLoud, "\uE7BA"),
            _ => (S.LevelMeasuring, "\uE994"),
        };
        bool showLevel = state.Running && meters.Loudness is not null;
        _level.Visibility = showLevel ? Visibility.Visible : Visibility.Collapsed;
        _levelText.Text = text;
        _levelIcon.Glyph = glyph;
    }
}
