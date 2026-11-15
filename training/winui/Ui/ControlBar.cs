using CausalVC.Training.Trainer;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CausalVC.Training.Ui;

/// Start or stop, queue, and the progress of the build, along the bottom (../ui/src/pages.rs:
/// control_bar). Floats over the page in in-app acrylic.
public sealed class ControlBar : Grid
{
    readonly Button _start;
    readonly Button _stop;
    readonly TextBlock _stopLabel;
    readonly TextBlock _status = Build.Text("");
    readonly ProgressBar _progress = new() { Minimum = 0, Maximum = 1, Height = 6, MinHeight = 6 };
    readonly TextBlock _times = Build.Caption("");

    public ControlBar(Action start, Action queue, Action stop)
    {
        ColumnSpacing = 12;
        foreach (var width in new[] { GridLength.Auto, GridLength.Auto, new GridLength(1, GridUnitType.Star) })
            ColumnDefinitions.Add(new ColumnDefinition { Width = width });

        _start = Build.Button(S.Start, (_, _) => start(), "", "AccentButtonStyle");
        _stop = Build.CriticalButton(S.Stop, (_, _) => stop(), "");
        _stopLabel = (TextBlock)((StackPanel)_stop.Content).Children[1];
        ToolTipService.SetToolTip(_stop, S.StopDetail);
        var primary = new Grid { VerticalAlignment = VerticalAlignment.Center, Children = { _start, _stop } };
        var queueButton = Build.Button(S.QueueBuild, (_, _) => queue(), "");
        queueButton.VerticalAlignment = VerticalAlignment.Center;
        SetColumn(queueButton, 1);

        _status.MaxLines = 2;
        _status.TextTrimming = TextTrimming.CharacterEllipsis;
        var status = new StackPanel { Spacing = 4, VerticalAlignment = VerticalAlignment.Center, Children = { _status, _progress, _times } };
        SetColumn(status, 2);

        Children.Add(primary);
        Children.Add(queueButton);
        Children.Add(status);
    }

    public void Update(TrainerState state)
    {
        _start.Visibility = state.Running ? Visibility.Collapsed : Visibility.Visible;
        _stop.Visibility = state.Running ? Visibility.Visible : Visibility.Collapsed;
        _stopLabel.Text = state.Phase == "stopping" ? S.EndNow : S.Stop;
        _status.Text = S.Phase(state);

        var progress = state.Running ? state.Progress : null;
        bool starting = state.Phase == "starting";
        _progress.Visibility = progress is not null || starting ? Visibility.Visible : Visibility.Collapsed;
        _progress.IsIndeterminate = starting && progress is null;
        if (progress is not null) _progress.Value = progress.Total > 0 ? (double)progress.Step / progress.Total : 0;
        _times.Visibility = progress is null ? Visibility.Collapsed : Visibility.Visible;
        if (progress is not null)
            _times.Text = $"{S.Progress(progress.Step, progress.Total)} · {S.ProgressTimes(progress.Elapsed, progress.Remaining, progress.Rate)}";
    }
}
