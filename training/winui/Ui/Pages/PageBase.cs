using System.Text.Json.Nodes;
using CausalVC.Training.Trainer;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CausalVC.Training.Ui.Pages;

public enum PageKind { Voice, Builds, Log, About }

/// What pages need from the window: the trainer and a few window services.
public interface IHost
{
    TrainerState State { get; }
    void Send(string cmd, JsonObject? args = null);
    void Toast(string text);
    XamlRoot? XamlRoot { get; }
    nint Hwnd { get; }
}

/// A scrolling column of settings sections, at most 900 px wide.
/// (A UserControl holding a ScrollViewer: WinUI 3's ScrollViewer is sealed.)
public abstract class PageBase : UserControl
{
    /// Room at the bottom for the control bar, which floats over the page.
    public const double ControlBarRoom = 120;

    protected readonly IHost Host;
    protected readonly StackPanel Column = new() { Spacing = 4, MaxWidth = 900 };

    protected PageBase(IHost host)
    {
        Host = host;
        Content = new ScrollViewer
        {
            HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
            Content = new Border
            {
                Padding = new Thickness(24, 0, 24, ControlBarRoom),
                Child = Column,
            },
        };
    }

    /// Re-render from the trainer's latest state.
    public virtual void Update(TrainerState state) { }
}
