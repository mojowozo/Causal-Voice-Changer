using System.Text.Json.Nodes;
using CausalVC.Realtime.Engine;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CausalVC.Realtime.Ui.Pages;

public enum PageKind { Model, Library, Audio, Performance, Recording, Log }

/// What pages need from the window: the engine and a few window services.
public interface IHost
{
    EngineState? State { get; }
    Meters Meters { get; }
    long? Send(string cmd, JsonObject? args = null);
    void SendSetting(string key, JsonNode? value);
    void Toast(string text);
    void ShowPage(PageKind page);
    /// Change the output gain by the output-level suggestion.
    void ApplyLoudness();
    XamlRoot? XamlRoot { get; }
    nint Hwnd { get; }
}

/// A scrolling column of settings sections, at most 900 px wide.
/// (A UserControl holding a ScrollViewer: WinUI 3's ScrollViewer is sealed.)
public abstract class PageBase : UserControl
{
    /// Room at the bottom for the control bar, which floats over the page.
    public const double ControlBarRoom = 112;

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

    /// Re-render from the engine's latest state.
    public abstract void Update(EngineState state);

    /// About 20 times a second while running.
    public virtual void UpdateMeters(EngineState state, Meters meters) { }

    protected void SendNum(Num num, double value)
    {
        if (Host.State is { } state) num.Set(state.Settings, value);
        Host.SendSetting(num.Key(), JsonValue.Create(value));
    }

    protected Grid ResetRow(string label, string group) => Build.Bar(
        new Border(),
        Build.Button(label, (_, _) => Host.Send("reset_settings", new JsonObject { ["group"] = group })));

    /// A ComboBox that reports the chosen index, ignoring changes made by Update.
    protected static ComboBox Choice(Action<int> chosen, out Action<IReadOnlyList<string>, int> set)
    {
        var box = new ComboBox { MinWidth = 220, MaxWidth = 420 };
        bool updating = false;
        box.SelectionChanged += (_, _) =>
        {
            if (!updating && box.SelectedIndex >= 0) chosen(box.SelectedIndex);
        };
        set = (items, index) =>
        {
            updating = true;
            Build.SetItems(box, items);
            if (box.SelectedIndex != index) box.SelectedIndex = index;
            updating = false;
        };
        return box;
    }
}
