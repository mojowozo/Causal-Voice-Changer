using CausalVC.Realtime.Engine;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;

namespace CausalVC.Realtime.Ui.Pages;

/// The engine's log, with save and clear (pages.rs: log_page). Not a scrolling column: the
/// text box fills the page and scrolls itself.
public sealed class LogPage : Grid
{
    readonly TextBox _text = new()
    {
        IsReadOnly = true,
        AcceptsReturn = true,
        TextWrapping = TextWrapping.NoWrap,
        FontSize = 12,
    };

    public LogPage(Action save, Action clear)
    {
        Padding = new Thickness(24, 0, 24, PageBase.ControlBarRoom);
        RowSpacing = 8;
        RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        _text.FontFamily = (FontFamily)Application.Current.Resources["MonoFont"];
        ScrollViewer.SetHorizontalScrollBarVisibility(_text, ScrollBarVisibility.Auto);
        ScrollViewer.SetVerticalScrollBarVisibility(_text, ScrollBarVisibility.Auto);
        Children.Add(Build.Bar(new Border(),
            Build.Button(S.SaveLog, (_, _) => save()),
            Build.Button(S.ClearLog, (_, _) => clear())));
        SetRow(_text, 1);
        Children.Add(_text);
    }

    /// Show the log and keep the view at its end.
    public void SetText(string text)
    {
        if (_text.Text == text) return;
        _text.Text = text;
        _text.SelectionStart = text.Length;
        if (_text.FindDescendant<ScrollViewer>() is { } scroller)
            scroller.ChangeView(null, scroller.ScrollableHeight, null, disableAnimation: true);
    }
}

static class TreeExtensions
{
    public static T? FindDescendant<T>(this DependencyObject parent) where T : DependencyObject
    {
        for (int i = 0; i < VisualTreeHelper.GetChildrenCount(parent); i++)
        {
            var child = VisualTreeHelper.GetChild(parent, i);
            if (child is T match) return match;
            if (child.FindDescendant<T>() is { } deeper) return deeper;
        }
        return null;
    }
}
