// Building blocks shared by the pages, matching COSMIC's settings widgets in the Linux front
// end: a section is a title over a card of rows; a row is a title, an optional description,
// and a control on the right. Themed colours come from styles in App.xaml, so they follow
// light/dark changes.

using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Windows.System;
using Windows.UI;

namespace CausalVC.Realtime.Ui;

static class Build
{
    public static Style Style(string key) => (Style)Application.Current.Resources[key];

    public static TextBlock Text(string text, string style = "BodyTextBlockStyle") =>
        new() { Text = text, Style = Style(style), TextWrapping = TextWrapping.Wrap };

    public static TextBlock Caption(string text) => Text(text, "SecondaryCaptionStyle");

    public static Button Button(string text, RoutedEventHandler click, string? glyph = null, string? style = null)
    {
        var button = new Button { Content = glyph is null ? (object)text : WithIcon(glyph, text) };
        if (style is not null) button.Style = Style(style);
        button.Click += click;
        return button;
    }

    /// A red button for stopping and deleting. Its colours are set as the button's own
    /// lightweight-styling resources, so hover and press states stay red too.
    public static Button CriticalButton(string text, RoutedEventHandler click, string? glyph = null)
    {
        var button = Button(text, click, glyph);
        var white = new SolidColorBrush(Color.FromArgb(255, 255, 255, 255));
        foreach (var (key, color) in new[]
        {
            ("ButtonBackground", Color.FromArgb(255, 0xC4, 0x2B, 0x1C)),
            ("ButtonBackgroundPointerOver", Color.FromArgb(255, 0xD1, 0x34, 0x38)),
            ("ButtonBackgroundPressed", Color.FromArgb(255, 0xA7, 0x2B, 0x1E)),
            ("ButtonBorderBrush", Color.FromArgb(255, 0xC4, 0x2B, 0x1C)),
            ("ButtonBorderBrushPointerOver", Color.FromArgb(255, 0xD1, 0x34, 0x38)),
            ("ButtonBorderBrushPressed", Color.FromArgb(255, 0xA7, 0x2B, 0x1E)),
        })
            button.Resources[key] = new SolidColorBrush(color);
        foreach (var key in new[] { "ButtonForeground", "ButtonForegroundPointerOver", "ButtonForegroundPressed" })
            button.Resources[key] = white;
        return button;
    }

    /// The theme's critical (red) colour, for a delete icon.
    public static Brush CriticalBrush => (Brush)Application.Current.Resources["SystemFillColorCriticalBrush"];

    public static StackPanel WithIcon(string glyph, string text) => new()
    {
        Orientation = Orientation.Horizontal,
        Spacing = 8,
        Children = { new FontIcon { Glyph = glyph, FontSize = 14 }, new TextBlock { Text = text } },
    };

    public static Button IconButton(string glyph, string tooltip, RoutedEventHandler click)
    {
        var button = new Button
        {
            Content = new FontIcon { Glyph = glyph, FontSize = 14 },
            Style = Style("SubtleIconButtonStyle"),
        };
        ToolTipService.SetToolTip(button, tooltip);
        button.Click += click;
        return button;
    }

    public static StackPanel Row(params UIElement[] children)
    {
        var row = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8, VerticalAlignment = VerticalAlignment.Center };
        foreach (var child in children) row.Children.Add(child);
        return row;
    }

    /// A row spanning the page with content on the left and buttons on the right.
    public static Grid Bar(UIElement left, params UIElement[] right)
    {
        var grid = new Grid { ColumnSpacing = 8 };
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        grid.Children.Add(left);
        var buttons = Row(right);
        Grid.SetColumn(buttons, 1);
        grid.Children.Add(buttons);
        return grid;
    }

    /// Replace a ComboBox's items only when they changed (re-setting closes an open list).
    public static void SetItems(ComboBox box, IReadOnlyList<string> items)
    {
        var current = box.ItemsSource as IReadOnlyList<string>;
        if (current is null || !current.SequenceEqual(items))
            box.ItemsSource = items.ToList();
    }
}

/// A title over a card of rows separated by dividers.
public sealed class SettingsSection : StackPanel
{
    readonly StackPanel _rows = new();

    public SettingsSection(string? title)
    {
        Spacing = 8;
        if (title is not null) Children.Add(Build.Text(title, "SectionTitleStyle"));
        Children.Add(new Border { Style = Build.Style("SettingsCardStyle"), Child = _rows });
    }

    public SettingsSection Add(UIElement row)
    {
        if (_rows.Children.Count > 0) _rows.Children.Add(new Border { Style = Build.Style("SettingsDividerStyle") });
        _rows.Children.Add(row is SettingRow ? row : new Border { Padding = new Thickness(16, 12, 16, 12), Child = row });
        return this;
    }

    /// Remove every row. Rows that were wrapped are taken out of their wrappers, so the same
    /// elements can be added again (an element can only have one parent).
    public void Clear()
    {
        foreach (var child in _rows.Children)
            if (child is Border { Child: not null } wrapper) wrapper.Child = null;
        _rows.Children.Clear();
    }
}

/// Title and description on the left, a control on the right; below each other when narrow.
public sealed class SettingRow : Grid
{
    const double WrapWidth = 560;
    readonly StackPanel _text = new() { Spacing = 2, VerticalAlignment = VerticalAlignment.Center };
    readonly TextBlock _title;
    readonly TextBlock _description;
    readonly ContentPresenter _control = new() { VerticalAlignment = VerticalAlignment.Center };
    readonly FontIcon _icon = new() { FontSize = 16, Visibility = Visibility.Collapsed, VerticalAlignment = VerticalAlignment.Center };

    public SettingRow(string title, string? description = null, UIElement? control = null)
    {
        Padding = new Thickness(16, 12, 16, 12);
        ColumnSpacing = 12;
        RowSpacing = 8;
        MinHeight = 64;
        ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

        _title = Build.Text(title);
        _description = Build.Caption(description ?? "");
        _description.Visibility = string.IsNullOrEmpty(description) ? Visibility.Collapsed : Visibility.Visible;
        _text.Children.Add(_title);
        _text.Children.Add(_description);
        _control.Content = control;

        Children.Add(_icon);
        SetColumn(_text, 1);
        Children.Add(_text);
        Children.Add(_control);
        SizeChanged += (_, args) => Arrange(args.NewSize.Width);
        Arrange(double.PositiveInfinity);
    }

    public string Title { set => _title.Text = value; }

    public string Description
    {
        set
        {
            _description.Text = value;
            _description.Visibility = string.IsNullOrEmpty(value) ? Visibility.Collapsed : Visibility.Visible;
        }
    }

    public UIElement? Control { set => _control.Content = value; }

    /// A Segoe Fluent glyph before the title, or null for none.
    public string? Glyph
    {
        set
        {
            _icon.Glyph = value ?? "";
            _icon.Visibility = value is null ? Visibility.Collapsed : Visibility.Visible;
        }
    }

    void Arrange(double width)
    {
        bool narrow = width < WrapWidth;
        SetRow(_control, narrow ? 1 : 0);
        SetColumn(_control, narrow ? 1 : 2);
        SetColumnSpan(_control, narrow ? 2 : 1);
        _control.HorizontalAlignment = narrow ? HorizontalAlignment.Left : HorizontalAlignment.Right;
    }
}

/// A slider row whose value can be clicked to type an exact one (Enter or clicking away
/// applies, Escape cancels), as in the Linux front end.
public sealed class SliderSetting
{
    const double SliderWidth = 240;
    const double ValueWidth = 80;

    public readonly Num Num;
    public readonly SettingRow Row;
    readonly Slider _slider;
    readonly Button _valueButton;
    readonly TextBox _valueBox;
    readonly Grid _valueHost = new() { Width = ValueWidth };
    readonly DispatcherQueueTimerWrapper _settle;
    readonly Action<Num, double> _changed;
    bool _updating;
    bool _cancelEdit;

    /// True while the user holds the slider: incoming state must not move it.
    public bool Dragging { get; private set; }

    /// <param name="changed">Called with values to send: live while dragging, or once the
    /// slider settles for settings that restart conversion.</param>
    public SliderSetting(string title, Num num, Action<Num, double> changed)
    {
        Num = num;
        _changed = changed;
        var (min, max, step) = num.Range();
        _slider = new Slider
        {
            Minimum = min,
            Maximum = max,
            StepFrequency = step,
            SmallChange = step,
            LargeChange = step * 10,
            Width = SliderWidth,
            IsThumbToolTipEnabled = false,
            VerticalAlignment = VerticalAlignment.Center,
        };
        _valueButton = new Button
        {
            Style = Build.Style("ValueButtonStyle"),
            HorizontalAlignment = HorizontalAlignment.Stretch,
        };
        ToolTipService.SetToolTip(_valueButton, S.ValueClickToType);
        _valueBox = new TextBox { Visibility = Visibility.Collapsed, TextAlignment = TextAlignment.Right };
        _valueHost.Children.Add(_valueButton);
        _valueHost.Children.Add(_valueBox);

        _settle = new DispatcherQueueTimerWrapper(TimeSpan.FromMilliseconds(400), () => _changed(Num, _slider.Value));
        _slider.ValueChanged += OnSliderChanged;
        _slider.AddHandler(UIElement.PointerPressedEvent, new PointerEventHandler((_, _) => Dragging = true), true);
        _slider.AddHandler(UIElement.PointerCaptureLostEvent, new PointerEventHandler((_, _) => EndDrag()), true);
        _slider.AddHandler(UIElement.PointerReleasedEvent, new PointerEventHandler((_, _) => EndDrag()), true);
        _valueButton.Click += (_, _) => BeginEdit();
        _valueBox.KeyDown += OnValueKey;
        _valueBox.LostFocus += (_, _) => EndEdit(apply: !_cancelEdit);
        _valueBox.TextChanged += (_, _) => ToolTipService.SetToolTip(_valueBox, null);

        var control = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        control.Children.Add(_slider);
        control.Children.Add(_valueHost);
        Row = new SettingRow(title, num.Description(), control);
    }

    void EndDrag()
    {
        if (!Dragging) return;
        Dragging = false;
        if (Num.NeedsRestart()) _settle.Fire();
    }

    void OnSliderChanged(object sender, RangeBaseValueChangedEventArgs args)
    {
        _valueButton.Content = Num.Format(args.NewValue);
        if (_updating) return;
        if (Num.NeedsRestart()) _settle.Restart(); // keyboard and drags: sent once it settles
        else _changed(Num, args.NewValue);
    }

    /// Show the engine's value, unless the user is moving or typing this one.
    public void Update(double value)
    {
        if (Dragging || _settle.Pending) return;
        _updating = true;
        _slider.Value = value;
        _valueButton.Content = Num.Format(value);
        _updating = false;
    }

    void BeginEdit()
    {
        _cancelEdit = false;
        _valueBox.Text = Num.EditText(_slider.Value);
        _valueButton.Visibility = Visibility.Collapsed;
        _valueBox.Visibility = Visibility.Visible;
        _valueBox.Focus(FocusState.Programmatic);
        _valueBox.SelectAll();
    }

    void OnValueKey(object sender, KeyRoutedEventArgs args)
    {
        if (args.Key == VirtualKey.Enter)
        {
            if (Num.Parse(_valueBox.Text) is null)
            {
                // Keep the field open so the typo can be fixed.
                ToolTipService.SetToolTip(_valueBox, S.ValueInvalid);
                _valueBox.SelectAll();
            }
            else
            {
                EndEdit(apply: true);
            }
            args.Handled = true;
        }
        else if (args.Key == VirtualKey.Escape)
        {
            _cancelEdit = true;
            EndEdit(apply: false);
            args.Handled = true;
        }
    }

    void EndEdit(bool apply)
    {
        if (_valueBox.Visibility != Visibility.Visible) return;
        _valueBox.Visibility = Visibility.Collapsed;
        _valueButton.Visibility = Visibility.Visible;
        // Invalid text is dropped: clicking away leaves the value as it was.
        if (apply && Num.Parse(_valueBox.Text) is double value)
        {
            _updating = true;
            _slider.Value = value;
            _updating = false;
            _changed(Num, value);
        }
    }
}

/// A one-shot, restartable timer on the UI thread.
public sealed class DispatcherQueueTimerWrapper
{
    readonly Microsoft.UI.Dispatching.DispatcherQueueTimer _timer;
    readonly Action _action;

    public DispatcherQueueTimerWrapper(TimeSpan delay, Action action)
    {
        _action = action;
        _timer = Microsoft.UI.Dispatching.DispatcherQueue.GetForCurrentThread().CreateTimer();
        _timer.Interval = delay;
        _timer.IsRepeating = false;
        _timer.Tick += (_, _) => _action();
    }

    public bool Pending => _timer.IsRunning;

    public void Restart()
    {
        _timer.Stop();
        _timer.Start();
    }

    /// Run now if a run is pending.
    public void Fire()
    {
        if (!_timer.IsRunning) return;
        _timer.Stop();
        _action();
    }
}
