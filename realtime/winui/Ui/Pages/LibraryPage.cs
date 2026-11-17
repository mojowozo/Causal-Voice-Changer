using System.Diagnostics;
using System.Text.Json.Nodes;
using CausalVC.Realtime.Engine;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CausalVC.Realtime.Ui.Pages;

/// Search, models grouped by voice, rename and delete (library.rs).
public sealed class LibraryPage : PageBase
{
    readonly AutoSuggestBox _search = new() { PlaceholderText = S.LibrarySearch, QueryIcon = new SymbolIcon(Symbol.Find) };
    readonly ComboBox _sort = new() { ItemsSource = new[] { S.LibrarySortName, S.LibrarySortNewest }, SelectedIndex = 0 };
    readonly TextBlock _count = Build.Caption("");
    readonly StackPanel _groups = new() { Spacing = 4 };
    string _rendered = "";

    public LibraryPage(IHost host) : base(host)
    {
        _search.TextChanged += (_, _) => Render();
        _sort.SelectionChanged += (_, _) => Render();
        var toolbar = new Grid { ColumnSpacing = 8, Margin = new Thickness(0, 8, 0, 0) };
        toolbar.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        toolbar.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        toolbar.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        toolbar.Children.Add(_search);
        Grid.SetColumn(_sort, 1);
        toolbar.Children.Add(_sort);
        var reload = Build.Button(S.Reload, (_, _) => Host.Send("reload_models"), "\uE72C");
        Grid.SetColumn(reload, 2);
        toolbar.Children.Add(reload);

        Column.Children.Add(toolbar);
        Column.Children.Add(_count);
        Column.Children.Add(_groups);
    }

    public override void Update(EngineState state) => Render();

    void Render()
    {
        if (Host.State is not { } state) return;
        bool newest = _sort.SelectedIndex == 1;
        string query = _search.Text ?? "";
        string? convertingWith = state.Running && state.Function == "vc" ? state.Settings.ModelName : null;
        // Rebuilding the rows resets focus and hover, so only when something shown changed.
        string key = string.Join("\n", state.Models.Select(m => $"{m.Name}\t{m.Voice}\t{m.SizeBytes}\t{m.Modified}"))
            + $"\n{query}\n{newest}\n{state.Settings.ModelName}\n{convertingWith}";
        if (key == _rendered) return;
        _rendered = key;

        var groups = Library.Grouped(state.Models, query, newest);
        _count.Text = S.LibraryCount(groups.Sum(group => group.Models.Count), state.Models.Count);
        _groups.Children.Clear();
        if (state.Models.Count == 0) _groups.Children.Add(Build.Text(S.ModelsEmpty));
        else if (groups.Count == 0) _groups.Children.Add(Build.Text(S.LibraryNoMatch));
        foreach (var (voice, models) in groups)
        {
            var section = new SettingsSection(S.LibraryVoice(voice, models.Count));
            foreach (var model in models) section.Add(ModelRow(model, state, convertingWith));
            _groups.Children.Add(section);
        }
    }

    SettingRow ModelRow(Model model, EngineState state, string? convertingWith)
    {
        string title = Library.CheckpointLabel(model);
        bool selected = model.Name == state.Settings.ModelName;
        bool inUse = convertingWith == model.Name;

        var use = Build.Button(selected ? S.LibrarySelected : S.LibraryUse,
            (_, _) => Host.SendSetting("model_name", model.Name));
        use.IsEnabled = !selected;
        var rename = Build.IconButton("\uE8AC", inUse ? S.LibraryRenameInUse : S.LibraryRename, async (_, _) => await Rename(model.Name));
        rename.IsEnabled = !inUse;
        var export = Build.IconButton("\uE72D", S.LibraryExport, async (_, _) => await Export(model.Name));
        var folder = Build.IconButton("\uE838", S.LibraryOpenFolder, (_, _) => OpenFolder(model.Name));
        var delete = Build.IconButton("\uE74D", inUse ? S.LibraryDeleteInUse : S.LibraryDelete, async (_, _) => await Delete(model.Name));
        ((FontIcon)delete.Content).Foreground = Build.CriticalBrush;
        delete.IsEnabled = !inUse;
        // A disabled button shows no tooltip, so the reason sits on a wrapper.
        var controls = Build.Row(use, Tip(rename, inUse ? S.LibraryRenameInUse : S.LibraryRename), export, folder,
            Tip(delete, inUse ? S.LibraryDeleteInUse : S.LibraryDelete));

        var row = new SettingRow(title, $"{Library.FormatSize(model.SizeBytes)} · {Library.FormatDate(model.Modified)}", controls)
        {
            Glyph = selected ? "\uE73E" : null,
        };
        ToolTipService.SetToolTip(row, model.Name);
        return row;
    }

    static Border Tip(UIElement child, string tip)
    {
        var wrapper = new Border { Child = child, Background = new Microsoft.UI.Xaml.Media.SolidColorBrush(Windows.UI.Color.FromArgb(0, 0, 0, 0)) };
        ToolTipService.SetToolTip(wrapper, tip);
        return wrapper;
    }

    void OpenFolder(string name)
    {
        var root = EngineConnection.ProjectDir();
        if (root is null) return;
        try
        {
            Process.Start(new ProcessStartInfo("explorer.exe") { ArgumentList = { Path.Combine(root, "models", name) } });
        }
        catch (Exception error)
        {
            Host.Toast(error.Message);
        }
    }

    async Task Export(string name)
    {
        var picker = new Windows.Storage.Pickers.FileSavePicker
        {
            SuggestedFileName = name,
            SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.DocumentsLibrary,
        };
        picker.FileTypeChoices.Add(S.VoiceFileType, new List<string> { ".cvc" });
        WinRT.Interop.InitializeWithWindow.Initialize(picker, Host.Hwnd);
        var file = await picker.PickSaveFileAsync();
        // The picker creates an empty file; the engine replaces it with the voice.
        if (file is not null) Host.Send("export_model", new JsonObject { ["name"] = name, ["path"] = file.Path });
    }

    async Task Delete(string name)
    {
        var dialog = new ContentDialog
        {
            XamlRoot = Host.XamlRoot,
            Title = S.DeleteModelTitle,
            Content = Build.Text(S.DeleteModelBody(name)),
            PrimaryButtonText = S.DeleteModelConfirm,
            CloseButtonText = S.Cancel,
            DefaultButton = ContentDialogButton.Close,
        };
        if (await dialog.ShowAsync() == ContentDialogResult.Primary)
            Host.Send("delete_model", new JsonObject { ["name"] = name });
    }

    async Task Rename(string name)
    {
        var input = new TextBox { Text = name };
        var dialog = new ContentDialog
        {
            XamlRoot = Host.XamlRoot,
            Title = S.RenameModelTitle,
            Content = new StackPanel { Spacing = 12, Children = { Build.Text(S.RenameModelBody), input } },
            PrimaryButtonText = S.RenameModelConfirm,
            CloseButtonText = S.Cancel,
            DefaultButton = ContentDialogButton.Primary,
            IsPrimaryButtonEnabled = false,
        };
        input.TextChanged += (_, _) =>
        {
            var trimmed = input.Text.Trim();
            dialog.IsPrimaryButtonEnabled = trimmed.Length > 0 && trimmed != name;
        };
        dialog.Opened += (_, _) =>
        {
            input.Focus(FocusState.Programmatic);
            input.SelectAll();
        };
        if (await dialog.ShowAsync() == ContentDialogResult.Primary)
        {
            var newName = input.Text.Trim();
            if (newName.Length > 0 && newName != name)
                Host.Send("rename_model", new JsonObject { ["name"] = name, ["new_name"] = newName });
        }
    }
}
