using System.Collections.ObjectModel;
using System.Text.Json.Nodes;
using CausalVC.Training.Trainer;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CausalVC.Training.Ui.Pages;

/// The build in progress and the queue after it (../ui/src/pages.rs: builds_page). Queued builds
/// can be dragged into another order, started now, or removed.
public sealed class BuildsPage : PageBase
{
    readonly SettingsSection _current = new(S.BuildsCurrent);
    readonly SettingsSection _queue = new(S.BuildsQueue);
    readonly ObservableCollection<ListViewItem> _rows = [];
    readonly ListView _list;
    int _dragFrom = -1;
    string _rendered = "";

    public BuildsPage(IHost host) : base(host)
    {
        _list = new ListView
        {
            ItemsSource = _rows,
            SelectionMode = ListViewSelectionMode.None,
            CanDragItems = true,
            CanReorderItems = true,
            AllowDrop = true,
            Padding = new Thickness(0, 4, 0, 4),
        };
        // The page scrolls; the list is as tall as its rows.
        ScrollViewer.SetVerticalScrollMode(_list, ScrollMode.Disabled);
        ScrollViewer.SetVerticalScrollBarVisibility(_list, ScrollBarVisibility.Disabled);
        // The ListView moves the row itself; the trainer is told, and its next state redraws the list.
        _list.DragItemsStarting += (_, args) =>
            _dragFrom = args.Items.FirstOrDefault() is ListViewItem row ? _rows.IndexOf(row) : -1;
        _list.DragItemsCompleted += (_, args) =>
        {
            int to = args.Items.FirstOrDefault() is ListViewItem row ? _rows.IndexOf(row) : -1;
            if (_dragFrom >= 0 && to >= 0 && to != _dragFrom)
                Host.Send("move", new JsonObject { ["from"] = _dragFrom, ["to"] = to });
            _dragFrom = -1;
        };

        Column.Children.Add(_current);
        Column.Children.Add(_queue);
        var note = Build.Caption(S.CloseNote);
        note.Margin = new Thickness(4, 12, 4, 0);
        Column.Children.Add(note);
    }

    public override void Update(TrainerState state)
    {
        // Progress arrives every second; the rows only change with the phase or the queue.
        string key = $"{state.Running}\n{S.Phase(state)}\n{state.Current?.Name}\n"
            + string.Join("\n", state.Queue.Select(build => $"{build.Name}\t{build.Steps}"));
        if (key == _rendered) return;
        _rendered = key;

        _current.Clear();
        _current.Add(state.Current is { } build && state.Running
            ? new SettingRow(build.Name, S.Phase(state), Build.Caption(S.BuildsSteps(build.Steps))) { Glyph = "" }
            : new SettingRow(S.BuildsNone));

        _queue.Clear();
        _rows.Clear();
        if (state.Queue.Count == 0)
        {
            _queue.Add(Build.Text(S.BuildsQueueEmpty));
            return;
        }
        for (int index = 0; index < state.Queue.Count; index++)
        {
            int position = index;
            var queued = state.Queue[index];
            var row = new SettingRow($"{index + 1}. {queued.Name}", S.BuildsSteps(queued.Steps), Build.Row(
                Build.Button(S.BuildNow, (_, _) => Host.Send("build_now", new JsonObject { ["index"] = position }), "\uE768"),
                Build.Button(S.Remove, (_, _) => Host.Send("remove", new JsonObject { ["index"] = position }), "\uE738")))
            { Glyph = "\uE76F" };  // a gripper: the row can be dragged
            // Each row is its own container, so moving it never re-parents the row.
            _rows.Add(new ListViewItem
            {
                Content = row,
                Padding = new Thickness(0),
                MinHeight = 0,
                HorizontalContentAlignment = HorizontalAlignment.Stretch,
            });
        }
        _queue.AddUnpadded(_list);
        _queue.Add(Build.Caption(S.BuildsQueueDrag));
    }
}
