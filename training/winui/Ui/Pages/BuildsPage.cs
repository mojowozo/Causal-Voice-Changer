using System.Text.Json.Nodes;
using CausalVC.Training.Trainer;

namespace CausalVC.Training.Ui.Pages;

/// The build in progress and the queue after it (../ui/src/pages.rs: builds_page).
public sealed class BuildsPage : PageBase
{
    readonly SettingsSection _current = new(S.BuildsCurrent);
    readonly SettingsSection _queue = new(S.BuildsQueue);
    string _rendered = "";

    public BuildsPage(IHost host) : base(host)
    {
        Column.Children.Add(_current);
        Column.Children.Add(_queue);
        var note = Build.Caption(S.CloseNote);
        note.Margin = new Microsoft.UI.Xaml.Thickness(4, 12, 4, 0);
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
        if (state.Queue.Count == 0) _queue.Add(Build.Text(S.BuildsQueueEmpty));
        for (int index = 0; index < state.Queue.Count; index++)
        {
            int position = index;
            var queued = state.Queue[index];
            _queue.Add(new SettingRow($"{index + 1}. {queued.Name}", S.BuildsSteps(queued.Steps),
                Build.Button(S.Remove, (_, _) => Host.Send("remove", new JsonObject { ["index"] = position }), "")));
        }
    }
}
