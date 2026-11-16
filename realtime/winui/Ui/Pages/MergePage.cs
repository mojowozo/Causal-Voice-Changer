using System.Text.Json.Nodes;
using CausalVC.Realtime.Engine;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CausalVC.Realtime.Ui.Pages;

/// Blend voices into a new one (pages.rs: merge_page): a voice and a strength per slot, and the
/// new voice's name. Only voices the engine says can be merged are offered.
public sealed class MergePage : PageBase
{
    /// One voice and its strength. Its controls and row are made once and re-added when slots
    /// come and go: an element can only have one parent, so it is never put in a second row.
    sealed class Slot
    {
        public readonly ComboBox Voice = new() { Width = 220, PlaceholderText = S.MergeSlotEmpty };
        public readonly Slider Strength = new() { Minimum = 1, Maximum = 100, StepFrequency = 1, Value = 50, Width = 160 };
        public readonly TextBlock Value = new() { Width = 32, VerticalAlignment = VerticalAlignment.Center };
        public readonly Button Remove;
        public readonly SettingRow Row;

        public Slot(Action<Slot> remove)
        {
            Remove = Build.IconButton("\uE738", S.MergeRemoveSlot, (_, _) => remove(this));
            Row = new SettingRow("", null, Build.Row(Voice, Strength, Value, Remove));
        }
    }

    readonly List<Slot> _slots = [];
    readonly SettingsSection _voices = new(S.MergeVoices);
    readonly TextBox _name = new() { PlaceholderText = S.MergeNamePlaceholder, Width = 260 };
    readonly Button _merge;
    readonly TextBlock _problem = Build.Caption("");
    readonly TextBlock _notEnough = Build.Text(S.MergeNotEnough);
    readonly Button _addSlot;
    List<string> _labels = [];
    List<string> _families = [];
    bool _merging;

    public MergePage(IHost host) : base(host)
    {
        _merge = Build.Button(S.MergeButton, (_, _) => Merge(), null, "AccentButtonStyle");
        _addSlot = Build.Button(S.MergeAddSlot, (_, _) => AddSlot(), "\uE710");
        _name.TextChanged += (_, _) => Refresh();
        _name.KeyDown += (_, args) =>
        {
            if (args.Key == Windows.System.VirtualKey.Enter) Merge();
        };
        var result = new SettingsSection(S.MergeResult)
            .Add(new SettingRow(S.MergeName, S.MergeNameDetail, _name))
            .Add(Build.Row(_merge, _problem));

        Column.Children.Add(Build.Text(S.MergeIntro));
        Column.Children.Add(_notEnough);
        Column.Children.Add(_voices);
        Column.Children.Add(result);
        AddSlot();
        AddSlot();
    }

    public override void Update(EngineState state)
    {
        var mergeable = state.Models.Where(model => model.MergeFamily is not null).ToList();
        var labels = mergeable.Select(model => model.Name).ToList();
        if (!labels.SequenceEqual(_labels))
        {
            _labels = labels;
            foreach (var slot in _slots)
            {
                var chosen = slot.Voice.SelectedItem as string;
                slot.Voice.ItemsSource = _labels.ToList();
                slot.Voice.SelectedIndex = chosen is null ? -1 : _labels.IndexOf(chosen);  // -1: deleted or renamed
            }
        }
        _families = mergeable.Select(model => model.MergeFamily!).ToList();
        _notEnough.Visibility = _labels.Count < 2 ? Visibility.Visible : Visibility.Collapsed;
        Refresh();
    }

    /// The engine answered the merge (MainWindow): a new name next time if it worked.
    public void MergeFinished(bool ok)
    {
        _merging = false;
        if (ok) _name.Text = "";
        Refresh();
    }

    void AddSlot()
    {
        var slot = new Slot(removed =>
        {
            if (_slots.Count <= 2) return;
            _slots.Remove(removed);
            RenderSlots();
        });
        slot.Voice.ItemsSource = _labels.ToList();
        slot.Voice.SelectionChanged += (_, _) => Refresh();
        slot.Strength.ValueChanged += (_, _) => Refresh();
        _slots.Add(slot);
        RenderSlots();
    }

    void RenderSlots()
    {
        _voices.Clear();
        for (int index = 0; index < _slots.Count; index++)
        {
            var slot = _slots[index];
            slot.Row.Title = S.MergeSlot(index + 1);
            slot.Remove.Visibility = _slots.Count > 2 ? Visibility.Visible : Visibility.Collapsed;
            _voices.Add(slot.Row);
        }
        _voices.Add(_addSlot);
        Refresh();
    }

    string? FamilyOf(string? name)
    {
        int index = name is null ? -1 : _labels.IndexOf(name);
        return index < 0 ? null : _families[index];
    }

    void Refresh()
    {
        var chosen = _slots.Where(slot => slot.Voice.SelectedItem is string).ToList();
        double total = chosen.Sum(slot => slot.Strength.Value);
        string? first = FamilyOf(chosen.FirstOrDefault()?.Voice.SelectedItem as string);
        foreach (var slot in _slots)
        {
            slot.Value.Text = $"{slot.Strength.Value:0}";
            slot.Row.Description = slot.Voice.SelectedItem is not string name ? S.MergeSlotEmpty
                : FamilyOf(name) != first ? S.MergeSlotIncompatible
                : S.MergeShare(slot.Strength.Value / total * 100);
        }
        var problem = Problem(chosen, first);
        _problem.Text = problem ?? "";
        _merge.Content = _merging ? S.MergeRunning : S.MergeButton;
        _merge.IsEnabled = problem is null && !_merging;
    }

    string? Problem(List<Slot> chosen, string? first)
    {
        var names = chosen.Select(slot => (slot.Voice.SelectedItem as string ?? "")).ToList();
        if (names.Count < 2) return S.MergeNeedTwo;
        if (names.Distinct().Count() < names.Count) return S.MergeDuplicate;
        if (names.Any(name => FamilyOf(name) != first)) return S.MergeIncompatible;
        string newName = _name.Text.Trim();
        if (newName.Length == 0) return S.MergeNeedName;
        if (Host.State?.Models.Any(model => string.Equals(model.Name, newName, StringComparison.OrdinalIgnoreCase)) == true)
            return S.ErrorModelNameTaken;
        return null;
    }

    void Merge()
    {
        var chosen = _slots.Where(slot => slot.Voice.SelectedItem is string).ToList();
        if (_merging || Problem(chosen, FamilyOf(chosen.FirstOrDefault()?.Voice.SelectedItem as string)) is not null) return;
        var voices = new JsonArray();
        foreach (var slot in chosen)
            voices.Add(new JsonObject { ["name"] = (slot.Voice.SelectedItem as string ?? ""), ["strength"] = slot.Strength.Value });
        if (Host.Send("merge_models", new JsonObject { ["voices"] = voices, ["name"] = _name.Text.Trim() }) is long id)
        {
            _merging = true;
            MergeRequested?.Invoke(id);
            Refresh();
        }
    }

    /// The request id of a merge just sent, so the window can route its answer here.
    public event Action<long>? MergeRequested;
}
