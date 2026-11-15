// Model library logic, as in ../ui/src/library.rs: model folder names such as
// "Sakura-shortstream-50m-1.6_200e_14200s" are split into the voice, a description of the
// run, and the checkpoint's epochs, so checkpoints of one voice are grouped together.

using CausalVC.Realtime.Engine;

namespace CausalVC.Realtime.Ui;

public readonly record struct ModelName(string Voice, string Detail, uint? Epochs);

public static class Library
{
    /// Split "Voice-detail_200e_14200s" into voice, detail and epochs. Names that do not
    /// follow the pattern become their own voice.
    public static ModelName Parse(string name)
    {
        string baseName = name;
        uint? epochs = null;
        // A trailing _<n>e_<n>s (epochs and steps).
        int last = name.LastIndexOf('_');
        int previous = last > 0 ? name.LastIndexOf('_', last - 1) : -1;
        if (previous >= 0)
        {
            var epoch = Number(name[(previous + 1)..last], 'e');
            var steps = Number(name[(last + 1)..], 's');
            if (epoch is not null && steps is not null)
            {
                baseName = name[..previous];
                epochs = epoch;
            }
        }
        int split = baseName.IndexOfAny(['-', '_', ' ']);
        return split > 0
            ? new ModelName(baseName[..split], baseName[(split + 1)..], epochs)
            : new ModelName(baseName, "", epochs);
    }

    static uint? Number(string text, char unit)
    {
        if (text.Length < 2 || text[^1] != unit) return null;
        var digits = text[..^1];
        return digits.All(char.IsAsciiDigit) && uint.TryParse(digits, out uint value) ? value : null;
    }

    /// Every whitespace-separated word of the query appears in the name.
    public static bool Matches(string name, string query)
    {
        name = name.ToLowerInvariant();
        return query.ToLowerInvariant()
            .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            .All(name.Contains);
    }

    /// Models matching the query, grouped by voice; newest sorts groups and their models by
    /// date (newest first), otherwise by name and epochs.
    public static List<(string Voice, List<Model> Models)> Grouped(IEnumerable<Model> models, string query, bool newest)
    {
        var groups = new List<(string Voice, List<Model> Models)>();
        foreach (var model in models.Where(model => Matches(model.Name, query)))
        {
            var voice = Parse(model.Name).Voice;
            int at = groups.FindIndex(group => string.Equals(group.Voice, voice, StringComparison.OrdinalIgnoreCase));
            if (at >= 0) groups[at].Models.Add(model);
            else groups.Add((voice, [model]));
        }
        // OrderBy is a stable sort, like the Rust front end's.
        var sorted = groups.Select(group => (group.Voice, newest
            ? group.Models.OrderByDescending(model => model.Modified).ToList()
            : group.Models
                .OrderBy(model => Parse(model.Name).Detail.ToLowerInvariant(), StringComparer.Ordinal)
                .ThenBy(model => Parse(model.Name).Epochs)
                .ToList()));
        return newest
            ? sorted.OrderByDescending(group => group.Item2.Max(model => model.Modified)).ToList()
            : sorted.OrderBy(group => group.Voice.ToLowerInvariant(), StringComparer.Ordinal).ToList();
    }

    public static string FormatSize(long bytes)
    {
        double megabytes = bytes / 1_000_000.0;
        return megabytes >= 1000
            ? $"{(megabytes / 1000).ToString("0.0", System.Globalization.CultureInfo.InvariantCulture)} GB"
            : $"{megabytes:0} MB";
    }

    public static string FormatDate(long unixSeconds) =>
        DateTimeOffset.FromUnixTimeSeconds(unixSeconds).UtcDateTime.ToString("yyyy-MM-dd");

    /// A voice: a .cvc file, or an installed voice's description (.nextvc.json is the earlier spelling).
    public static bool IsModelFile(string path)
    {
        var lower = path.ToLowerInvariant();
        return lower.EndsWith(".cvc") || lower.EndsWith(".causalvc.json") || lower.EndsWith(".nextvc.json");
    }
}
