// What the trainer reports (../ui/src/trainer.rs has the same types), read leniently: a
// missing or odd field falls back to a default rather than failing the whole event.

using System.Text.Json;
using System.Text.Json.Nodes;

namespace CausalVC.Training.Trainer;

public abstract record TrainerEvent
{
    public sealed record Started : TrainerEvent;
    public sealed record Ended(string Reason) : TrainerEvent;
    public sealed record Hello(Form Form, List<string> Bases) : TrainerEvent;
    public sealed record Gpus(List<Gpu> List) : TrainerEvent;
    public sealed record StateChanged(TrainerState State) : TrainerEvent;
    public sealed record Log(string Text) : TrainerEvent;
    public sealed record Error(string Code, string Message, string? Field) : TrainerEvent;

    public static TrainerEvent? Parse(string line)
    {
        JsonObject? message;
        try { message = JsonNode.Parse(line) as JsonObject; }
        catch (JsonException) { return null; }
        if (message?.StrOrNull("event") is not { } name) return null;
        var data = message["data"] as JsonObject ?? new JsonObject();
        return name switch
        {
            "hello" => new Hello(Form.From(data["form"] as JsonObject ?? new JsonObject()), Strings(data["bases"])),
            "gpus" => new Gpus((data["gpus"] as JsonArray ?? []).OfType<JsonObject>()
                .Select(gpu => new Gpu(gpu.Str("id"), gpu.StrOrNull("label"))).ToList()),
            "state" => new StateChanged(TrainerState.From(data)),
            "log" => new Log(data.Str("text")),
            "error" => new Error(data.Str("code"), data.Str("message"), data.StrOrNull("field")),
            _ => null,
        };
    }

    static List<string> Strings(JsonNode? list) =>
        (list as JsonArray ?? []).Select(item => item is JsonValue value && value.TryGetValue(out string? text) ? text : null)
            .OfType<string>().ToList();
}

public sealed record Gpu(string Id, string? Label);

public sealed record Build(string Name, long Steps);

public sealed record Progress(long Step, long Total, string Elapsed, string Remaining, double Rate);

/// The training form as saved from the last build.
public sealed class Form
{
    public string Audio = "";
    public string Name = "";
    public long Batch = 8;
    public long Steps = 10000;
    public long SaveEvery = 10000;
    public string Base = "default";
    public string Device = "";
    public bool Fixes = true;
    public string AppModels = "";

    public static Form From(JsonObject json) => new()
    {
        Audio = json.Str("audio"),
        Name = json.Str("name"),
        Batch = json.LongOrNull("batch") ?? 8,
        Steps = json.LongOrNull("steps") ?? 10000,
        SaveEvery = json.LongOrNull("save_every") ?? 10000,
        Base = json.Str("base", "default"),
        Device = json.Str("device"),
        Fixes = json["fixes"] is not JsonValue || json.Bool("fixes"),
        AppModels = json.Str("app_models"),
    };
}

public sealed class TrainerState
{
    public string Phase = "idle";
    public Build? Current;
    public List<Build> Queue = [];
    public Progress? Progress;
    public long? ExitCode;

    public bool Running => Phase is "starting" or "training" or "stopping";

    public static TrainerState From(JsonObject json)
    {
        static Build? BuildFrom(JsonNode? node) =>
            node is JsonObject build ? new Build(build.Str("name"), build.LongOrNull("steps") ?? 0) : null;
        var progress = json["progress"] as JsonObject;
        return new TrainerState
        {
            Phase = json.Str("phase", "idle"),
            Current = BuildFrom(json["current"]),
            Queue = (json["queue"] as JsonArray ?? []).Select(BuildFrom).OfType<Build>().ToList(),
            Progress = progress is null ? null : new Progress(
                progress.LongOrNull("step") ?? 0, progress.LongOrNull("total") ?? 0,
                progress.Str("elapsed"), progress.Str("remaining"), progress.DoubleOrNull("rate") ?? 0),
            ExitCode = json.LongOrNull("exit_code"),
        };
    }
}

static class JsonAccess
{
    public static string Str(this JsonObject json, string key, string fallback = "") => json.StrOrNull(key) ?? fallback;

    public static string? StrOrNull(this JsonObject json, string key) =>
        json[key] is JsonValue value && value.TryGetValue(out string? text) ? text : null;

    public static bool Bool(this JsonObject json, string key) =>
        json[key] is JsonValue value && value.TryGetValue(out bool flag) && flag;

    public static double? DoubleOrNull(this JsonObject json, string key) =>
        json[key] is JsonValue value && value.TryGetValue(out double number) ? number : null;

    public static long? LongOrNull(this JsonObject json, string key)
    {
        if (json[key] is not JsonValue value) return null;
        if (value.TryGetValue(out long whole)) return whole;
        return value.TryGetValue(out double number) ? (long)Math.Round(number) : null;
    }
}
