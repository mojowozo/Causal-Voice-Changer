// Engine messages and state (docs/engine-protocol.md), parsed leniently: unknown fields are
// ignored and missing optional ones take defaults, so an engine a little newer still works.

using System.Text.Json.Nodes;

namespace CausalVC.Realtime.Engine;

public abstract record EngineEvent
{
    /// Connected; commands can be sent.
    public sealed record Connected : EngineEvent;
    /// The connection dropped or the engine could not be started.
    public sealed record Disconnected(string Reason) : EngineEvent;
    /// Sent once per connection; State is null while the engine starts.
    public sealed record Hello(EngineState? State, string? InitError) : EngineEvent;
    public sealed record Ready(EngineState State) : EngineEvent;
    public sealed record Fatal(string Message) : EngineEvent;
    public sealed record StateChanged(EngineState State) : EngineEvent;
    public sealed record MetersChanged(Meters Meters) : EngineEvent;
    public sealed record Status(string Code, JsonObject Data) : EngineEvent;
    public sealed record Error(ErrorInfo Info) : EngineEvent;
    public sealed record Log(string Text) : EngineEvent;
    /// Failure is set when the command failed (named so it does not clash with the Error event).
    public sealed record Response(long Id, JsonNode? Result, ErrorInfo? Failure) : EngineEvent;

    public static EngineEvent? Parse(string line)
    {
        if (JsonNode.Parse(line) is not JsonObject message) return null;
        if (message["id"] is JsonValue idValue && idValue.TryGetValue(out long id) && message.ContainsKey("ok"))
        {
            bool ok = message["ok"]?.GetValue<bool>() ?? false;
            return ok
                ? new Response(id, message["result"], null)
                : new Response(id, null, ErrorInfo.From(message["error"] as JsonObject));
        }
        var name = message["event"]?.GetValue<string>();
        var data = message["data"];
        return name switch
        {
            "hello" => new Hello(
                data?["state"] is JsonObject state ? EngineState.From(state) : null,
                data?["init_error"]?.GetValue<string>()),
            "ready" when data is JsonObject state => new Ready(EngineState.From(state)),
            "state" when data is JsonObject state => new StateChanged(EngineState.From(state)),
            "fatal" => new Fatal(data?["message"]?.GetValue<string>() ?? "The engine could not start."),
            "meters" when data is JsonObject meters => new MetersChanged(Meters.From(meters)),
            "status" when data is JsonObject status => new Status(status.Str("code"), status),
            "error" when data is JsonObject error => new Error(ErrorInfo.From(error)),
            "log" => new Log(data?["text"]?.GetValue<string>() ?? ""),
            _ => null,
        };
    }
}

public sealed record ErrorInfo(string Code, string Message, JsonObject Details)
{
    public static ErrorInfo From(JsonObject? error) => new(
        error?.Str("code") ?? "internal_error",
        error?.Str("message") ?? "",
        error?["details"] as JsonObject ?? new JsonObject());
}

public sealed class EngineState
{
    public bool Running;
    public string Function = "vc";
    public Settings Settings = new();
    public List<Model> Models = [];
    public List<string> Inputs = [];
    public List<string> Outputs = [];
    public List<Gpu> Gpus = [];
    public int? SampleRate;
    public long? DelayMs;
    public bool Recording;
    public string? FilePath;
    public double? FileDuration;
    public bool FilePlaying;
    public bool Ffmpeg;
    public bool RnNoise;
    /// Hushwire's model: "ready", "needs_download", "downloading", "failed" or "unavailable".
    public string Hushwire = "";
    /// The ASIO device in the selected route, or null when none is ASIO.
    public AsioInfo? Asio;

    public static EngineState From(JsonObject json)
    {
        var devices = json["devices"] as JsonObject;
        var file = json["file"] as JsonObject;
        return new EngineState
        {
            Running = json.Bool("running"),
            Function = json.Str("function", "vc"),
            Settings = Settings.From(json["settings"] as JsonObject ?? new JsonObject()),
            Models = (json["models"] as JsonArray ?? []).OfType<JsonObject>().Select(Model.From).ToList(),
            Inputs = Labels(devices?["inputs"]),
            Outputs = Labels(devices?["outputs"]),
            Gpus = (json["gpus"] as JsonArray ?? []).OfType<JsonObject>()
                .Select(gpu => new Gpu(gpu.Str("id"), gpu.StrOrNull("label"))).ToList(),
            SampleRate = json.IntOrNull("samplerate"),
            DelayMs = json.LongOrNull("delay_ms"),
            Recording = json.Bool("recording"),
            FilePath = file?.StrOrNull("path"),
            FileDuration = file?.DoubleOrNull("duration"),
            FilePlaying = file?.Bool("playing") ?? false,
            Ffmpeg = json.Bool("ffmpeg"),
            RnNoise = json.Bool("rnnoise"),
            Hushwire = json.Str("hushwire"),
            Asio = json["asio"] is JsonObject asio ? AsioInfo.From(asio) : null,
        };
    }

    static List<string> Labels(JsonNode? list) =>
        (list as JsonArray ?? []).OfType<JsonObject>().Select(device => device.Str("label")).ToList();
}

/// What the ASIO driver accepts (null when it could not be asked) and the
/// buffer it is delivering while running, in samples.
public sealed record AsioInfo(AsioSizes? Sizes, int? InUse)
{
    public static AsioInfo From(JsonObject json) => new(
        json["sizes"] is JsonObject sizes
            ? new AsioSizes(sizes.IntOrNull("min") ?? 0, sizes.IntOrNull("max") ?? 0,
                sizes.IntOrNull("preferred") ?? 0, sizes.IntOrNull("granularity") ?? 0)
            : null,
        json.IntOrNull("in_use"));
}

/// Granularity 0: only min, max or the preferred size; -1: powers of two.
public sealed record AsioSizes(int Min, int Max, int Preferred, int Granularity)
{
    public bool Fixed => Min == Max;
}

public sealed class Settings
{
    public string ModelName = "";
    public string Gpu = "auto";
    public string InputSource = "microphone";
    public string InputDevice = "";
    public string OutputDevice = "";
    public string? MonitorDevice;
    public double BlockTime = 0.1;
    /// Samples; 0 keeps the driver's own setting.
    public int AsioBuffer;
    public bool InputDenoise;
    public string InputDenoiseMethod = "hushwire";
    public string LoudnessTarget = "voice_chat";
    public string RecordingFolder = "";
    public string RecordingMode = "separate";
    public double FileInputVolume = 1;
    public double InputGainDb;
    public double OutputGainDb;
    public double MonitorGainDb;
    public double NoiseGateDb = -60;
    public double Pitch;

    public static Settings From(JsonObject json) => new()
    {
        ModelName = json.Str("model_name"),
        Gpu = json.Str("gpu", "auto"),
        InputSource = json.Str("input_source", "microphone"),
        InputDevice = json.Str("input_device"),
        OutputDevice = json.Str("output_device"),
        MonitorDevice = json.StrOrNull("monitor_device"),
        BlockTime = json.DoubleOrNull("block_time") ?? 0.1,
        AsioBuffer = json.IntOrNull("asio_buffer") ?? 0,
        InputDenoise = json.Bool("input_denoise"),
        InputDenoiseMethod = json.Str("input_denoise_method", "hushwire"),
        LoudnessTarget = json.Str("loudness_target", "voice_chat"),
        RecordingFolder = json.Str("recording_folder"),
        RecordingMode = json.Str("recording_mode", "separate"),
        FileInputVolume = json.DoubleOrNull("file_input_volume") ?? 1,
        InputGainDb = json.DoubleOrNull("input_gain_db") ?? 0,
        OutputGainDb = json.DoubleOrNull("output_gain_db") ?? 0,
        MonitorGainDb = json.DoubleOrNull("monitor_gain_db") ?? 0,
        NoiseGateDb = json.DoubleOrNull("noise_gate_db") ?? -60,
        Pitch = json.DoubleOrNull("pitch") ?? 0,
    };
}

/// MergeFamily: voices with the same value can be merged together; null when this one cannot be.
public sealed record Model(string Name, string ModelFile, long SizeBytes, long Modified, string? MergeFamily = null)
{
    public static Model From(JsonObject json) => new(
        json.Str("name"), json.Str("model_file"), json.LongOrNull("size_bytes") ?? 0, json.LongOrNull("modified") ?? 0,
        json.StrOrNull("merge_family"));
}

public sealed record Gpu(string Id, string? Label);

public sealed record Loudness(
    double? Lufs, double TargetLufs, double SpeechSeconds, double OverPercent, string? Verdict, double AdjustDb)
{
    public static Loudness From(JsonObject json) => new(
        json.DoubleOrNull("lufs"),
        json.DoubleOrNull("target_lufs") ?? 0,
        json.DoubleOrNull("speech_seconds") ?? 0,
        json.DoubleOrNull("over_percent") ?? 0,
        json.StrOrNull("verdict"),
        json.DoubleOrNull("adjust_db") ?? 0);
}

public sealed record Meters(
    double Input, double Output, double Monitor, double? InferMs, double? QueuedMs,
    double? RecordingSeconds, double? FilePosition, Loudness? Loudness)
{
    public static readonly Meters Empty = new(0, 0, 0, null, null, null, null, null);

    public static Meters From(JsonObject json) => new(
        json.DoubleOrNull("input") ?? 0,
        json.DoubleOrNull("output") ?? 0,
        json.DoubleOrNull("monitor") ?? 0,
        json.DoubleOrNull("infer_ms"),
        json.DoubleOrNull("queued_ms"),
        json.DoubleOrNull("recording_seconds"),
        json.DoubleOrNull("file_position"),
        json["loudness"] is JsonObject loudness ? Loudness.From(loudness) : null);
}

/// Lenient accessors: a missing, null or mistyped field gives the default.
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

    public static int? IntOrNull(this JsonObject json, string key) => (int?)json.LongOrNull(key);
}
