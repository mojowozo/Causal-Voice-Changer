// Numeric settings edited with sliders, as in ../ui/src/app.rs (Num).

using System.Globalization;
using CausalVC.Realtime.Engine;

namespace CausalVC.Realtime.Ui;

public enum Num { InputGain, OutputGain, MonitorGain, NoiseGate, Pitch, BlockTime, FileVolume }

public static class NumInfo
{
    static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;

    public static string Key(this Num num) => num switch
    {
        Num.InputGain => "input_gain_db",
        Num.OutputGain => "output_gain_db",
        Num.MonitorGain => "monitor_gain_db",
        Num.NoiseGate => "noise_gate_db",
        Num.Pitch => "pitch",
        Num.BlockTime => "block_time",
        _ => "file_input_volume",
    };

    /// Slider range and step.
    public static (double Min, double Max, double Step) Range(this Num num) => num switch
    {
        Num.InputGain or Num.OutputGain or Num.MonitorGain => (-24, 24, 0.5),
        Num.NoiseGate => (-60, 0, 1),
        // 0.1-semitone steps allow fine tuning such as +11.5.
        Num.Pitch => (-24, 24, 0.1),
        // Whole 20 ms frames; delay is the model's 60 ms plus one block.
        Num.BlockTime => (0.02, 0.2, 0.02),
        _ => (0, 1, 0.01),
    };

    /// Changing these stops a running stream, so they are sent once the slider settles.
    public static bool NeedsRestart(this Num num) => num == Num.BlockTime;

    public static double Get(this Num num, Settings settings) => num switch
    {
        Num.InputGain => settings.InputGainDb,
        Num.OutputGain => settings.OutputGainDb,
        Num.MonitorGain => settings.MonitorGainDb,
        Num.NoiseGate => settings.NoiseGateDb,
        Num.Pitch => settings.Pitch,
        Num.BlockTime => settings.BlockTime,
        _ => settings.FileInputVolume,
    };

    public static void Set(this Num num, Settings settings, double value)
    {
        switch (num)
        {
            case Num.InputGain: settings.InputGainDb = value; break;
            case Num.OutputGain: settings.OutputGainDb = value; break;
            case Num.MonitorGain: settings.MonitorGainDb = value; break;
            case Num.NoiseGate: settings.NoiseGateDb = value; break;
            case Num.Pitch: settings.Pitch = value; break;
            case Num.BlockTime: settings.BlockTime = value; break;
            default: settings.FileInputVolume = value; break;
        }
    }

    /// The value as shown beside its slider.
    public static string Format(this Num num, double value) => num switch
    {
        Num.Pitch => value.ToString("+0.0;-0.0;0.0", Invariant),
        Num.NoiseGate => value <= -60 ? S.Off : $"{value.ToString("0", Invariant)} dB",
        Num.BlockTime => $"{value.ToString("0.00", Invariant)} s",
        Num.FileVolume => $"{(value * 100).ToString("0", Invariant)}%",
        _ => $"{value.ToString("+0.0;-0.0;0.0", Invariant)} dB",
    };

    /// The value as plain text for the edit field (no unit or sign padding).
    public static string EditText(this Num num, double value)
    {
        double shown = num == Num.FileVolume ? value * 100 : value;
        return Math.Round(shown, 3).ToString("0.###", Invariant);
    }

    /// Parse a typed value: units, a leading +, and a decimal comma are accepted, "off" means
    /// -60 dB for the noise gate, and the file volume is typed in percent. The result is
    /// clamped to the slider's range.
    public static double? Parse(this Num num, string text)
    {
        var (min, max, _) = num.Range();
        text = text.Trim().ToLowerInvariant();
        if (num == Num.NoiseGate && (text == "off" || text == S.Off.ToLowerInvariant())) return min;
        text = text.TrimEnd('%', 's');
        if (text.EndsWith("db")) text = text[..^2];
        text = text.Trim().TrimStart('+').Replace(',', '.');
        if (!double.TryParse(text, NumberStyles.Float, Invariant, out double value) || !double.IsFinite(value))
            return null;
        if (num == Num.FileVolume) value /= 100;
        return Math.Clamp(value, min, max);
    }

    public static string? Description(this Num num) => num switch
    {
        Num.Pitch => S.PitchDetail,
        Num.InputGain => S.InputGainDetail,
        Num.OutputGain => S.OutputGainDetail,
        Num.MonitorGain => S.MonitorGainDetail,
        Num.NoiseGate => S.NoiseGateDetail,
        Num.BlockTime => S.ChunkDetail,
        _ => S.FileVolumeDetail,
    };
}
