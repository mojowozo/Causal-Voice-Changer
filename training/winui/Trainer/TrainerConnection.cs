// The Python side (python -m causalvc.trainer), run as a child process: commands go to its
// stdin, events come back on its stdout, one JSON object per line (the protocol is described at
// the top of causalvc/trainer.py). Mirrors ../ui/src/trainer.rs. Closing its stdin (this app
// closing) makes it stop a running build after saving a checkpoint.

using System.Diagnostics;
using System.Text;
using System.Text.Json.Nodes;

namespace CausalVC.Training.Trainer;

public sealed class TrainerConnection : IAsyncDisposable
{
    static readonly TimeSpan RestartDelay = TimeSpan.FromSeconds(3);

    readonly Action<TrainerEvent> _onEvent;
    readonly CancellationTokenSource _stop = new();
    readonly object _lock = new();
    Process? _process;
    Task? _loop;

    /// <param name="onEvent">Called on a background thread; marshal to the UI thread.</param>
    public TrainerConnection(Action<TrainerEvent> onEvent) => _onEvent = onEvent;

    public void Start() => _loop = Task.Run(RunAsync);

    public void Send(string cmd, JsonObject? args = null)
    {
        var line = new JsonObject { ["cmd"] = cmd, ["args"] = args ?? new JsonObject() }.ToJsonString();
        lock (_lock)
        {
            try
            {
                if (_process is { HasExited: false } process)
                {
                    process.StandardInput.WriteLine(line);
                    process.StandardInput.Flush();
                }
            }
            catch (Exception error) when (error is IOException or InvalidOperationException)
            {
                // The process is gone; the read loop reports that.
            }
        }
    }

    async Task RunAsync()
    {
        while (!_stop.IsCancellationRequested)
        {
            Process process;
            try
            {
                process = Spawn();
            }
            catch (Exception error)
            {
                _onEvent(new TrainerEvent.Ended(error.Message));
                await Delay();
                continue;
            }
            lock (_lock) _process = process;
            // Read alongside stdout: an unread stderr pipe fills up and blocks Python.
            var errorsRead = process.StandardError.ReadToEndAsync();
            _onEvent(new TrainerEvent.Started());
            string reason = "The trainer stopped unexpectedly.";
            try
            {
                while (await process.StandardOutput.ReadLineAsync(_stop.Token) is { } line)
                    if (TrainerEvent.Parse(line) is { } trainerEvent) _onEvent(trainerEvent);
            }
            catch (OperationCanceledException)
            {
                return;
            }
            catch (IOException error)
            {
                reason = error.Message;
            }
            string errors = "";
            try { errors = await errorsRead; } catch (IOException) { }
            if (errors.Length > 4000) errors = "\u2026" + errors[^4000..];
            lock (_lock) _process = null;
            _onEvent(new TrainerEvent.Ended(string.IsNullOrWhiteSpace(errors) ? reason : $"{reason}\n\n{errors.Trim()}"));
            await Delay();
        }
    }

    async Task Delay()
    {
        try { await Task.Delay(RestartDelay, _stop.Token); } catch (OperationCanceledException) { }
    }

    /// The repository: CAUSALVC_DIR, or the first folder above the exe or the working
    /// directory that holds causalvc\trainer.py.
    public static string? ProjectDir()
    {
        if (Environment.GetEnvironmentVariable("CAUSALVC_DIR") is { Length: > 0 } dir) return dir;
        foreach (var start in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
        {
            for (var folder = new DirectoryInfo(start); folder is not null; folder = folder.Parent)
                if (File.Exists(Path.Combine(folder.FullName, "causalvc", "trainer.py"))) return folder.FullName;
        }
        return null;
    }

    static Process Spawn()
    {
        var root = ProjectDir() ?? throw new InvalidOperationException(
            "Could not find the project folder (causalvc\\trainer.py). Set CAUSALVC_DIR.");
        // run-training.bat keeps the Windows environment in .venv-win (the folder may be shared
        // with Linux, whose environment is .venv).
        var python = new[]
        {
            Path.Combine(root, ".venv-win", "Scripts", "python.exe"),
            Path.Combine(root, ".venv", "Scripts", "python.exe"),
        }.FirstOrDefault(File.Exists) ?? throw new InvalidOperationException(
            $"Python environment not found in {root}\\.venv-win. Run run-training.bat once: it installs everything.");
        var start = new ProcessStartInfo(python)
        {
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardInputEncoding = new UTF8Encoding(false),
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        foreach (var argument in new[] { "-u", "-m", "causalvc.trainer" }) start.ArgumentList.Add(argument);
        start.Environment["PYTHONPATH"] = root;
        start.Environment["PYTHONUTF8"] = "1";
        start.Environment["PYTHONIOENCODING"] = "utf-8";
        return Process.Start(start) ?? throw new InvalidOperationException($"Could not start {python}.");
    }

    public async ValueTask DisposeAsync()
    {
        _stop.Cancel();
        lock (_lock)
        {
            // Closing stdin tells the trainer to save and stop a running build; it then exits.
            try { _process?.StandardInput.Close(); } catch (IOException) { }
            _process = null;
        }
        if (_loop is not null)
        {
            try { await _loop; } catch (OperationCanceledException) { }
        }
    }
}
