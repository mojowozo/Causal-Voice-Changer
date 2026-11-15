// Connection to the Python engine (python -m engine.server), as in ../ui/src/engine.rs.
//
// On Windows the engine listens on a loopback TCP port and writes the port and a random
// token to %LOCALAPPDATA%\CausalVC\realtime-engine.json; the first line a client sends is
// {"auth": token}. After that the protocol is newline-delimited JSON, as documented in
// ../docs/engine-protocol.md. The connection starts the engine when none answers, and
// reconnects forever when it drops.

using System.Diagnostics;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace CausalVC.Realtime.Engine;

public sealed class EngineConnection : IAsyncDisposable
{
    /// Seconds the engine keeps running after its last client disconnects.
    const string EngineIdleExitSeconds = "10";
    static readonly TimeSpan SpawnConnectTimeout = TimeSpan.FromSeconds(30);
    static readonly TimeSpan ReconnectDelay = TimeSpan.FromSeconds(1);

    readonly Action<EngineEvent> _post;
    readonly CancellationTokenSource _stop = new();
    readonly object _writeLock = new();
    long _nextId = 1;
    StreamWriter? _writer;
    Task? _loop;

    /// <param name="post">Receives every event; called on a background thread.</param>
    public EngineConnection(Action<EngineEvent> post) => _post = post;

    public static string EndpointFile => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "CausalVC", "realtime-engine.json");

    public void Start() => _loop = Task.Run(() => RunAsync(_stop.Token));

    /// Queue one command; returns its request id, or null when not connected.
    public long? Send(string cmd, JsonObject? args = null)
    {
        lock (_writeLock)
        {
            if (_writer is null) return null;
            long id = _nextId++;
            var line = new JsonObject { ["id"] = id, ["cmd"] = cmd, ["args"] = args ?? new JsonObject() };
            try
            {
                _writer.Write(line.ToJsonString() + "\n");
                _writer.Flush();
            }
            catch (Exception)
            {
                // The read loop notices the broken connection and reports it.
                return null;
            }
            return id;
        }
    }

    async Task RunAsync(CancellationToken stop)
    {
        while (!stop.IsCancellationRequested)
        {
            TcpClient client;
            try
            {
                client = await ConnectAsync(stop);
            }
            catch (OperationCanceledException) { return; }
            catch (Exception error)
            {
                _post(new EngineEvent.Disconnected(error.Message));
                await Delay(ReconnectDelay * 3, stop);
                continue;
            }

            string reason;
            using (client)
            {
                var stream = client.GetStream();
                var reader = new StreamReader(stream, new UTF8Encoding(false));
                lock (_writeLock)
                    _writer = new StreamWriter(stream, new UTF8Encoding(false)) { NewLine = "\n" };
                _post(new EngineEvent.Connected());
                reason = await ReadLoopAsync(reader, stop);
                lock (_writeLock) _writer = null;
            }
            if (stop.IsCancellationRequested) return;
            _post(new EngineEvent.Disconnected(reason));
            await Delay(ReconnectDelay, stop);
        }
    }

    async Task<string> ReadLoopAsync(StreamReader reader, CancellationToken stop)
    {
        try
        {
            while (true)
            {
                string? line = await reader.ReadLineAsync(stop);
                if (line is null) return "The engine closed the connection.";
                if (line.Length == 0) continue;
                EngineEvent? parsed;
                try { parsed = EngineEvent.Parse(line); }
                catch (Exception) { parsed = null; } // a newer engine's message we do not know
                if (parsed is not null) _post(parsed);
            }
        }
        catch (OperationCanceledException) { return "Closed."; }
        catch (Exception error) { return error.Message; }
    }

    static async Task Delay(TimeSpan delay, CancellationToken stop)
    {
        try { await Task.Delay(delay, stop); } catch (OperationCanceledException) { }
    }

    /// Connect to a running engine, or start one and wait until it answers.
    async Task<TcpClient> ConnectAsync(CancellationToken stop)
    {
        var client = await TryConnectAsync(stop);
        if (client is not null) return client;

        var process = SpawnEngine();
        var deadline = DateTime.UtcNow + SpawnConnectTimeout;
        while (true)
        {
            await Task.Delay(150, stop);
            client = await TryConnectAsync(stop);
            if (client is not null) return client;
            if (process.HasExited)
                throw new InvalidOperationException(
                    $"The engine exited during startup (code {process.ExitCode}). Run run-realtime.bat from a terminal to see why.");
            if (DateTime.UtcNow >= deadline)
                throw new TimeoutException($"The engine did not start listening within {SpawnConnectTimeout.TotalSeconds:0} s.");
        }
    }

    static async Task<TcpClient?> TryConnectAsync(CancellationToken stop)
    {
        int port;
        string token;
        try
        {
            using var json = JsonDocument.Parse(await File.ReadAllTextAsync(EndpointFile, stop));
            port = json.RootElement.GetProperty("port").GetInt32();
            token = json.RootElement.GetProperty("token").GetString() ?? "";
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception) { return null; } // no engine yet, or a half-written file

        var client = new TcpClient { NoDelay = true };
        try
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(stop);
            timeout.CancelAfter(TimeSpan.FromSeconds(2));
            await client.ConnectAsync("127.0.0.1", port, timeout.Token);
            var auth = Encoding.UTF8.GetBytes(new JsonObject { ["auth"] = token }.ToJsonString() + "\n");
            await client.GetStream().WriteAsync(auth, stop);
            return client;
        }
        catch (OperationCanceledException) when (!stop.IsCancellationRequested)
        {
            client.Dispose();
            return null;
        }
        catch (SocketException)
        {
            client.Dispose(); // a stale file left by an engine that crashed
            return null;
        }
    }

    /// The realtime/ folder: CAUSALVC_REALTIME_DIR, or the first ancestor of the exe or the
    /// working directory that contains engine\server.py.
    public static string? ProjectDir()
    {
        var configured = Environment.GetEnvironmentVariable("CAUSALVC_REALTIME_DIR");
        if (!string.IsNullOrEmpty(configured)) return configured;
        foreach (var start in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
        {
            for (var dir = new DirectoryInfo(start); dir is not null; dir = dir.Parent)
                if (File.Exists(Path.Combine(dir.FullName, "engine", "server.py")))
                    return dir.FullName;
        }
        return null;
    }

    static Process SpawnEngine()
    {
        var root = ProjectDir() ?? throw new InvalidOperationException(
            "Could not find the project folder (engine\\server.py). Set CAUSALVC_REALTIME_DIR.");
        var repository = Path.GetFullPath(Path.Combine(root, ".."));
        // run-realtime.bat keeps the Windows environment in .venv-win (the folder may be shared
        // with Linux, whose environment is .venv); an environment of the app's own wins.
        string[] candidates =
        [
            Path.Combine(root, ".venv", "Scripts", "python.exe"),
            Path.Combine(repository, ".venv-win", "Scripts", "python.exe"),
            Path.Combine(repository, ".venv", "Scripts", "python.exe"),
        ];
        var python = candidates.FirstOrDefault(File.Exists) ?? throw new InvalidOperationException(
            $"Python environment not found ({candidates[1]}). Run run-realtime.bat once: it installs everything.");

        var start = new ProcessStartInfo(python)
        {
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (var arg in new[] { "-m", "engine.server", "--exit-when-idle", EngineIdleExitSeconds })
            start.ArgumentList.Add(arg);
        // Device names and log text are Unicode; the hidden console's code page is not.
        start.Environment["PYTHONUTF8"] = "1";
        start.Environment["PYTHONIOENCODING"] = "utf-8";
        return Process.Start(start) ?? throw new InvalidOperationException($"Could not start {python}.");
    }

    public async ValueTask DisposeAsync()
    {
        _stop.Cancel();
        if (_loop is not null)
        {
            try { await _loop.WaitAsync(TimeSpan.FromSeconds(1)); } catch (Exception) { }
        }
    }
}
