// English text, from the Linux front end's ../ui/i18n/en/*.ftl (same wording, so both
// front ends read alike). Windows-specific lines are marked.

namespace CausalVC.Realtime.Ui;

static class S
{
    public const string AppTitle = "CausalVC Realtime";

    // Drag and drop
    public const string DropTitle = "Drop to add voices";
    public const string DropDetail = "Each .cvc file becomes a voice. These are written when a voice is built, and by Export in the library.";
    public static string ModelImported(string names) => $"Added model: {names}";

    // Typed values
    public const string ValueClickToType = "Click to type a value";
    public const string ValueInvalid = "Enter a number";

    // Navigation
    public const string PageModel = "Model";
    public const string PageLibrary = "Library";
    public const string PageAudio = "Audio";
    public const string PagePerformance = "Performance";
    public const string PageRecording = "Recording";
    public const string PageLog = "Log";

    // Engine connection
    public const string EngineStarting = "Starting the engine…";
    public const string EngineStartingDetail = "Loading PyTorch, models, and audio devices.";
    public const string EngineUnavailable = "The engine is not available";
    public const string EngineRetrying = "Retrying automatically.";

    // Model page
    public const string SectionModels = "Models";
    public const string Model = "Model";
    public const string ModelsEmpty = "No voices yet. Build one with the training window, or drag a voice's .cvc file onto this window to add it.";
    public static string ModelFiles(string model) => $"Voice file: {model}";
    public const string Reload = "Reload";
    public const string SectionVoice = "Voice";
    public const string Pitch = "Pitch (semitones)";
    public const string PitchDetail = "How far to shift your voice: +12 is one octave up.";
    public const string OutputLevel = "Output level";
    public const string OutputLevelIdle = "While converting, shows how loud you sound to others. Voice chat aims for -21 LUFS; streaming for -16 LUFS, the usual level of streamed voice.";
    public const string OutputLevelListening = "Talk to measure your level.";
    public static string OutputLevelMeasuring(string lufs, string seconds) => $"{lufs} LUFS so far; measuring ({seconds} of 5 s of speech)";
    public static string OutputLevelReading(string lufs, string target, string verdict) => $"{lufs} LUFS (target {target}): {verdict}";
    public const string LoudnessTargetVoiceChat = "Voice chat (-21)";
    public const string LoudnessTargetStreaming = "Streaming (-16)";
    public const string OutputLevelGood = "a good level.";
    public const string OutputLevelQuiet = "too quiet; raise the output gain.";
    public const string OutputLevelQuietPeaky = "quiet, but your peaks already reach full scale, so raising the gain would squash them.";
    public const string OutputLevelLoud = "too loud; lower the output gain.";
    public const string OutputLevelSquashed = "too loud; peaks are being squashed. Lower the output gain.";
    public const string OutputLevelApply = "Apply";
    public const string LevelGood = "Level: good";
    public const string LevelQuiet = "Level: too quiet";
    public const string LevelLoud = "Level: too loud";
    public const string LevelMeasuring = "Level: measuring…";
    public const string SectionLevels = "Levels";
    public const string InputGain = "Input gain";
    public const string InputGainDetail = "Boosts or cuts your mic before conversion. The converted voice follows your input level, so this also changes how loud you come out.";
    public const string OutputGain = "Output gain";
    public const string OutputGainDetail = "Volume of the converted voice on the output device, which is what others hear. Output level below suggests a value.";
    public const string MonitorGain = "Monitor gain";
    public const string MonitorGainDetail = "Volume of the copy you hear yourself on the monitor device. Doesn't change what others hear.";
    public const string NoiseGate = "Noise gate";
    public const string NoiseGateDetail = "Mutes your mic whenever it's quieter than this, so background sound between words isn't converted. Off at -60 dB.";
    public const string Off = "Off";
    public const string ResetModelSettings = "Reset model settings";

    // Library
    public const string LibrarySearch = "Search models";
    public const string LibrarySortName = "By name";
    public const string LibrarySortNewest = "Newest first";
    public static string LibraryCount(int shown, int total) => $"{shown} of {total} models";
    public const string LibraryNoMatch = "No models match your search.";
    public static string LibraryVoice(string voice, int count) => $"{voice} ({count})";
    public static string LibraryCheckpoint(string detail, uint epochs) => $"{detail} · {epochs} epochs";
    public static string LibraryEpochs(uint epochs) => $"{epochs} epochs";
    public const string LibraryUse = "Use";
    public const string LibrarySelected = "Selected";
    public const string LibraryOpenFolder = "Open folder";
    public const string LibraryExport = "Export as one .cvc file, to share";
    public const string VoiceFileType = "CausalVC voice";
    public static string StatusModelExported(string name, string path) => $"Exported {name} to {path}";
    // Windows: the Recycle Bin rather than the Trash.
    public const string LibraryDelete = "Move to the Recycle Bin";
    public const string LibraryDeleteInUse = "Stop converting to delete this model";
    public const string DeleteModelTitle = "Move model to the Recycle Bin?";
    public static string DeleteModelBody(string name) => $"{name} and its saved settings (pitch, gains) will be moved to the Recycle Bin. You can restore it from there.";
    public const string DeleteModelConfirm = "Move to Recycle Bin";
    public const string Cancel = "Cancel";
    public static string StatusModelDeleted(string name) => $"Moved {name} to the Recycle Bin";
    public const string ErrorModelInUse = "Stop converting before deleting this model.";
    public static string ErrorDeleteFailed(string detail) => $"Could not move the model to the Recycle Bin: {detail}";
    public const string LibraryRename = "Rename";
    public const string LibraryRenameInUse = "Stop converting to rename this model";
    public const string RenameModelTitle = "Rename model";
    public const string RenameModelBody = "This renames the model's folder; its saved settings stay with it.";
    public const string RenameModelConfirm = "Rename";
    public static string StatusModelRenamed(string name, string newName) => $"Renamed {name} to {newName}";
    public const string ErrorModelNameTaken = "Another model already has that name.";
    public const string ErrorBadModelName = "That name can't be used: it can't be empty, start with a dot, or contain / \\ : * ? \" < > |";
    public static string ErrorRenameFailed(string detail) => $"Could not rename the model: {detail}";

    // Audio page
    public const string SectionInput = "Input";
    public const string InputSource = "Source";
    public const string InputSourceDetail = "Convert your microphone live, or play an audio file through the model.";
    public const string SourceMicrophone = "Microphone / audio device";
    public const string SourceFile = "Audio file";
    public const string AudioFile = "File";
    public const string NoFile = "No file selected";
    public const string Browse = "Browse…";
    public const string FileVolume = "File volume";
    public const string FileVolumeDetail = "Playback volume of the converted file.";
    public const string InputDevice = "Input device";
    public const string InputDeviceDetail = "Your microphone.";
    public const string SectionOutput = "Output";
    public const string OutputDevice = "Output device";
    public const string OutputDeviceDetail = "Where the converted voice goes, usually the virtual mic that Discord or OBS listens to.";
    public const string MonitorDevice = "Monitor device";
    public const string MonitorDeviceDetail = "An optional second output so you can hear yourself, such as your headphones.";
    public const string MonitorDisabled = "Disabled";
    public const string ReloadDevices = "Reload devices";
    public static string SampleRateValue(int rate) => $"Sample rate: {rate} Hz";
    public const string SampleRateIdle = "Sample rate: — (not running)";
    // Windows in place of the Linux JACK hint.
    public const string DeviceHint = "[ASIO] devices give the lowest latency; set their buffer size in the driver's control panel. [WASAPI] works with every device.";

    // Performance page
    public const string SectionBuffering = "Buffering";
    public const string Chunk = "Chunk";
    public const string ChunkDetail = "How much audio is converted at a time, in steps of 20 ms. Total delay is the model's fixed 60 ms plus one chunk; shorter chunks lower the delay but can stutter if the GPU can't keep up (watch Inference in the bottom bar). Changing it restarts conversion.";
    public const string SectionInference = "Inference";
    public const string Gpu = "GPU";
    public const string GpuDetail = "The graphics card that runs the model. Changing it restarts conversion.";
    public const string GpuAutomatic = "Automatic (recommended)";
    public const string SectionNoise = "Noise reduction";
    public const string InputDenoise = "Input noise reduction";
    public const string InputDenoiseDetail = "Removes background noise like keyboards, fans and traffic from your mic before conversion.";
    public const string DenoiseMethod = "Noise reduction model";
    public const string DenoiseMethodHushwire = "Hushwire (Amazon Voice Focus)";
    public const string DenoiseMethodRnnoise = "RNNoise";
    public const string DenoiseHushwireReady = "Strongest cleanup, keeps the voice natural. Adds about 55 ms of delay.";
    public const string DenoiseHushwireNeedsDownload = "Hushwire's model comes from Amazon and is downloaded once, after you accept. Until then your mic isn't denoised.";
    public const string DenoiseHushwireDownloading = "Downloading Hushwire's model\u2026";
    public const string DenoiseHushwireFailed = "Hushwire's model could not be downloaded. Check your connection and try again.";
    public const string DenoiseHushwireUnavailable = "Hushwire needs the wasmtime package, which is missing; run run-realtime.bat again to reinstall it.";
    public const string DenoiseRnnoiseReady = "Lighter cleanup with less delay: adds 20 ms.";
    public const string DenoiseHushwireDownload = "Download model\u2026";
    public const string HushwireNoticeTitle = "Download Hushwire's model?";
    public const string HushwireNoticeBody = "Hushwire's noise reduction model is Amazon Voice Focus, which is Amazon's proprietary software and is not included with CausalVC. Downloading fetches it from Amazon's servers (static.sdkassets.chime.aws) to this computer. Its use is subject to Amazon's terms. Neither Hushwire nor CausalVC is affiliated with or endorsed by Amazon.";
    public const string HushwireNoticeAccept = "Accept and download";
    // Windows: RNNoise comes with the pyrnnoise package, so it is only missing after a broken install.
    public const string InputDenoiseMissing = "Removes background noise from your mic before conversion. RNNoise is not installed, so this has no effect; run run-realtime.bat again to reinstall it.";
    public const string ResetPerformance = "Reset performance settings";

    // Recording page
    public const string SectionRecording = "Recording";
    public const string RecordingMode = "Mode";
    public const string RecordingModeDetail = "Recordings capture your raw mic and the converted voice; choose how they're saved.";
    public const string RecordingSeparate = "Separate files (input + output)";
    public const string RecordingMix = "Mix (input + output)";
    public const string RecordingStereo = "Split L/R (left: input / right: output)";
    public const string SaveFolder = "Save folder";
    public const string Change = "Change…";
    public const string OpenFolder = "Open folder";
    public const string Record = "Record";
    public const string StopRecording = "Stop recording";
    public const string RecordHint = "Start conversion before recording.";

    // Log page
    public const string SaveLog = "Save log";
    public const string ClearLog = "Clear log";
    public static string LogSaved(string path) => $"Log saved: {path}";

    // Control bar
    public const string Start = "Start";
    public const string Starting = "Starting…";
    public const string Converting = "Converting";
    public const string Passthrough = "Passthrough";
    public const string PassthroughActive = "Passthrough active";
    public const string MeterIn = "In";
    public const string MeterOut = "Out";
    public const string MeterMon = "Mon";
    public static string Latency(string value) => $"Latency: {value} ms";
    public const string LatencyDetail = "Delay from your mic to the output: chunk and device buffers, plus the audio queued right now. If it jumps, the engine fell behind for a moment; it catches up during your next pause. The real end-to-end delay is still somewhat higher.";
    public static string Inference(string value) => $"Inference: {value} ms";
    public const string InferenceDetail = "Time to convert each chunk. It must stay well below the chunk length, or the audio stutters.";

    // Engine status
    public const string StatusPreparing = "Preparing…";
    public const string StatusImportingModel = "Adding model…";
    public static string StatusRecordingSaved(string paths) => $"Recording saved:\n{paths}";

    public static string StatusText(string code, System.Text.Json.Nodes.JsonObject data)
    {
        string Text(string key) => data[key] is System.Text.Json.Nodes.JsonValue value && value.TryGetValue(out string? text) ? text : "";
        return code switch
        {
            "preparing" => StatusPreparing,
            "loading_model" => "Loading model…",
            "preparing_inference" => "Preparing inference…",
            "starting_audio" => "Starting audio devices…",
            "conversion_started" => "Conversion started",
            "passthrough_started" => "Passthrough started",
            "conversion_stopped" => "Conversion stopped",
            "passthrough_stopped" => "Passthrough stopped",
            "settings_changed" => "Stopped for a settings change",
            "stream_stopped" => "The audio device stopped unexpectedly",
            "file_selected" => $"Audio file selected: {Text("name")}",
            "playing" => "Playing",
            "paused" => "Paused",
            "playback_stopped" => "Playback stopped",
            "recording_started" => "Recording started",
            "importing_model" => StatusImportingModel,
            "model_deleted" => StatusModelDeleted(Text("name")),
            "model_renamed" => StatusModelRenamed(Text("name"), Text("new_name")),
            "model_exported" => StatusModelExported(Text("name"), Text("path")),
            "hushwire_downloaded" => "Hushwire's model is downloaded; noise reduction is ready",
            // The toast from the import response names the model.
            "model_imported" => "",
            "recording_saved" => StatusRecordingSaved(string.Join("\n",
                (data["paths"] as System.Text.Json.Nodes.JsonArray ?? [])
                    .Select(path => path?.GetValue<string>() ?? ""))),
            _ => code,
        };
    }

    public static string ErrorText(Engine.ErrorInfo error) => error.Code switch
    {
        "no_model" => "Select a model from the models folder.",
        "model_file_missing" => "The model file was not found.",
        "no_audio_file" => "Select an audio file first.",
        // Windows: where to get FFmpeg.
        "ffmpeg_missing" => "FFmpeg was not found. Install it (winget install Gyan.FFmpeg) and restart, or put ffmpeg.exe in realtime\\tools\\ffmpeg.",
        "record_requires_running" => RecordHint,
        "no_common_samplerate" => "The input, output, and monitor devices have no common sample rate. Set them to the same rate (usually 48 kHz).",
        "audio_start_failed" => $"Could not start the audio devices: {error.Message}",
        "asio_one_device" => "ASIO can only use one device at a time. Choose the same ASIO device for input and output, or a [WASAPI] device for one of them.",
        "monitor_failed" => $"The monitor device could not start; continuing on the main output only. ({error.Message})",
        "not_ready" => "The engine is still starting.",
        "import_no_model_file" => "Drop a voice's .cvc file (written when a voice is built).",
        "export_failed" => $"Could not export the voice: {error.Message}",
        "hushwire_download_failed" => $"Could not download Hushwire's model: {error.Message}",
        "hushwire_unavailable" => "Hushwire needs the wasmtime package; run run-realtime.bat again to reinstall it.",
        "import_failed" => $"Could not add the model: {error.Message}",
        "model_load_failed" => $"Could not load the model: {error.Message}",
        "model_in_use" => ErrorModelInUse,
        "delete_failed" => ErrorDeleteFailed(error.Message),
        "model_name_taken" => ErrorModelNameTaken,
        "bad_model_name" => ErrorBadModelName,
        "rename_failed" => ErrorRenameFailed(error.Message),
        _ => error.Message,
    };
}
