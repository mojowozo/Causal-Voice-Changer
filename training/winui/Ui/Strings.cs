// The app's text, in English: the same wording as the Linux front end's
// ../ui/i18n/en/causalvc_training.ftl.

using CausalVC.Training.Trainer;

namespace CausalVC.Training.Ui;

static class S
{
    public const string AppTitle = "CausalVC Trainer";

    public const string PageVoice = "New voice";
    public const string PageBuilds = "Builds";
    public const string PageLog = "Log";

    public const string ServiceStarting = "Starting…";
    public const string ServiceStartingDetail = "Loading the training tools.";
    public const string ServiceUnavailable = "The trainer could not start";
    public const string ServiceRetrying = "Trying again in a few seconds. The message below says what went wrong.";

    public const string SectionDataset = "Dataset";
    public const string Audio = "Recordings";
    public const string AudioDetail = "A folder of the target's speech, or one long recording. Any format and sample rate; pauses can stay in.";
    public const string AudioPlaceholder = "Paste a path, or choose a folder or file";
    public const string ChooseFolder = "Folder…";
    public const string ChooseFile = "File…";
    public const string VoiceName = "Voice name";
    public const string VoiceNameDetail = "Letters, digits, '-', '_' or '.'. Building the same name again picks up where it left off.";

    public const string SectionTraining = "Training";
    public const string Steps = "Steps";
    public const string StepsDetail = "How long to train. 10,000 is a good start; more steps follow the voice more closely.";
    public const string SaveEvery = "Save a version every";
    public const string SaveEveryDetail = "Steps between saved versions (NAME-1000s, NAME-2000s and so on), to compare and keep the best.";
    public const string Batch = "Batch size";
    public const string BatchDetail = "Clips trained on at once. Lower it if the GPU runs out of memory.";
    public const string Restart = "Start over";
    public const string RestartDetail = "Ignore this voice name's earlier progress and train from the beginning. Applies to the next build only.";

    public const string SectionHardware = "Starting point and hardware";
    public const string Base = "Base";
    public const string BaseDetail = "The voice model training starts from. Your own bases go in the custom-base folder.";
    public const string BaseDefault = "Default";
    public const string Gpu = "GPU";
    public const string GpuDetail = "The graphics card that trains the voice.";
    public const string GpuLoading = "Looking for GPUs…";
    public const string GpuCpu = "No GPU found (very slow)";

    public const string SectionInstall = "Voice changer";
    public const string Install = "Add to the voice changer";
    public const string InstallDetail = "Every saved version appears in the voice changer's library as it is written.";
    public const string InstallFolder = "Voices folder";
    public const string Change = "Change…";

    public const string Start = "Start training";
    public const string QueueBuild = "Add to queue";
    public static string BuildQueued(string name, int position) =>
        $"{name} added to the queue ({(position == 1 ? "next up" : $"#{position} in line")}). See it on the Builds page.";
    public const string Stop = "Stop";
    public const string EndNow = "End now";
    public const string StopDetail = "Stop saves a checkpoint first; building the same voice again resumes. Click again to end at once without saving.";

    public static string Progress(long step, long total) => $"Step {step:N0} of {total:N0}";
    public static string ProgressTimes(string elapsed, string remaining, double rate) =>
        $"{elapsed} elapsed · {remaining} left · {rate:0.00} steps/s";

    public const string BuildsCurrent = "Now";
    public const string BuildsNone = "Nothing is training.";
    public const string BuildsQueue = "Up next";
    public const string BuildsQueueEmpty = "The queue is empty. Use “Add to queue” to train several voices one after another.";
    public static string BuildsSteps(long steps) => $"{steps:N0} steps";
    public const string Remove = "Remove";
    public const string BuildNow = "Build now";
    public const string BuildsQueueDrag = "Drag a build to change the order. Build now saves the current build and puts it first in line, to resume after.";
    public const string CloseNote = "Closing this window stops training the same way Stop does: a checkpoint is saved first.";

    public const string LogEmpty = "Messages from training appear here.";
    public const string ClearLog = "Clear";

    public static string Phase(TrainerState state)
    {
        string name = state.Current?.Name ?? "";
        return state.Phase switch
        {
            "starting" => $"Starting {name} (loading the models takes a little while)…",
            "training" => $"Training {name}",
            "stopping" when state.Next is { } next => $"Saving {name} to switch to {next.Name}; {name} resumes next…",
            "stopping" => $"Stopping {name}: saving a checkpoint first…",
            "finished" => $"{name} is finished",
            "stopped" => $"{name} stopped; build it again to continue",
            "failed" => $"{name} failed (exit code {state.ExitCode}); the log says why",
            _ => "Ready. Fill in the voice and start training.",
        };
    }

    public static string Error(string code, string message) => code switch
    {
        "invalid_name" => "Give the voice a name using letters, digits, '-', '_' or '.'.",
        "missing_audio" => "The recordings folder or file doesn't exist.",
        "invalid_numbers" => "Steps, the save interval and the batch size must be whole numbers of at least 1.",
        "busy" => "A voice is already training. Add this one to the queue instead.",
        _ => message,
    };
}
