app-title = CausalVC Trainer

page-voice = New voice
page-builds = Builds
page-log = Log

service-starting = Starting…
service-starting-detail = Loading the training tools.
service-unavailable = The trainer could not start
service-retrying = Trying again in a few seconds. The message below says what went wrong.

section-dataset = Dataset
audio = Recordings
audio-detail = A folder of the target's speech, or one long recording. Any format and sample rate; pauses can stay in.
audio-placeholder = Paste a path, or choose a folder or file
choose-folder = Folder…
choose-file = File…
pick-audio-folder = Choose the folder of recordings
pick-audio-file = Choose a recording
voice-name = Voice name
voice-name-detail = Letters, digits, '-', '_' or '.'. Building the same name again picks up where it left off.

section-training = Training
steps = Steps
steps-detail = How long to train. 10,000 is a good start; more steps follow the voice more closely.
save-every = Save a version every
save-every-detail = Steps between saved versions (NAME-1000s, NAME-2000s and so on), to compare and keep the best.
batch = Batch size
batch-detail = Clips trained on at once. Lower it if the GPU runs out of memory.
restart = Start over
restart-detail = Ignore this voice name's earlier progress and train from the beginning. Applies to the next build only.

section-hardware = Starting point and hardware
base = Base
base-detail = The voice model training starts from. Your own bases go in the custom-base folder.
base-default = Default
gpu = GPU
gpu-detail = The graphics card that trains the voice.
gpu-loading = Looking for GPUs…
gpu-cpu = No GPU found (very slow)

section-install = Voice changer
install = Add to the voice changer
install-detail = Every saved version appears in the voice changer's library as it is written.
install-folder = Voices folder
pick-install-folder = Choose the voice changer's voices folder
change = Change…

start = Start training
queue-build = Add to queue
build-queued = { $name } added to the queue ({ $position ->
    [1] next up
   *[other] #{ $position } in line
}). See it on the Builds page.
stop = Stop
end-now = End now
stop-detail = Stop saves a checkpoint first; building the same voice again resumes. Click again to end at once without saving.

phase-idle = Ready. Fill in the voice and start training.
phase-starting = Starting { $name } (loading the models takes a little while)…
phase-training = Training { $name }
phase-stopping = Stopping { $name }: saving a checkpoint first…
phase-switching = Saving { $name } to switch to { $next }; { $name } resumes next…
phase-finished = { $name } is finished
phase-stopped = { $name } stopped; build it again to continue
phase-failed = { $name } failed (exit code { $code }); the log says why
progress = Step { $step } of { $total }
progress-times = { $elapsed } elapsed · { $remaining } left · { $rate } steps/s

builds-current = Now
builds-none = Nothing is training.
builds-queue = Up next
builds-queue-empty = The queue is empty. Use “Add to queue” to train several voices one after another.
builds-steps = { $steps } steps
remove = Remove
build-now = Build now
builds-queue-drag = Drag a build by its handle to change the order. Build now saves the current build and puts it first in line, to resume after.
close-note = Closing this window stops training the same way Stop does: a checkpoint is saved first.

log-empty = Messages from training appear here.
clear-log = Clear

error-invalid_name = Give the voice a name using letters, digits, '-', '_' or '.'.
error-missing_audio = The recordings folder or file doesn't exist.
error-invalid_numbers = Steps, the save interval and the batch size must be whole numbers of at least 1.
error-busy = A voice is already training. Add this one to the queue instead.
error-other = { $message }
