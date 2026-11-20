//! Page and control-bar views.

use cosmic::iced::{Alignment, Length};
use cosmic::prelude::*;
use cosmic::widget::{self, settings};

use crate::app::{
    App, Connection, DENOISE_METHODS, LOUDNESS_TARGETS, Function, INPUT_SOURCES, Message, Num, Page,
    RECORDING_MODES, Toggle, VALUE_INPUT_ID,
};
use crate::engine::State;
use crate::fl;

const SLIDER_WIDTH: f32 = 240.0;
const VALUE_WIDTH: f32 = 72.0;
const METER_WIDTH: f32 = 90.0;

pub fn view(app: &App, page: Page) -> Element<'_, Message> {
    if page == Page::About {
        // Shown whether or not the engine runs: an update may be what makes it start again.
        return padded(about_page(app));
    }
    let Some(state) = app
        .state
        .as_ref()
        .filter(|_| matches!(app.connection, Connection::Ready))
    else {
        return connection_view(app);
    };
    let content = match page {
        Page::Model => model_page(app, state),
        Page::Library => crate::library::page(app, state),
        Page::Merge => merge_page(app),
        Page::Audio => audio_page(app, state),
        Page::Performance => performance_page(app, state),
        Page::Recording => recording_page(app, state),
        Page::Log => return log_page(app),
        Page::About => unreachable!(),
    };
    padded(content)
}

fn padded(content: Element<'_, Message>) -> Element<'_, Message> {
    widget::scrollable(
        // Room at the end so the last section clears the control bar.
        widget::container(content).max_width(900).padding(
            cosmic::iced::Padding::from([0, cosmic::theme::spacing().space_s])
                .bottom(cosmic::theme::spacing().space_l),
        ),
    )
    .width(Length::Fill)
    .into()
}

fn about_page(app: &App) -> Element<'_, Message> {
    page_column(vec![app.updates.section().map(Message::Update)])
}

fn connection_view(app: &App) -> Element<'_, Message> {
    let space = cosmic::theme::spacing();
    let column = match &app.connection {
        Connection::Failed(reason) => widget::column::with_capacity(4)
            .push(widget::icon::from_name("dialog-error-symbolic").size(48))
            .push(widget::text::title3(fl!("engine-unavailable")))
            .push(widget::text::body(fl!("engine-retrying")))
            .push(
                widget::container(widget::text::monotext(reason.as_str()))
                    .class(cosmic::theme::Container::Card)
                    .padding(space.space_s)
                    .max_width(720),
            ),
        _ => widget::column::with_capacity(4)
            .push(widget::progress_bar::circular::Circular::new().size(40.0))
            .push(widget::text::title3(fl!("engine-starting")))
            .push(widget::text::caption(fl!("engine-starting-detail"))),
    };
    widget::container(column.spacing(space.space_s).align_x(Alignment::Center))
        .center(Length::Fill)
        .into()
}

fn page_column<'a>(sections: Vec<Element<'a, Message>>) -> Element<'a, Message> {
    settings::view_column(sections).into()
}

/// A labelled slider with its formatted value.  Clicking the value opens a
/// field to type an exact one (Enter or clicking away applies, Escape cancels).
fn slider_item<'a>(
    app: &'a App,
    title: String,
    num: Num,
    state: &State,
    format: impl Fn(f32) -> String,
) -> Element<'a, Message> {
    let (min, max, step) = num.range();
    let value = num.get(&state.settings);
    let value_widget: Element<_> = match &app.editing {
        Some((editing, text)) if *editing == num => {
            let invalid = num.parse(text).is_none();
            let mut input = widget::text_input::inline_input("", text.as_str())
                .id(VALUE_INPUT_ID.clone())
                .on_input(Message::EditInput)
                .on_submit(|_| Message::EditSubmit)
                .on_unfocus(Message::EditBlur)
                .width(Length::Fixed(VALUE_WIDTH));
            if invalid {
                input = input.error(fl!("value-invalid"));
            }
            input.into()
        }
        _ => widget::tooltip(
            widget::button::custom(
                widget::text::body(format(value))
                    .width(Length::Fill)
                    .align_x(Alignment::End),
            )
            .class(cosmic::theme::Button::Text)
            .padding([2, 4])
            .width(Length::Fixed(VALUE_WIDTH))
            .on_press(Message::EditValue(num)),
            widget::text::caption(fl!("value-click-to-type")),
            widget::tooltip::Position::Top,
        )
        .into(),
    };
    let mut item = settings::item::builder(title);
    if let Some(description) = num_description(num) {
        item = item.description(description);
    }
    item.control(
        widget::row::with_capacity(2)
            .push(
                widget::slider(min..=max, value.clamp(min, max), move |v| {
                    Message::Num(num, v)
                })
                .step(step)
                .on_release(Message::NumReleased(num))
                .width(Length::Fixed(SLIDER_WIDTH)),
            )
            .push(value_widget)
            .spacing(cosmic::theme::spacing().space_xs)
            .align_y(Alignment::Center),
    )
    .into()
}

/// What each slider does, shown under its title.
fn num_description(num: Num) -> Option<String> {
    Some(match num {
        Num::Pitch => fl!("pitch-detail"),
        Num::PitchOffset => fl!("pitch-offset-detail"),
        Num::PitchCeiling => fl!("pitch-ceiling-detail"),
        Num::PitchCeilingKeep => fl!("pitch-ceiling-keep-detail"),
        Num::IndexRate => fl!("index-rate-detail"),
        Num::InputGain => fl!("input-gain-detail"),
        Num::OutputGain => fl!("output-gain-detail"),
        Num::MonitorGain => fl!("monitor-gain-detail"),
        Num::NoiseGate => fl!("noise-gate-detail"),
        Num::BlockTime => fl!("chunk-detail"),
        Num::FileVolume => fl!("file-volume-detail"),
    })
}

fn dropdown_item<'a>(
    title: String,
    description: Option<String>,
    labels: &'a [String],
    selected: Option<usize>,
    on_select: fn(usize) -> Message,
) -> Element<'a, Message> {
    let mut item = settings::item::builder(title);
    if let Some(description) = description {
        item = item.description(description);
    }
    item.control(widget::dropdown(labels, selected, on_select)).into()
}

fn position_of(labels: &[String], value: &str) -> Option<usize> {
    labels.iter().position(|label| label == value)
}

fn signed(decimals: usize) -> impl Fn(f32) -> String {
    move |value| format!("{value:+.decimals$}")
}

fn decibels(value: f32) -> String {
    format!("{value:+.1} dB")
}

fn seconds(value: f32) -> String {
    format!("{value:.2} s")
}

pub fn mmss(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    format!("{:02}:{:02}", total / 60, total % 60)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

fn model_page<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let selected = state.settings.model_name.as_str();
    let files = state
        .models
        .iter()
        .find(|model| model.name == selected)
        .map(|model| fl!("model-files", model = model.model_file.clone()))
        .unwrap_or_default();

    let mut sections = Vec::with_capacity(5);
    sections.extend([
        settings::section()
            .title(fl!("section-models"))
            .add(if state.models.is_empty() {
                Element::from(widget::text::body(fl!("models-empty")))
            } else {
                dropdown_item(
                    fl!("model"),
                    None,
                    &app.voice_names,
                    state
                        .selected_voice()
                        .and_then(|voice| position_of(&app.voice_names, &voice.name)),
                    Message::SelectModel,
                )
            })
            .add_maybe((app.checkpoint_names.len() > 1).then(|| {
                dropdown_item(
                    fl!("checkpoint"),
                    Some(fl!("checkpoint-description")),
                    &app.checkpoint_labels,
                    position_of(&app.checkpoint_names, selected),
                    Message::SelectCheckpoint,
                )
            }))
            .add(
                widget::row::with_capacity(4)
                    .push(widget::text::caption(files).width(Length::Fill))
                    .push(
                        widget::button::standard(fl!("page-library"))
                            .leading_icon(widget::icon::from_name("view-list-symbolic"))
                            .on_press(Message::OpenLibrary),
                    )
                    .push(
                        widget::button::standard(fl!("reload"))
                            .leading_icon(widget::icon::from_name("view-refresh-symbolic"))
                            .on_press(Message::ReloadModels),
                    )
                    .spacing(cosmic::theme::spacing().space_xs)
                    .align_y(Alignment::Center),
            )
            .into(),
        voice_section(app, state),
        settings::section()
            .title(fl!("section-levels"))
            .add(slider_item(
                app,
                fl!("input-gain"),
                Num::InputGain,
                state,
                decibels,
            ))
            .add(slider_item(
                app,
                fl!("output-gain"),
                Num::OutputGain,
                state,
                decibels,
            ))
            .add(output_level_item(app, state))
            .add(slider_item(
                app,
                fl!("monitor-gain"),
                Num::MonitorGain,
                state,
                decibels,
            ))
            .add(slider_item(
                app,
                fl!("noise-gate"),
                Num::NoiseGate,
                state,
                |value| {
                    if value <= -60.0 {
                        fl!("off")
                    } else {
                        format!("{value:.0} dB")
                    }
                },
            ))
            .into(),
        reset_row(fl!("reset-model-settings"), "general"),
    ]);
    page_column(sections)
}

/// The pitch, and behind a disclosure row the pitch offset and ceiling, which few voices need.
fn voice_section<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let offset = state.settings.pitch_offset_hz;
    let ceiling = state.settings.pitch_ceiling_hz;
    let open = app.pitch_advanced_open;
    let toggle = widget::button::icon(widget::icon::from_name(if open {
        "go-up-symbolic"
    } else {
        "go-down-symbolic"
    }))
    .tooltip(if open { fl!("pitch-advanced-hide") } else { fl!("pitch-advanced-show") })
    .on_press(Message::TogglePitchAdvanced);
    let advanced = settings::item::builder(fl!("pitch-advanced"))
        .description(fl!(
            "pitch-advanced-detail",
            offset = hertz(offset),
            ceiling = ceiling_text(ceiling)
        ))
        .control(toggle);
    let mut section = settings::section()
        .title(fl!("section-voice"))
        .add(slider_item(app, fl!("pitch"), Num::Pitch, state, signed(1)))
        .add(index_item(app, state))
        .add(advanced);
    if open {
        section = section
            .add(slider_item(app, fl!("pitch-offset"), Num::PitchOffset, state, hertz))
            .add(slider_item(app, fl!("pitch-ceiling"), Num::PitchCeiling, state, ceiling_text))
            .add(slider_item(
                app,
                fl!("pitch-ceiling-keep"),
                Num::PitchCeilingKeep,
                state,
                |value| format!("{:.0}%", value * 100.0),
            ));
    }
    section.into()
}

/// The Index slider, for a voice whose file carries an index; otherwise a note saying why there is none.
fn index_item<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let has_index = state
        .models
        .iter()
        .any(|model| model.name == state.settings.model_name && model.has_index);
    if !has_index {
        return settings::item::builder(fl!("index-rate"))
            .description(fl!("index-rate-none"))
            .control(widget::text::body(fl!("off")))
            .into();
    }
    slider_item(app, fl!("index-rate"), Num::IndexRate, state, |value| {
        if value <= 0.0 {
            fl!("off")
        } else {
            format!("{:.0}%", value * 100.0)
        }
    })
}

fn hertz(value: f32) -> String {
    format!("{value:+.0} Hz")
}

fn ceiling_text(value: f32) -> String {
    if value <= 0.0 {
        fl!("off")
    } else {
        format!("{value:.0} Hz")
    }
}

/// How loud the output sounds to others, with a one-click gain fix.
fn output_level_item<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let space = cosmic::theme::spacing();
    let reading = app.meters.loudness.as_ref().filter(|_| state.running);
    let target_index = LOUDNESS_TARGETS
        .iter()
        .position(|target| *target == state.settings.loudness_target);
    let description = match reading {
        None => fl!("output-level-idle"),
        Some(reading) => match (reading.lufs, reading.verdict.as_deref()) {
            (None, _) => fl!("output-level-listening"),
            (Some(lufs), None) => fl!(
                "output-level-measuring",
                lufs = format!("{lufs:.1}"),
                seconds = format!("{:.0}", reading.speech_seconds)
            ),
            (Some(lufs), Some(verdict)) => {
                let text = match verdict {
                    "quiet" if reading.adjust_db > 0.0 => fl!("output-level-quiet"),
                    // Peaks already reach full scale: raising would squash them.
                    "quiet" => fl!("output-level-quiet-peaky"),
                    "loud" if reading.over_percent > 1.0 => fl!("output-level-squashed"),
                    "loud" => fl!("output-level-loud"),
                    _ => fl!("output-level-good"),
                };
                fl!(
                    "output-level-reading",
                    lufs = format!("{lufs:.1}"),
                    target = format!("{:.0}", reading.target_lufs),
                    verdict = text
                )
            }
        },
    };
    let mut control = widget::row::with_capacity(3)
        .spacing(space.space_xs)
        .align_y(Alignment::Center);
    if let Some(reading) = reading.filter(|r| r.verdict.is_some() && r.adjust_db != 0.0) {
        control = control
            .push(widget::text::body(format!("{:+.1} dB", reading.adjust_db)))
            .push(
                widget::button::suggested(fl!("output-level-apply"))
                    .on_press(Message::ApplyLoudness),
            );
    }
    control = control.push(widget::dropdown(
        &app.loudness_target_labels,
        target_index,
        Message::SelectLoudnessTarget,
    ));
    settings::item::builder(fl!("output-level"))
        .description(description)
        .control(control)
        .into()
}

/// One-line output-level verdict for the control bar.
fn level_summary(app: &App) -> Option<(String, &'static str)> {
    let reading = app.meters.loudness.as_ref()?;
    let (text, icon) = match reading.verdict.as_deref() {
        Some("good") => (fl!("level-good"), "emblem-ok-symbolic"),
        Some("quiet") => (fl!("level-quiet"), "audio-volume-low-symbolic"),
        Some("loud") => (fl!("level-loud"), "dialog-warning-symbolic"),
        _ => (fl!("level-measuring"), "audio-volume-medium-symbolic"),
    };
    Some((text, icon))
}

fn reset_row<'a>(label: String, group: &'static str) -> Element<'a, Message> {
    widget::row::with_capacity(4)
        .push(widget::space::horizontal())
        .push(widget::button::standard(label).on_press(Message::Reset(group)))
        .into()
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

fn audio_page<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let space = cosmic::theme::spacing();
    let settings_ = &state.settings;
    let file_source = settings_.input_source == "file";
    let source_index = INPUT_SOURCES
        .iter()
        .position(|s| *s == settings_.input_source);

    let mut input = settings::section()
        .title(fl!("section-input"))
        .add(dropdown_item(
            fl!("input-source"),
            Some(fl!("input-source-detail")),
            &app.source_labels,
            source_index,
            Message::SelectInputSource,
        ));
    if file_source && !state.ffmpeg {
        input = input.add(widget::text::body(fl!("error-ffmpeg-missing")));
    }
    if file_source {
        let file_name = state
            .file
            .path
            .as_deref()
            .and_then(|path| std::path::Path::new(path).file_name())
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| fl!("no-file"));
        input = input.add(settings::item(
            fl!("audio-file"),
            widget::row::with_capacity(4)
                .push(widget::text::body(file_name))
                .push(widget::button::standard(fl!("browse")).on_press(Message::PickAudioFile))
                .spacing(space.space_s)
                .align_y(Alignment::Center),
        ));
        input = input.add(file_player(app, state));
        input = input.add(slider_item(
            app,
            fl!("file-volume"),
            Num::FileVolume,
            state,
            |value| format!("{:.0}%", value * 100.0),
        ));
    } else {
        input = input.add(dropdown_item(
            fl!("input-device"),
            Some(fl!("input-device-detail")),
            &app.input_labels,
            position_of(&app.input_labels, &settings_.input_device),
            Message::SelectInput,
        ));
    }

    let monitor_index = match &settings_.monitor_device {
        None => Some(0),
        Some(label) => position_of(&app.output_labels, label).map(|i| i + 1),
    };
    let output = settings::section()
        .title(fl!("section-output"))
        .add(dropdown_item(
            fl!("output-device"),
            Some(fl!("output-device-detail")),
            &app.output_labels,
            position_of(&app.output_labels, &settings_.output_device),
            Message::SelectOutput,
        ))
        .add(dropdown_item(
            fl!("monitor-device"),
            Some(fl!("monitor-device-detail")),
            &app.monitor_labels,
            monitor_index,
            Message::SelectMonitor,
        ));

    let rate = state
        .samplerate
        .map(|rate| fl!("sample-rate-value", rate = rate))
        .unwrap_or_else(|| fl!("sample-rate-idle"));
    page_column(vec![
        input.into(),
        output.into(),
        widget::row::with_capacity(4)
            .push(
                widget::text::caption(format!("{}\n{}", rate, fl!("jack-hint")))
                    .width(Length::Fill),
            )
            .push(
                widget::button::standard(fl!("reload-devices"))
                    .leading_icon(widget::icon::from_name("view-refresh-symbolic"))
                    .on_press(Message::ReloadDevices),
            )
            .spacing(space.space_s)
            .align_y(Alignment::Center)
            .into(),
        cleanup_section(app, state),
        reset_row(fl!("reset-cleanup"), "cleanup"),
    ])
}

/// What is done to the sound on its way through: noise reduction on the mic, treble softening on the voice.
fn cleanup_section<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let settings_ = &state.settings;
    settings::section()
        .title(fl!("section-cleanup"))
        .add(
            settings::item::builder(fl!("input-denoise"))
                .description(fl!("input-denoise-detail"))
                .toggler(settings_.input_denoise, |value| {
                    Message::Toggle(Toggle::InputDenoise, value)
                }),
        )
        .add(denoise_method_item(app, state))
        .add(
            settings::item::builder(fl!("soften-treble"))
                .description(fl!("soften-treble-detail"))
                .toggler(settings_.soften_treble, |value| {
                    Message::Toggle(Toggle::SoftenTreble, value)
                }),
        )
        .into()
}

fn file_player<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let space = cosmic::theme::spacing();
    let has_file = state.file.path.is_some();
    let duration = state.file.duration.unwrap_or(0.0).max(1.0) as f32;
    let position = app
        .seek_preview
        .unwrap_or(app.meters.file_position.unwrap_or(0.0) as f32);
    let play_pause = if state.file.playing {
        widget::button::icon(widget::icon::from_name("media-playback-pause-symbolic"))
            .on_press(Message::FilePause)
    } else {
        widget::button::icon(widget::icon::from_name("media-playback-start-symbolic"))
            .on_press_maybe(has_file.then_some(Message::FilePlay))
    };
    widget::row::with_capacity(4)
        .push(play_pause)
        .push(
            widget::button::icon(widget::icon::from_name("media-playback-stop-symbolic"))
                .on_press_maybe(has_file.then_some(Message::FileStop)),
        )
        .push(
            widget::slider(0.0..=duration, position.min(duration), Message::FileSeek)
                .on_release(Message::FileSeekReleased)
                .width(Length::Fill),
        )
        .push(widget::text::body(format!(
            "{} / {}",
            mmss(position as f64),
            mmss(state.file.duration.unwrap_or(0.0))
        )))
        .spacing(space.space_xs)
        .align_y(Alignment::Center)
        .into()
}

// ---------------------------------------------------------------------------
// Performance
// ---------------------------------------------------------------------------

fn performance_page<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let settings_ = &state.settings;
    let gpu_index = state.gpus.iter().position(|gpu| gpu.id == settings_.gpu);
    page_column(vec![
        settings::section()
            .title(fl!("section-buffering"))
            .add(slider_item(
                app,
                fl!("chunk"),
                Num::BlockTime,
                state,
                seconds,
            ))
            .into(),
        settings::section()
            .title(fl!("section-inference"))
            .add(dropdown_item(
                fl!("gpu"),
                Some(fl!("gpu-detail")),
                &app.gpu_labels,
                gpu_index,
                Message::SelectGpu,
            ))
            .add(
                settings::item::builder(fl!("cuda-graphs"))
                    .description(fl!("cuda-graphs-detail"))
                    .toggler(settings_.cuda_graphs, |value| {
                        Message::Toggle(Toggle::CudaGraphs, value)
                    }),
            )
            .into(),
        reset_row(fl!("reset-performance"), "performance"),
    ])
}

/// Which denoiser, and what it needs: Hushwire's model is downloaded once, after the notice.
fn denoise_method_item<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let space = cosmic::theme::spacing();
    let method = state.settings.input_denoise_method.as_str();
    let index = DENOISE_METHODS.iter().position(|m| *m == method);
    let (description, download) = match (method, state.hushwire.as_str()) {
        ("rnnoise", _) if state.rnnoise => (fl!("denoise-rnnoise-ready"), false),
        ("rnnoise", _) => (fl!("input-denoise-spectral"), false),
        (_, "ready") => (fl!("denoise-hushwire-ready"), false),
        (_, "downloading") => (fl!("denoise-hushwire-downloading"), false),
        (_, "failed") => (fl!("denoise-hushwire-failed"), true),
        (_, "unavailable") => (fl!("denoise-hushwire-unavailable"), false),
        _ => (fl!("denoise-hushwire-needs-download"), true),
    };
    let mut control = widget::row::with_capacity(2)
        .spacing(space.space_xs)
        .align_y(Alignment::Center);
    if download {
        control = control.push(
            widget::button::suggested(fl!("denoise-hushwire-download"))
                .on_press(Message::AskHushwireDownload),
        );
    }
    control = control.push(widget::dropdown(
        &app.denoise_method_labels,
        index,
        Message::SelectDenoiseMethod,
    ));
    settings::item::builder(fl!("denoise-method"))
        .description(description)
        .control(control)
        .into()
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

fn recording_page<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let space = cosmic::theme::spacing();
    let settings_ = &state.settings;
    let mode_index = RECORDING_MODES
        .iter()
        .position(|mode| *mode == settings_.recording_mode);
    let record_button = if state.recording {
        widget::button::destructive(fl!("stop-recording"))
            .leading_icon(widget::icon::from_name("media-playback-stop-symbolic"))
            .on_press(Message::ToggleRecording)
    } else {
        widget::button::suggested(fl!("record"))
            .leading_icon(widget::icon::from_name("media-record-symbolic"))
            .on_press_maybe(state.running.then_some(Message::ToggleRecording))
    };
    let elapsed = match app.meters.recording_seconds {
        Some(seconds) if state.recording => {
            let total = seconds as u64;
            format!(
                "● REC {:02}:{:02}:{:02}",
                total / 3600,
                (total / 60) % 60,
                total % 60
            )
        }
        _ if !state.running => fl!("record-hint"),
        _ => String::new(),
    };
    page_column(vec![
        settings::section()
            .title(fl!("section-recording"))
            .add(dropdown_item(
                fl!("recording-mode"),
                Some(fl!("recording-mode-detail")),
                &app.recording_mode_labels,
                mode_index,
                Message::SelectRecordingMode,
            ))
            .add(settings::item(
                fl!("save-folder"),
                widget::row::with_capacity(4)
                    .push(widget::text::caption(settings_.recording_folder.as_str()))
                    .push(
                        widget::button::standard(fl!("change"))
                            .on_press(Message::PickRecordingFolder),
                    )
                    .push(
                        widget::button::standard(fl!("open-folder"))
                            .on_press(Message::OpenRecordingFolder),
                    )
                    .spacing(space.space_xs)
                    .align_y(Alignment::Center),
            ))
            .into(),
        widget::row::with_capacity(4)
            .push(widget::text::body(elapsed).width(Length::Fill))
            .push(record_button)
            .align_y(Alignment::Center)
            .into(),
    ])
}

// ---------------------------------------------------------------------------
// Log
// ---------------------------------------------------------------------------

fn log_page(app: &App) -> Element<'_, Message> {
    let space = cosmic::theme::spacing();
    widget::column::with_capacity(4)
        .push(
            widget::row::with_capacity(4)
                .push(widget::space::horizontal())
                .push(widget::button::standard(fl!("save-log")).on_press(Message::SaveLog))
                .push(widget::button::standard(fl!("clear-log")).on_press(Message::ClearLog))
                .spacing(space.space_xs),
        )
        .push(
            widget::container(
                widget::scrollable(widget::text::monotext(app.log.as_str()).width(Length::Fill))
                    .anchor_bottom()
                    .height(Length::Fill),
            )
            .class(cosmic::theme::Container::Card)
            .padding(space.space_s)
            .height(Length::Fill),
        )
        .spacing(space.space_s)
        .padding([0, space.space_s, space.space_s, space.space_s])
        .height(Length::Fill)
        .into()
}

// ---------------------------------------------------------------------------
// Control bar
// ---------------------------------------------------------------------------

pub fn control_bar<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let space = cosmic::theme::spacing();
    let converting = state.running && state.function == "vc";
    let passthrough = state.running && state.function == "passthrough";
    let enabled = !app.starting;

    let start = if converting {
        widget::button::destructive(fl!("converting"))
            .leading_icon(widget::icon::from_name("media-playback-stop-symbolic"))
    } else {
        widget::button::suggested(if app.starting {
            fl!("starting")
        } else {
            fl!("start")
        })
        .leading_icon(widget::icon::from_name("media-playback-start-symbolic"))
    }
    .on_press_maybe(enabled.then_some(Message::Run(Function::Convert)));
    let passthrough_button = if passthrough {
        widget::button::destructive(fl!("passthrough-active"))
    } else {
        widget::button::standard(fl!("passthrough"))
    }
    .on_press_maybe(enabled.then_some(Message::Run(Function::Passthrough)));

    let meter = |label: String, value: f32| {
        widget::row::with_capacity(4)
            .push(widget::text::caption(label).width(Length::Fixed(32.0)))
            .push(
                widget::progress_bar::linear::Linear::new()
                    .progress(value)
                    .girth(6.0)
                    .width(Length::Fixed(METER_WIDTH)),
            )
            .spacing(space.space_xxs)
            .align_y(Alignment::Center)
    };
    let meters = widget::column::with_capacity(4)
        .push(meter(fl!("meter-in"), app.meters.input))
        .push(meter(fl!("meter-out"), app.meters.output))
        .push(meter(fl!("meter-mon"), app.meters.monitor))
        .spacing(2);

    let dash = || "—".to_string();
    let with_tip = |text: String, tip: String| {
        widget::tooltip(
            widget::text::caption(text),
            widget::text::caption(tip),
            widget::tooltip::Position::Top,
        )
    };
    let stats = widget::column::with_capacity(4)
        .push(with_tip(
            fl!(
                "latency",
                value = state
                    .delay_ms
                    .map(|ms| (ms + i64::from(app.meters.queued_ms.unwrap_or(0))).to_string())
                    .unwrap_or_else(dash)
            ),
            fl!("latency-detail"),
        ))
        .push(with_tip(
            fl!(
                "inference",
                value = app
                    .meters
                    .infer_ms
                    .map(|ms| ms.to_string())
                    .unwrap_or_else(dash)
            ),
            fl!("inference-detail"),
        ));
    let stats = match level_summary(app).filter(|_| state.running) {
        Some((text, icon)) => stats.push(
            widget::row::with_capacity(2)
                .push(widget::icon::from_name(icon).size(14))
                .push(widget::text::caption(text))
                .spacing(space.space_xxs)
                .align_y(Alignment::Center),
        ),
        None => stats,
    }
    .width(Length::Fixed(140.0));

    widget::container(
        widget::row::with_capacity(4)
            .push(start)
            .push(passthrough_button)
            .push(widget::text::body(app.status.as_str()).width(Length::Fill))
            .push(meters)
            .push(stats)
            .spacing(space.space_s)
            .align_y(Alignment::Center),
    )
    .class(cosmic::theme::Container::Card)
    .padding(space.space_s)
    .into()
}

/// Shown over the page while files are dragged onto the window.
pub fn drop_overlay<'a>() -> Element<'a, Message> {
    let space = cosmic::theme::spacing();
    widget::container(
        widget::container(
            widget::column::with_capacity(3)
                .push(widget::icon::from_name("document-open-symbolic").size(48))
                .push(widget::text::title3(fl!("drop-title")))
                .push(widget::text::body(fl!("drop-detail")))
                .spacing(space.space_xs)
                .align_x(Alignment::Center),
        )
        .class(cosmic::theme::Container::Dialog(true))
        .padding(space.space_l),
    )
    .center(Length::Fill)
    .class(cosmic::theme::Container::Transparent)
    .into()
}

#[cfg(test)]
mod tests {
    use super::mmss;

    #[test]
    fn formats_minutes_and_seconds() {
        assert_eq!(mmss(0.0), "00:00");
        assert_eq!(mmss(75.9), "01:15");
        assert_eq!(mmss(-3.0), "00:00");
    }
}

/// Why the Merge page's voices and name cannot be merged yet, or None when they can.
pub fn merge_problem(app: &App) -> Option<String> {
    let chosen: Vec<&String> = app.merge_slots.iter().filter_map(|(name, _)| name.as_ref()).collect();
    if chosen.len() < 2 {
        return Some(fl!("merge-need-two"));
    }
    if (1..chosen.len()).any(|i| chosen[..i].contains(&chosen[i])) {
        return Some(fl!("merge-duplicate"));
    }
    let family = |name: &String| {
        app.merge_labels
            .iter()
            .position(|label| label == name)
            .and_then(|index| app.merge_families.get(index))
    };
    if chosen.iter().any(|name| family(name) != family(chosen[0])) {
        return Some(fl!("merge-incompatible"));
    }
    let name = app.merge_name.trim();
    if name.is_empty() {
        return Some(fl!("merge-need-name"));
    }
    if app.model_names.iter().any(|existing| existing.to_lowercase() == name.to_lowercase()) {
        return Some(fl!("error-model-name-taken"));
    }
    None
}

fn merge_page(app: &App) -> Element<'_, Message> {
    let space = cosmic::theme::spacing();
    let total: f32 = app
        .merge_slots
        .iter()
        .filter(|(name, _)| name.is_some())
        .map(|(_, strength)| strength)
        .sum();
    let first_family = app
        .merge_slots
        .iter()
        .find_map(|(name, _)| name.as_ref())
        .and_then(|name| app.merge_labels.iter().position(|label| label == name))
        .and_then(|index| app.merge_families.get(index));
    let mut voices = settings::section().title(fl!("merge-voices"));
    for (slot, (name, strength)) in app.merge_slots.iter().enumerate() {
        let selected = name.as_ref().and_then(|name| app.merge_labels.iter().position(|label| label == name));
        let fits = selected.and_then(|index| app.merge_families.get(index)) == first_family || selected.is_none();
        let description = match (selected, fits) {
            (Some(_), false) => fl!("merge-slot-incompatible"),
            (Some(_), true) if total > 0.0 => fl!("merge-share", percent = format!("{:.0}", strength / total * 100.0)),
            _ => fl!("merge-slot-empty"),
        };
        let mut control = widget::row::with_capacity(4)
            .spacing(space.space_xs)
            .align_y(Alignment::Center)
            .push(
                widget::dropdown(app.merge_labels.as_slice(), selected, move |index| Message::MergeVoice(slot, index))
                    .width(Length::Fixed(220.0)),
            )
            .push(
                widget::slider(1.0_f32..=100.0, *strength, move |value| Message::MergeStrength(slot, value))
                    .step(1.0_f32)
                    .width(Length::Fixed(160.0)),
            )
            .push(widget::text::body(format!("{strength:.0}")).width(Length::Fixed(32.0)));
        if app.merge_slots.len() > 2 {
            control = control.push(
                widget::button::icon(widget::icon::from_name("list-remove-symbolic"))
                    .tooltip(fl!("merge-remove-slot"))
                    .on_press(Message::MergeRemoveSlot(slot)),
            );
        }
        let number = slot + 1;
        voices = voices.add(
            settings::item::builder(fl!("merge-slot", number = number))
                .description(description)
                .control(control),
        );
    }
    voices = voices.add(settings::item_row(vec![
        widget::button::standard(fl!("merge-add-slot"))
            .leading_icon(widget::icon::from_name("list-add-symbolic"))
            .on_press(Message::MergeAddSlot)
            .into(),
    ]));

    let problem = merge_problem(app);
    let can_merge = problem.is_none() && !app.merging;
    let result = settings::section()
        .title(fl!("merge-result"))
        .add(
            settings::item::builder(fl!("merge-name"))
                .description(fl!("merge-name-detail"))
                .control(
                    widget::text_input(fl!("merge-name-placeholder"), app.merge_name.as_str())
                        .on_input(Message::MergeName)
                        .on_submit(|_| Message::Merge)
                        .width(Length::Fixed(260.0)),
                ),
        )
        .add(settings::item_row(vec![
            widget::button::suggested(if app.merging { fl!("merge-running") } else { fl!("merge-button") })
                .on_press_maybe(can_merge.then_some(Message::Merge))
                .into(),
            widget::text::caption(problem.unwrap_or_default()).into(),
        ]));

    let mut column = vec![
        widget::text::body(fl!("merge-intro")).into(),
        voices.into(),
        result.into(),
    ];
    if app.merge_labels.len() < 2 {
        column.insert(1, widget::text::body(fl!("merge-not-enough")).into());
    }
    settings::view_column(column).into()
}
