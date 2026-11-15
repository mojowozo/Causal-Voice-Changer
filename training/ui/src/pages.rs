//! The pages, the bottom control bar, and the view shown while the trainer starts.

use cosmic::iced::{Alignment, Length};
use cosmic::prelude::*;
use cosmic::widget::{self, settings};

use crate::app::{App, Connection, Field, Message, Page, PickTarget};
use crate::fl;
use crate::trainer::State;

const NUMBER_WIDTH: f32 = 120.0;
const PATH_WIDTH: f32 = 320.0;

pub fn view(app: &App, page: Page) -> Element<'_, Message> {
    if !matches!(app.connection, Connection::Ready) {
        return connection_view(app);
    }
    let content = match page {
        Page::Voice => voice_page(app),
        Page::Builds => builds_page(app),
        Page::Log => return log_page(app),
    };
    widget::scrollable(
        widget::container(content)
            .max_width(900)
            .padding([0, cosmic::theme::spacing().space_s]),
    )
    .width(Length::Fill)
    .into()
}

fn connection_view(app: &App) -> Element<'_, Message> {
    let space = cosmic::theme::spacing();
    let column = match &app.connection {
        Connection::Failed(reason) => widget::column::with_capacity(4)
            .push(widget::icon::from_name("dialog-error-symbolic").size(48))
            .push(widget::text::title3(fl!("service-unavailable")))
            .push(widget::text::body(fl!("service-retrying")))
            .push(
                widget::container(widget::text::monotext(reason.as_str()))
                    .class(cosmic::theme::Container::Card)
                    .padding(space.space_s)
                    .max_width(720),
            ),
        _ => widget::column::with_capacity(3)
            .push(widget::progress_bar::circular::Circular::new().size(40.0))
            .push(widget::text::title3(fl!("service-starting")))
            .push(widget::text::caption(fl!("service-starting-detail"))),
    };
    widget::container(column.spacing(space.space_s).align_x(Alignment::Center))
        .center(Length::Fill)
        .into()
}

/// A text field that shows the trainer's complaint about it, if any.
fn field<'a>(app: &'a App, field: Field, value: &'a str, width: f32) -> Element<'a, Message> {
    let mut input = widget::text_input("", value)
        .on_input(move |text| Message::Text(field, text))
        .width(Length::Fixed(width));
    if let Some((key, text)) = &app.field_error
        && key == field.key()
    {
        input = input.error(text.as_str());
    }
    input.into()
}

fn voice_page(app: &App) -> Element<'_, Message> {
    let space = cosmic::theme::spacing();
    let audio_error = app.field_error.as_ref().filter(|(key, _)| key == "audio");
    let audio_path: Element<_> = if app.audio.is_empty() {
        widget::text::caption(fl!("audio-none")).into()
    } else {
        widget::text::body(app.audio.as_str())
            .width(Length::Fixed(PATH_WIDTH))
            .into()
    };
    let mut audio = settings::item::builder(fl!("audio")).description(match audio_error {
        Some((_, text)) => text.clone(),
        None => fl!("audio-detail"),
    });
    if audio_error.is_some() {
        audio = audio.icon(widget::icon::from_name("dialog-warning-symbolic").size(16));
    }
    let audio = audio.control(
        widget::row::with_capacity(3)
            .push(audio_path)
            .push(widget::button::standard(fl!("choose-folder")).on_press(Message::Pick(PickTarget::AudioFolder)))
            .push(widget::button::standard(fl!("choose-file")).on_press(Message::Pick(PickTarget::AudioFile)))
            .spacing(space.space_xs)
            .align_y(Alignment::Center),
    );

    let dataset = settings::section()
        .title(fl!("section-dataset"))
        .add(audio)
        .add(
            settings::item::builder(fl!("voice-name"))
                .description(fl!("voice-name-detail"))
                .control(field(app, Field::Name, &app.name, 240.0)),
        );

    let training = settings::section()
        .title(fl!("section-training"))
        .add(
            settings::item::builder(fl!("steps"))
                .description(fl!("steps-detail"))
                .control(field(app, Field::Steps, &app.steps, NUMBER_WIDTH)),
        )
        .add(
            settings::item::builder(fl!("save-every"))
                .description(fl!("save-every-detail"))
                .control(field(app, Field::SaveEvery, &app.save_every, NUMBER_WIDTH)),
        )
        .add(
            settings::item::builder(fl!("batch"))
                .description(fl!("batch-detail"))
                .control(field(app, Field::Batch, &app.batch, NUMBER_WIDTH)),
        )
        .add(
            settings::item::builder(fl!("fixes"))
                .description(fl!("fixes-detail"))
                .toggler(app.fixes, Message::Fixes),
        )
        .add(
            settings::item::builder(fl!("restart"))
                .description(fl!("restart-detail"))
                .toggler(app.restart, Message::Restart),
        );

    let hardware = settings::section()
        .title(fl!("section-hardware"))
        .add(
            settings::item::builder(fl!("base"))
                .description(fl!("base-detail"))
                .control(widget::dropdown(&app.base_labels, Some(app.base), Message::SelectBase)),
        )
        .add(
            settings::item::builder(fl!("gpu"))
                .description(fl!("gpu-detail"))
                .control(widget::dropdown(
                    &app.gpu_labels,
                    (!app.gpus.is_empty()).then_some(app.gpu),
                    Message::SelectGpu,
                )),
        );

    let mut install = settings::section().title(fl!("section-install")).add(
        settings::item::builder(fl!("install"))
            .description(fl!("install-detail"))
            .toggler(app.install, Message::Install),
    );
    if app.install {
        install = install.add(
            settings::item::builder(fl!("install-folder")).control(
                widget::row::with_capacity(2)
                    .push(
                        widget::text::body(app.app_models.as_str())
                            .width(Length::Fixed(PATH_WIDTH)),
                    )
                    .push(widget::button::standard(fl!("change")).on_press(Message::Pick(PickTarget::AppModels)))
                    .spacing(space.space_xs)
                    .align_y(Alignment::Center),
            ),
        );
    }

    settings::view_column(vec![dataset.into(), training.into(), hardware.into(), install.into()]).into()
}

fn builds_page(app: &App) -> Element<'_, Message> {
    let state = &app.state;
    let current = settings::section().title(fl!("builds-current")).add(match &state.current {
        Some(build) if state.running() => settings::item::builder(build.name.clone())
            .description(phase_text(state))
            .control(widget::text::caption(fl!("builds-steps", steps = build.steps))),
        _ => settings::item::builder(fl!("builds-none")).control(widget::Space::new()),
    });
    let mut queue = settings::section().title(fl!("builds-queue"));
    if state.queue.is_empty() {
        queue = queue.add(settings::item_row(vec![
            widget::text::body(fl!("builds-queue-empty")).into(),
        ]));
    }
    for (index, build) in state.queue.iter().enumerate() {
        queue = queue.add(
            settings::item::builder(format!("{}. {}", index + 1, build.name))
                .description(fl!("builds-steps", steps = build.steps))
                .control(
                    widget::button::standard(fl!("remove"))
                        .leading_icon(widget::icon::from_name("list-remove-symbolic"))
                        .on_press(Message::Remove(index)),
                ),
        );
    }
    settings::view_column(vec![
        current.into(),
        queue.into(),
        widget::text::caption(fl!("close-note")).into(),
    ])
    .into()
}

fn log_page(app: &App) -> Element<'_, Message> {
    let space = cosmic::theme::spacing();
    let text: Element<_> = if app.log.is_empty() {
        widget::text::body(fl!("log-empty")).into()
    } else {
        widget::text::monotext(app.log.as_str()).width(Length::Fill).into()
    };
    widget::column::with_capacity(2)
        .push(
            widget::row::with_capacity(2)
                .push(widget::space::horizontal())
                .push(widget::button::standard(fl!("clear-log")).on_press(Message::ClearLog)),
        )
        .push(
            widget::container(widget::scrollable(text).anchor_bottom().height(Length::Fill))
                .class(cosmic::theme::Container::Card)
                .padding(space.space_s)
                .height(Length::Fill),
        )
        .spacing(space.space_s)
        .padding([0, space.space_s, space.space_s, space.space_s])
        .height(Length::Fill)
        .into()
}

/// One line saying what the trainer is doing.
pub fn phase_text(state: &State) -> String {
    let name = state
        .current
        .as_ref()
        .map(|build| build.name.clone())
        .unwrap_or_default();
    match state.phase.as_str() {
        "starting" => fl!("phase-starting", name = name),
        "training" => fl!("phase-training", name = name),
        "stopping" => fl!("phase-stopping", name = name),
        "finished" => fl!("phase-finished", name = name),
        "stopped" => fl!("phase-stopped", name = name),
        "failed" => fl!(
            "phase-failed",
            name = name,
            code = state.exit_code.map(|code| code.to_string()).unwrap_or_default()
        ),
        _ => fl!("phase-idle"),
    }
}

pub fn control_bar(app: &App) -> Element<'_, Message> {
    let space = cosmic::theme::spacing();
    let state = &app.state;
    let primary = if state.running() {
        let stopping = state.phase == "stopping";
        widget::tooltip(
            widget::button::destructive(if stopping { fl!("end-now") } else { fl!("stop") })
                .leading_icon(widget::icon::from_name("media-playback-stop-symbolic"))
                .on_press(Message::Stop),
            widget::text::caption(fl!("stop-detail")),
            widget::tooltip::Position::Top,
        )
        .into()
    } else {
        Element::from(
            widget::button::suggested(fl!("start"))
                .leading_icon(widget::icon::from_name("media-playback-start-symbolic"))
                .on_press(Message::Start),
        )
    };
    let queue = widget::button::standard(fl!("queue-build"))
        .leading_icon(widget::icon::from_name("list-add-symbolic"))
        .on_press(Message::Queue);

    let mut status = widget::column::with_capacity(3)
        .push(widget::text::body(phase_text(state)))
        .spacing(space.space_xxs)
        .width(Length::Fill);
    if let Some(progress) = state.progress.as_ref().filter(|_| state.running()) {
        status = status
            .push(
                widget::progress_bar::linear::Linear::new()
                    .progress(progress.step as f32 / progress.total.max(1) as f32)
                    .girth(6.0)
                    .width(Length::Fill),
            )
            .push(widget::text::caption(format!(
                "{} · {}",
                fl!("progress", step = progress.step, total = progress.total),
                fl!(
                    "progress-times",
                    elapsed = progress.elapsed.clone(),
                    remaining = progress.remaining.clone(),
                    rate = format!("{:.2}", progress.rate)
                )
            )));
    } else if state.phase == "starting" {
        status = status.push(widget::progress_bar::indeterminate_linear().girth(6.0).width(Length::Fill));
    }

    widget::container(
        widget::row::with_capacity(3)
            .push(primary)
            .push(queue)
            .push(status)
            .spacing(space.space_s)
            .align_y(Alignment::Center),
    )
    .class(cosmic::theme::Container::Card)
    .padding(space.space_s)
    .into()
}
