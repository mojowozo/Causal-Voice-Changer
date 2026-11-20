//! Application state and message handling.

use cosmic::app::{Core, Task};
use cosmic::dialog::file_chooser;
use cosmic::iced::Subscription;
use cosmic::prelude::*;
use cosmic::widget::{self, nav_bar, toaster};
use serde_json::{Value, json};

use crate::fl;
use crate::pages;
use crate::trainer::{self, ErrorInfo, Event, Form, Gpu, Handle, State};

pub const APP_ID: &str = "io.github.causalvc.Training";
const MAX_LOG_CHARS: usize = 60_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Page {
    Voice,
    Builds,
    Log,
    About,
}

/// The form's text fields; their names match the trainer's config keys.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Field {
    Name,
    Audio,
    Steps,
    SaveEvery,
    Batch,
    AppModels,
}

impl Field {
    pub fn key(self) -> &'static str {
        match self {
            Field::Name => "name",
            Field::Audio => "audio",
            Field::Steps => "steps",
            Field::SaveEvery => "save_every",
            Field::Batch => "batch",
            Field::AppModels => "app_models",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PickTarget {
    AudioFolder,
    AudioFile,
    AppModels,
}

/// The trainer process, as shown to the user.
#[derive(Clone, Debug)]
pub enum Connection {
    Starting,
    Ready,
    Failed(String),
}

#[derive(Clone, Debug)]
pub enum Message {
    Trainer(Event),
    /// The update window and settings.
    Update(crate::update::UpdateMsg),
    Text(Field, String),
    Restart(bool),
    Noise(bool),
    Colour(bool),
    Install(bool),
    SelectBase(usize),
    SelectGpu(usize),
    Pick(PickTarget),
    Picked(PickTarget, Option<String>),
    Start,
    Queue,
    Stop,
    Remove(usize),
    BuildNow(usize),
    /// Dragging a queued build by its handle: pressed on one row, over another, released.
    DragStart(usize),
    DragOver(usize),
    DragEnd,
    ClearLog,
    CloseToast(toaster::ToastId),
}

pub struct App {
    core: Core,
    nav: nav_bar::Model,
    handle: Option<Handle>,
    pub connection: Connection,
    pub state: State,
    pub log: String,
    toasts: toaster::Toasts<Message>,
    // The form.
    pub name: String,
    pub audio: String,
    pub steps: String,
    pub save_every: String,
    pub batch: String,
    pub restart: bool,
    pub noise: bool,
    pub colour: bool,
    pub install: bool,
    pub app_models: String,
    /// Base names from the trainer ("default" first) and their labels.
    pub bases: Vec<String>,
    pub base_labels: Vec<String>,
    pub base: usize,
    /// GPUs once the trainer has found them; `wanted_gpu` is the saved choice.
    pub gpus: Vec<Gpu>,
    pub gpu_labels: Vec<String>,
    pub gpu: usize,
    wanted_gpu: String,
    /// The field the trainer refused, with the reason, until it is edited.
    pub field_error: Option<(String, String)>,
    /// The queued build being dragged, and the row it would be dropped on.
    pub dragging: Option<usize>,
    pub drop_on: Option<usize>,
    /// New releases: the trainer's update state and whether the update window is open.
    pub updates: crate::update::Updates,
}

impl cosmic::Application for App {
    type Executor = cosmic::executor::Default;
    type Flags = ();
    type Message = Message;
    const APP_ID: &'static str = APP_ID;

    fn core(&self) -> &Core {
        &self.core
    }

    fn core_mut(&mut self) -> &mut Core {
        &mut self.core
    }

    fn init(core: Core, _flags: ()) -> (Self, Task<Message>) {
        let mut nav = nav_bar::Model::default();
        for (page, title, icon) in [
            (Page::Voice, fl!("page-voice"), "avatar-default-symbolic"),
            (Page::Builds, fl!("page-builds"), "view-list-symbolic"),
            (Page::Log, fl!("page-log"), "utilities-terminal-symbolic"),
            (Page::About, fl!("page-about"), "help-about-symbolic"),
        ] {
            nav.insert()
                .text(title)
                .icon(widget::icon::from_name(icon))
                .data(page);
        }
        nav.activate_position(0);
        let mut app = App {
            core,
            nav,
            handle: None,
            connection: Connection::Starting,
            state: State {
                phase: "idle".into(),
                ..State::default()
            },
            log: String::new(),
            toasts: toaster::Toasts::new(Message::CloseToast),
            name: String::new(),
            audio: String::new(),
            steps: "10000".into(),
            save_every: "1000".into(),
            batch: "8".into(),
            restart: false,
            noise: false,
            colour: true,
            install: true,
            app_models: String::new(),
            bases: vec!["default".into()],
            base_labels: vec![fl!("base-default")],
            base: 0,
            gpus: Vec::new(),
            gpu_labels: vec![fl!("gpu-loading")],
            gpu: 0,
            wanted_gpu: "cuda:0".into(),
            field_error: None,
            dragging: None,
            drop_on: None,
            updates: crate::update::Updates::default(),
        };
        let task = app.update_title();
        (app, task)
    }

    fn nav_model(&self) -> Option<&nav_bar::Model> {
        Some(&self.nav)
    }

    fn on_nav_select(&mut self, id: nav_bar::Id) -> Task<Message> {
        self.nav.activate(id);
        self.update_title()
    }

    fn subscription(&self) -> Subscription<Message> {
        use cosmic::iced::{Event as WindowEvent, event, mouse};
        let trainer = Subscription::run(trainer::connection).map(Message::Trainer);
        if self.dragging.is_none() {
            return trainer;
        }
        // The release ends a drag wherever it happens, also outside the rows.
        let release = event::listen_with(|event, _status, _window| match event {
            WindowEvent::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => Some(Message::DragEnd),
            _ => None,
        });
        Subscription::batch([trainer, release])
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Trainer(event) => return self.handle_trainer(event),
            Message::Update(msg) => {
                let effect = self.updates.on_msg(msg);
                return self.apply_update_effect(effect);
            }
            Message::Text(field, text) => {
                if self.field_error.as_ref().is_some_and(|(f, _)| f == field.key()) {
                    self.field_error = None;
                }
                *self.text_mut(field) = text;
            }
            Message::Restart(on) => self.restart = on,
            Message::Noise(on) => self.noise = on,
            Message::Colour(on) => self.colour = on,
            Message::Install(on) => self.install = on,
            Message::SelectBase(index) => self.base = index,
            Message::SelectGpu(index) => {
                self.gpu = index;
                if let Some(gpu) = self.gpus.get(index) {
                    self.wanted_gpu = gpu.id.clone();
                }
            }
            Message::Pick(target) => {
                let title = match target {
                    PickTarget::AudioFolder => fl!("pick-audio-folder"),
                    PickTarget::AudioFile => fl!("pick-audio-file"),
                    PickTarget::AppModels => fl!("pick-install-folder"),
                };
                return pick(target, title);
            }
            Message::Picked(_, None) => {}
            Message::Picked(target, Some(path)) => {
                let field = match target {
                    PickTarget::AudioFolder | PickTarget::AudioFile => Field::Audio,
                    PickTarget::AppModels => Field::AppModels,
                };
                return self.update(Message::Text(field, path));
            }
            Message::Start => self.send_build("start"),
            Message::Queue => self.send_build("queue"),
            Message::Stop => self.send("stop", json!({})),
            Message::Remove(index) => self.send("remove", json!({"index": index})),
            Message::BuildNow(index) => self.send("build_now", json!({"index": index})),
            Message::DragStart(index) => {
                self.dragging = Some(index);
                self.drop_on = Some(index);
            }
            Message::DragOver(index) => {
                if self.dragging.is_some() {
                    self.drop_on = Some(index);
                }
            }
            Message::DragEnd => {
                if let (Some(from), Some(to)) = (self.dragging.take(), self.drop_on.take()) {
                    if from != to && from < self.state.queue.len() && to < self.state.queue.len() {
                        self.send("move", json!({"from": from, "to": to}));
                    }
                }
            }
            Message::ClearLog => self.log.clear(),
            Message::CloseToast(id) => self.toasts.remove(id),
        }
        Task::none()
    }

    fn view(&self) -> Element<'_, Message> {
        let page = self.nav.active_data::<Page>().copied().unwrap_or(Page::Voice);
        widget::toaster(&self.toasts, pages::view(self, page))
    }

    fn dialog(&self) -> Option<Element<'_, Message>> {
        self.updates.dialog().map(|dialog| dialog.map(Message::Update))
    }

    fn footer(&self) -> Option<Element<'_, Message>> {
        matches!(self.connection, Connection::Ready).then(|| pages::control_bar(self))
    }
}

impl App {
    fn update_title(&mut self) -> Task<Message> {
        let page = self.nav.text(self.nav.active()).unwrap_or_default().to_string();
        self.set_header_title(page.clone());
        let title = format!("{page} — {}", fl!("app-title"));
        match self.core.main_window_id() {
            Some(id) => self.set_window_title(title, id),
            None => Task::none(),
        }
    }

    fn text_mut(&mut self, field: Field) -> &mut String {
        match field {
            Field::Name => &mut self.name,
            Field::Audio => &mut self.audio,
            Field::Steps => &mut self.steps,
            Field::SaveEvery => &mut self.save_every,
            Field::Batch => &mut self.batch,
            Field::AppModels => &mut self.app_models,
        }
    }

    fn send(&self, cmd: &str, args: Value) {
        if let Some(handle) = &self.handle {
            handle.send(cmd, args);
        }
    }

    /// The form as a build config; the trainer checks it.
    fn send_build(&mut self, cmd: &str) {
        let device = self
            .gpus
            .get(self.gpu)
            .map_or(self.wanted_gpu.clone(), |gpu| gpu.id.clone());
        self.send(
            cmd,
            json!({
                "name": self.name.trim(),
                "audio": self.audio.trim(),
                "steps": self.steps.trim(),
                "save_every": self.save_every.trim(),
                "batch": self.batch.trim(),
                "base": self.bases.get(self.base).cloned().unwrap_or_else(|| "default".into()),
                "device": device,
                // Not offered in the window: every voice trains with the quality fixes.
                "fixes": true,
                "noise": self.noise,
                "colour": self.colour,
                "restart": self.restart,
                "app_models": if self.install { self.app_models.trim() } else { "" },
            }),
        );
        // Like the Tk window: starting over applies to the build just sent.
        self.restart = false;
        self.field_error = None;
    }

    fn handle_trainer(&mut self, event: Event) -> Task<Message> {
        match event {
            Event::Started(handle) => {
                self.handle = Some(handle);
                self.connection = Connection::Ready;
            }
            Event::Ended(reason) => {
                self.handle = None;
                self.connection = Connection::Failed(reason);
            }
            Event::Hello { form, bases, update } => {
                self.apply_form(form, bases);
                if let Some(info) = update {
                    let effect = self.updates.on_info(info);
                    return self.apply_update_effect(effect);
                }
            }
            Event::Update(info) => {
                let effect = self.updates.on_info(info);
                return self.apply_update_effect(effect);
            }
            Event::Gpus(gpus) => {
                self.gpu_labels = gpus
                    .iter()
                    .map(|gpu| match &gpu.label {
                        Some(label) => format!("{label} ({})", gpu.id),
                        None if gpu.id == "cpu" => fl!("gpu-cpu"),
                        None => gpu.id.clone(),
                    })
                    .collect();
                self.gpu = gpus.iter().position(|gpu| gpu.id == self.wanted_gpu).unwrap_or(0);
                self.gpus = gpus;
            }
            Event::State(state) => {
                if state.queue.len() != self.state.queue.len() {
                    // The rows changed under the pointer: drop the drag rather than move the wrong build.
                    self.dragging = None;
                    self.drop_on = None;
                }
                let ended = self.state.running() && !state.running();
                self.state = state;
                if ended {
                    return self.toast(pages::phase_text(&self.state));
                }
            }
            Event::Log(text) => self.append_log(&text),
            Event::Queued { name, position } => {
                return self.toast(fl!("build-queued", name = name, position = position));
            }
            Event::Error(error) => {
                let text = error_text(&error);
                if let Some(field) = error.field.clone() {
                    self.field_error = Some((field, text.clone()));
                }
                return self.toast(text);
            }
        }
        Task::none()
    }

    fn apply_form(&mut self, form: Form, bases: Vec<String>) {
        self.base_labels = bases
            .iter()
            .map(|base| if base == "default" { fl!("base-default") } else { base.clone() })
            .collect();
        self.base = bases.iter().position(|base| *base == form.base).unwrap_or(0);
        self.bases = bases;
        // The trainer restarts after a crash: keep what was typed since.
        if !self.name.is_empty() || !self.audio.is_empty() {
            return;
        }
        self.name = form.name;
        self.audio = form.audio;
        self.steps = form.steps.to_string();
        self.save_every = form.save_every.to_string();
        self.batch = form.batch.to_string();
        self.noise = form.noise;
        self.colour = form.colour;
        self.install = !form.app_models.is_empty();
        if !form.app_models.is_empty() {
            self.app_models = form.app_models;
        } else if let Some(root) = trainer::project_dir() {
            self.app_models = root.join("realtime").join("models").to_string_lossy().into_owned();
        }
        if !form.device.is_empty() {
            self.wanted_gpu = form.device;
        }
    }

    fn apply_update_effect(&mut self, effect: crate::update::Effect) -> Task<Message> {
        use crate::update::Effect;
        match effect {
            Effect::None => Task::none(),
            Effect::Send(cmd, args) => {
                self.send(cmd, args);
                Task::none()
            }
            Effect::Install => {
                self.send("install_update", json!({ "pid": std::process::id() }));
                Task::none()
            }
            // The installer waits for this window and the trainer (which ends with it).
            Effect::Exit => cosmic::iced::exit(),
            Effect::Toast(text) => self.toast(text),
            Effect::Open(url) => match crate::update::open_url(&url) {
                Ok(()) => Task::none(),
                Err(error) => self.toast(error.to_string()),
            },
        }
    }

    fn toast(&mut self, text: String) -> Task<Message> {
        self.toasts
            .push(toaster::Toast::new(text))
            .map(cosmic::Action::App)
    }

    fn append_log(&mut self, text: &str) {
        self.log.push_str(text);
        self.log.push('\n');
        if self.log.len() > MAX_LOG_CHARS {
            let mut cut = self.log.len() - MAX_LOG_CHARS;
            while !self.log.is_char_boundary(cut) {
                cut += 1;
            }
            self.log.drain(..cut);
        }
    }
}

pub fn error_text(error: &ErrorInfo) -> String {
    match error.code.as_str() {
        "invalid_name" => fl!("error-invalid_name"),
        "missing_audio" => fl!("error-missing_audio"),
        "invalid_numbers" => fl!("error-invalid_numbers"),
        "busy" => fl!("error-busy"),
        _ => fl!("error-other", message = error.message.clone()),
    }
}

fn pick(target: PickTarget, title: String) -> Task<Message> {
    cosmic::task::future(async move {
        let dialog = file_chooser::open::Dialog::new().title(title);
        let response = match target {
            PickTarget::AudioFile => dialog.open_file().await,
            PickTarget::AudioFolder | PickTarget::AppModels => dialog.open_folder().await,
        };
        let path = response
            .ok()
            .and_then(|response| response.url().to_file_path().ok())
            .map(|path| path.to_string_lossy().into_owned());
        Message::Picked(target, path)
    })
}
