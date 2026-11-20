//! The update window and the update settings, word for word the same in the realtime app
//! (realtime/ui/src/update.rs) and the training app (training/ui/src/update.rs): keep both copies equal.
//!
//! The Python side (causalvc/update.py, behind the engine or the trainer) checks GitHub, downloads
//! and installs; it reports every change as an `update` event carrying an [`UpdateInfo`]. This module
//! shows that and turns the user's choices into commands for it.

use cosmic::iced::Length;
use cosmic::prelude::*;
use cosmic::widget::{self, settings};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::fl;

/// The backend's update state (see causalvc.update.Updater).
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct UpdateInfo {
    /// idle, checking, none, available, downloading, ready, installing, error, failed.
    pub state: String,
    pub current: String,
    /// This folder is a git checkout: updates come with git pull, not from here.
    pub git: bool,
    pub check_at_launch: bool,
    pub version: String,
    pub name: String,
    pub notes: String,
    pub url: String,
    pub progress: f32,
    pub error: String,
    /// The last check was asked for (Check for updates), not the one at launch.
    pub manual: bool,
}

#[derive(Clone, Debug)]
pub enum UpdateMsg {
    /// Download and install (or, in a git checkout, open the release page).
    Update,
    /// Ask me next time.
    Later,
    /// Skip this version.
    Skip,
    /// Check for updates now.
    Check,
    /// Check for updates at launch.
    SetCheck(bool),
    OpenPage,
    /// Close the window (a download carries on).
    Dismiss,
}

/// What the app does next.
pub enum Effect {
    None,
    /// Send this command to the backend.
    Send(&'static str, Value),
    /// The update is downloaded: send `install_update` with this process's id.
    Install,
    /// The installer has started and waits for this window: close it.
    Exit,
    Toast(String),
    /// Open this address in the browser.
    Open(String),
}

#[derive(Default)]
pub struct Updates {
    pub info: Option<UpdateInfo>,
    /// The update window is open.
    pub open: bool,
}

impl Updates {
    /// A new state from the backend (in its hello, or an `update` event).
    pub fn on_info(&mut self, info: UpdateInfo) -> Effect {
        let changed = self.info.as_ref().map(|old| old.state.as_str()) != Some(info.state.as_str());
        let effect = if !changed {
            Effect::None
        } else {
            match info.state.as_str() {
                "available" | "failed" => {
                    self.open = true;
                    Effect::None
                }
                "ready" => Effect::Install,
                "installing" => Effect::Exit,
                "none" if info.manual => {
                    self.open = false;
                    Effect::Toast(fl!("update-none", version = info.current.clone()))
                }
                "error" => {
                    self.open = false;
                    Effect::Toast(info.error.clone())
                }
                _ => Effect::None,
            }
        };
        self.info = Some(info);
        effect
    }

    pub fn on_msg(&mut self, msg: UpdateMsg) -> Effect {
        let (version, url, git) = self
            .info
            .as_ref()
            .map(|info| (info.version.clone(), info.url.clone(), info.git))
            .unwrap_or_default();
        match msg {
            UpdateMsg::Update if git => {
                self.open = false;
                Effect::Open(url)
            }
            UpdateMsg::Update => Effect::Send("download_update", json!({})),
            UpdateMsg::Later => {
                self.open = false;
                Effect::Send("later_update", json!({}))
            }
            UpdateMsg::Skip => {
                self.open = false;
                Effect::Send("skip_update", json!({ "version": version }))
            }
            UpdateMsg::Check => Effect::Send("check_update", json!({ "manual": true })),
            UpdateMsg::SetCheck(on) => {
                if let Some(info) = &mut self.info {
                    info.check_at_launch = on;
                }
                Effect::Send("set_update_check", json!({ "on": on }))
            }
            UpdateMsg::OpenPage => Effect::Open(url),
            UpdateMsg::Dismiss => {
                self.open = false;
                Effect::None
            }
        }
    }

    pub fn dialog(&self) -> Option<Element<'_, UpdateMsg>> {
        if !self.open {
            return None;
        }
        let info = self.info.as_ref()?;
        let dialog = match info.state.as_str() {
            "available" => {
                let notes = (!info.notes.is_empty()).then(|| {
                    widget::container(widget::scrollable(
                        widget::text::body(info.notes.as_str()).width(Length::Fill),
                    ))
                    .max_height(240.0)
                });
                let mut dialog = widget::dialog()
                    .title(fl!("update-available-title", version = info.version.clone()))
                    .body(if info.git {
                        fl!("update-available-git", current = info.current.clone())
                    } else {
                        fl!("update-available-body", current = info.current.clone())
                    })
                    .primary_action(
                        widget::button::suggested(if info.git {
                            fl!("update-open-page")
                        } else {
                            fl!("update-now")
                        })
                        .on_press(UpdateMsg::Update),
                    )
                    .secondary_action(widget::button::standard(fl!("update-later")).on_press(UpdateMsg::Later))
                    .tertiary_action(widget::button::text(fl!("update-skip")).on_press(UpdateMsg::Skip));
                if let Some(notes) = notes {
                    dialog = dialog.control(notes);
                }
                dialog
            }
            "downloading" | "ready" | "installing" => widget::dialog()
                .title(fl!("update-downloading", version = info.version.clone()))
                .body(if info.state == "downloading" {
                    fl!("update-downloading-body")
                } else {
                    fl!("update-restarting")
                })
                .control(widget::progress_bar::linear::Linear::new().progress(info.progress).girth(6.0))
                .secondary_action(widget::button::standard(fl!("update-hide")).on_press(UpdateMsg::Dismiss)),
            "failed" => widget::dialog()
                .title(fl!("update-failed-title"))
                .body(info.error.as_str())
                .primary_action(widget::button::suggested(fl!("update-try-again")).on_press(UpdateMsg::Check))
                .secondary_action(widget::button::standard(fl!("update-close")).on_press(UpdateMsg::Dismiss)),
            _ => return None,
        };
        Some(dialog.into())
    }

    /// The settings: this version, checking at launch, and checking now.
    pub fn section(&self) -> Element<'_, UpdateMsg> {
        let info = self.info.clone().unwrap_or_default();
        let known = self.info.is_some();
        let busy = matches!(info.state.as_str(), "checking" | "downloading" | "ready" | "installing");
        let percent = (info.progress * 100.0).round() as i64;
        let status = match info.state.as_str() {
            _ if !known => fl!("update-status-unknown"),
            "checking" => fl!("update-status-checking"),
            "available" => fl!("update-status-available", version = info.version.clone()),
            "downloading" => fl!("update-status-downloading", percent = percent),
            "none" => fl!("update-none", version = info.current.clone()),
            "error" | "failed" => info.error.clone(),
            _ => fl!("update-status-idle"),
        };
        let current = if known { info.current.clone() } else { String::from("—") };
        settings::section()
            .title(fl!("section-updates"))
            .add(settings::item(fl!("update-version"), widget::text::body(current)))
            .add(
                settings::item::builder(fl!("update-check-at-launch"))
                    .description(fl!("update-check-at-launch-detail"))
                    .toggler(info.check_at_launch, UpdateMsg::SetCheck),
            )
            .add(
                settings::item::builder(fl!("update-check"))
                    .description(status)
                    .control(
                        widget::row::with_capacity(2)
                            .push_maybe((!info.url.is_empty()).then(|| {
                                widget::button::text(fl!("update-open-page")).on_press(UpdateMsg::OpenPage)
                            }))
                            .push(
                                widget::button::standard(fl!("update-check-button"))
                                    .on_press_maybe((known && !busy).then_some(UpdateMsg::Check)),
                            )
                            .spacing(cosmic::theme::spacing().space_xs),
                    ),
            )
            .into()
    }
}

/// Open an address in the default browser.
pub fn open_url(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    let mut command = std::process::Command::new("explorer");
    #[cfg(not(target_os = "windows"))]
    let mut command = std::process::Command::new("xdg-open");
    command.arg(url).spawn().map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(state: &str, manual: bool) -> UpdateInfo {
        UpdateInfo { state: state.into(), version: "v9.0.0".into(), current: "v1.0.0".into(), manual, ..Default::default() }
    }

    #[test]
    fn a_found_update_opens_the_window_once() {
        let mut updates = Updates::default();
        assert!(matches!(updates.on_info(info("available", false)), Effect::None));
        assert!(updates.open && updates.dialog().is_some());
        assert!(matches!(updates.on_msg(UpdateMsg::Later), Effect::Send("later_update", _)));
        assert!(!updates.open);
        updates.on_info(info("available", false)); // the same state again: stays closed
        assert!(!updates.open);
    }

    #[test]
    fn skip_names_the_version_and_downloaded_means_install() {
        let mut updates = Updates::default();
        updates.on_info(info("available", false));
        match updates.on_msg(UpdateMsg::Skip) {
            Effect::Send("skip_update", args) => assert_eq!(args["version"], "v9.0.0"),
            _ => panic!("skip should send skip_update"),
        }
        assert!(matches!(updates.on_info(info("ready", false)), Effect::Install));
        assert!(matches!(updates.on_info(info("installing", false)), Effect::Exit));
    }

    #[test]
    fn up_to_date_is_only_said_when_asked() {
        let mut updates = Updates::default();
        assert!(matches!(updates.on_info(info("none", false)), Effect::None));
        let mut asked = Updates::default();
        assert!(matches!(asked.on_info(info("none", true)), Effect::Toast(_)));
    }

    #[test]
    fn a_git_checkout_opens_the_release_page_instead() {
        let mut updates = Updates::default();
        updates.on_info(UpdateInfo { git: true, url: "https://x".into(), ..info("available", false) });
        assert!(matches!(updates.on_msg(UpdateMsg::Update), Effect::Open(url) if url == "https://x"));
    }

    #[test]
    fn backend_json_parses_with_missing_fields() {
        let parsed: UpdateInfo =
            serde_json::from_value(json!({"state": "available", "version": "v2", "progress": 0.5})).unwrap();
        assert_eq!(parsed.state, "available");
        assert!(!parsed.check_at_launch);
    }
}
