//! The Python side (`python -m causalvc.trainer`), run as a child process.
//!
//! Commands go to its stdin and events come back on its stdout, one JSON object
//! per line (the protocol is described at the top of `causalvc/trainer.py`).
//! [`connection`] is an iced subscription that starts the process, restarts it
//! if it ends, and turns its lines into [`Event`]s. Dropping the process's stdin
//! (this app closing) makes it stop a running build after saving a checkpoint.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use cosmic::iced::futures::{SinkExt, Stream};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

const RESTART_DELAY: Duration = Duration::from_secs(3);

/// The training form, as saved from the last build.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Form {
    #[serde(default)]
    pub audio: String,
    #[serde(default)]
    pub name: String,
    #[serde(default = "default_batch")]
    pub batch: u64,
    #[serde(default = "default_steps")]
    pub steps: u64,
    #[serde(default = "default_steps")]
    pub save_every: u64,
    #[serde(default = "default_base")]
    pub base: String,
    #[serde(default)]
    pub device: String,
    #[serde(default)]
    pub app_models: String,
    #[serde(default)]
    pub noise: bool,
}

fn default_batch() -> u64 {
    8
}

fn default_steps() -> u64 {
    10_000
}

fn default_base() -> String {
    "default".to_string()
}

#[derive(Clone, Debug, Deserialize)]
pub struct Gpu {
    pub id: String,
    pub label: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Build {
    pub name: String,
    pub steps: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Progress {
    pub step: u64,
    pub total: u64,
    pub elapsed: String,
    pub remaining: String,
    pub rate: f64,
}

/// What the trainer is doing.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct State {
    pub phase: String,
    pub current: Option<Build>,
    #[serde(default)]
    pub queue: Vec<Build>,
    pub progress: Option<Progress>,
    pub exit_code: Option<i64>,
    /// The build that starts once the current one has saved and stopped (Build now).
    #[serde(default)]
    pub next: Option<Build>,
}

impl State {
    pub fn running(&self) -> bool {
        matches!(self.phase.as_str(), "starting" | "training" | "stopping")
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ErrorInfo {
    pub code: String,
    #[serde(default)]
    pub message: String,
    pub field: Option<String>,
}

#[derive(Clone, Debug)]
pub enum Event {
    /// The process is running; commands can be sent through the handle.
    Started(Handle),
    /// It could not be started, or it ended.
    Ended(String),
    Hello { form: Form, bases: Vec<String> },
    Gpus(Vec<Gpu>),
    State(State),
    Log(String),
    Error(ErrorInfo),
    /// A queued build is waiting; `position` 1 is next.
    Queued { name: String, position: u64 },
}

/// Sends commands to the trainer; cheap to clone.
#[derive(Clone, Debug)]
pub struct Handle {
    sender: mpsc::UnboundedSender<String>,
}

impl Handle {
    pub fn send(&self, cmd: &str, args: Value) {
        // A closed channel means the process is gone; the subscription reports that.
        let _ = self.sender.send(json!({"cmd": cmd, "args": args}).to_string());
    }
}

/// The repository: `CAUSALVC_DIR`, or the first ancestor of the executable or
/// working directory that holds `causalvc/trainer.py`.
pub fn project_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("CAUSALVC_DIR") {
        return Some(PathBuf::from(dir));
    }
    let starts = [std::env::current_exe().ok(), std::env::current_dir().ok()];
    starts.into_iter().flatten().find_map(|start| {
        start
            .ancestors()
            .find(|dir| dir.join("causalvc").join("trainer.py").is_file())
            .map(Path::to_path_buf)
    })
}

fn spawn() -> Result<tokio::process::Child, String> {
    let root = project_dir().ok_or_else(|| {
        "Could not find the project folder (causalvc/trainer.py). Set CAUSALVC_DIR.".to_string()
    })?;
    let python = root.join(".venv").join("bin").join("python");
    if !python.is_file() {
        return Err(format!(
            "Python environment not found: {}. Run ./run-training.sh once: it installs everything.",
            python.display()
        ));
    }
    tokio::process::Command::new(&python)
        .args(["-u", "-m", "causalvc.trainer"])
        .current_dir(&root)
        .env("PYTHONPATH", &root)
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("Could not start {}: {error}", python.display()))
}

pub fn parse_line(line: &str) -> Option<Event> {
    let message: Value = serde_json::from_str(line).ok()?;
    let event = message.get("event")?.as_str()?;
    let data = message.get("data").cloned().unwrap_or(Value::Null);
    Some(match event {
        "hello" => Event::Hello {
            form: serde_json::from_value(data.get("form")?.clone()).ok()?,
            bases: serde_json::from_value(data.get("bases")?.clone()).ok()?,
        },
        "gpus" => Event::Gpus(serde_json::from_value(data.get("gpus")?.clone()).ok()?),
        "state" => Event::State(serde_json::from_value(data).ok()?),
        "log" => Event::Log(data.get("text")?.as_str()?.to_string()),
        "error" => Event::Error(serde_json::from_value(data).ok()?),
        "queued" => Event::Queued {
            name: data.get("name")?.as_str()?.to_string(),
            position: data.get("position").and_then(Value::as_u64).unwrap_or(1),
        },
        _ => return None,
    })
}

/// Subscription stream: one trainer process at a time, restarted if it ends.
pub fn connection() -> impl Stream<Item = Event> {
    cosmic::iced::stream::channel(256, async |mut output| {
        loop {
            let mut child = match spawn() {
                Ok(child) => child,
                Err(error) => {
                    let _ = output.send(Event::Ended(error)).await;
                    tokio::time::sleep(RESTART_DELAY).await;
                    continue;
                }
            };
            let (Some(mut stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
                let _ = output.send(Event::Ended("no pipes to the trainer".into())).await;
                tokio::time::sleep(RESTART_DELAY).await;
                continue;
            };
            let (sender, mut receiver) = mpsc::unbounded_channel::<String>();
            let _ = output.send(Event::Started(Handle { sender })).await;
            let writer = tokio::spawn(async move {
                while let Some(mut line) = receiver.recv().await {
                    line.push('\n');
                    if stdin.write_all(line.as_bytes()).await.is_err() {
                        break;
                    }
                    let _ = stdin.flush().await;
                }
            });
            let mut lines = BufReader::new(stdout).lines();
            let reason = loop {
                match lines.next_line().await {
                    Ok(Some(line)) => {
                        if let Some(event) = parse_line(&line) {
                            let _ = output.send(event).await;
                        }
                    }
                    Ok(None) => break "The trainer stopped unexpectedly.".to_string(),
                    Err(error) => break error.to_string(),
                }
            };
            writer.abort();
            let _ = child.wait().await;
            let _ = output.send(Event::Ended(reason)).await;
            tokio::time::sleep(RESTART_DELAY).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_trainer_events() {
        match parse_line(
            r#"{"event":"hello","data":{"form":{"audio":"/a","name":"Kira","batch":8,"steps":100,"save_every":50,"base":"default","device":"cuda:0","fixes":true,"app_models":""},"bases":["default","Mine"],"gpus":[]}}"#,
        ) {
            Some(Event::Hello { form, bases }) => {
                assert_eq!(form.name, "Kira");
                assert_eq!(form.save_every, 50);
                assert_eq!(bases, ["default", "Mine"]);
            }
            other => panic!("unexpected {other:?}"),
        }
        match parse_line(
            r#"{"event":"state","data":{"phase":"training","current":{"name":"Kira","steps":100},"queue":[{"name":"B","steps":5}],"progress":{"step":10,"total":100,"percent":10,"elapsed":"0:05","remaining":"0:45","rate":2.0},"exit_code":null}}"#,
        ) {
            Some(Event::State(state)) => {
                assert!(state.running());
                assert_eq!(state.queue[0].name, "B");
                assert_eq!(state.progress.unwrap().step, 10);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(
            parse_line(r#"{"event":"error","data":{"code":"busy","message":"m","field":null}}"#),
            Some(Event::Error(ErrorInfo { code, .. })) if code == "busy"
        ));
        assert!(matches!(
            parse_line(r#"{"event":"gpus","data":{"gpus":[{"id":"cuda:0","label":"RTX"},{"id":"cpu","label":null}]}}"#),
            Some(Event::Gpus(gpus)) if gpus.len() == 2 && gpus[1].label.is_none()
        ));
        assert!(parse_line(r#"{"event":"later","data":{}}"#).is_none());
        assert!(parse_line("nope").is_none());
    }
}
