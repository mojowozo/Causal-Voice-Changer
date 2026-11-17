//! The model library page: search, models grouped by voice, delete.
//!
//! Model folders are usually named like `Sakura-shortstream-50m-1.6_200e_14200s`:
//! the voice, a description of the training run, then the epochs and steps
//! of the checkpoint.  Grouping by voice puts several checkpoints of one
//! voice side by side: the engine names each model's voice (its folder,
//! `models/Kira/Kira-1000s.cvc`); a name is parsed only for an older engine.

use cosmic::iced::{Alignment, Length};
use cosmic::prelude::*;
use cosmic::widget::{self, settings};

use crate::app::{App, Message};
use crate::engine::{Model, State};
use crate::fl;

/// A model folder name split into its parts.
#[derive(Debug, PartialEq)]
pub struct ModelName<'a> {
    pub voice: &'a str,
    /// The rest of the name without the epoch/step suffix (may be empty).
    pub detail: &'a str,
    pub epochs: Option<u32>,
}

/// Split `Voice-detail_200e_14200s` into voice, detail and epochs.  Names
/// that do not follow the pattern become their own voice.
pub fn parse_model_name(name: &str) -> ModelName<'_> {
    let mut base = name;
    let mut epochs = None;
    // Trailing `_<n>e_<n>s` (epochs and steps), a common way to name checkpoints.
    let mut parts = name.rsplitn(3, '_');
    if let (Some(steps), Some(epoch), Some(rest)) = (parts.next(), parts.next(), parts.next()) {
        let number = |text: &str, unit: char| {
            text.strip_suffix(unit)
                .filter(|digits| !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()))
                .and_then(|digits| digits.parse::<u32>().ok())
        };
        if let (Some(count), Some(_)) = (number(epoch, 'e'), number(steps, 's')) {
            base = rest;
            epochs = Some(count);
        }
    }
    let split = base.find(['-', '_', ' ']).filter(|&at| at > 0);
    let (voice, detail) = match split {
        Some(at) => (&base[..at], &base[at + 1..]),
        None => (base, ""),
    };
    ModelName {
        voice,
        detail,
        epochs,
    }
}

/// Every whitespace-separated word of the query appears in the name.
pub fn matches(name: &str, query: &str) -> bool {
    let name = name.to_lowercase();
    query
        .to_lowercase()
        .split_whitespace()
        .all(|word| name.contains(word))
}

/// The voice a model belongs to: the engine's, else the first part of its name.
pub fn voice_of(model: &Model) -> &str {
    if model.voice.is_empty() {
        parse_model_name(&model.name).voice
    } else {
        &model.voice
    }
}

/// A short name for a checkpoint within its voice: `10000 steps` for `Kira-10000s`,
/// else the part of its name after the voice.
pub fn checkpoint_label(model: &Model) -> String {
    let voice = voice_of(model);
    let rest = model
        .name
        .get(..voice.len())
        .filter(|start| start.eq_ignore_ascii_case(voice))
        .map(|_| model.name[voice.len()..].trim_start_matches(['-', '_', ' ']))
        .unwrap_or(&model.name);
    match model.step {
        Some(step) if rest == format!("{step}s") => fl!("checkpoint-step", step = step),
        _ => {
            let parsed = parse_model_name(&model.name);
            match (parsed.detail.is_empty(), parsed.epochs) {
                (false, Some(epochs)) => fl!("library-checkpoint", detail = parsed.detail, epochs = epochs),
                (true, Some(epochs)) => fl!("library-epochs", epochs = epochs),
                _ if !rest.is_empty() => rest.to_string(),
                _ => model.name.clone(),
            }
        }
    }
}

/// Models matching `query` (by name or voice), grouped by voice.  `newest` sorts groups and
/// their models by date (newest first); otherwise by name, and checkpoints by step.
pub fn grouped<'a>(models: &'a [Model], query: &str, newest: bool) -> Vec<(String, Vec<&'a Model>)> {
    let mut groups: Vec<(String, Vec<&Model>)> = Vec::new();
    for model in models
        .iter()
        .filter(|model| matches(&format!("{} {}", voice_of(model), model.name), query))
    {
        let voice = voice_of(model);
        match groups
            .iter_mut()
            .find(|(name, _)| name.eq_ignore_ascii_case(voice))
        {
            Some((_, members)) => members.push(model),
            None => groups.push((voice.to_string(), vec![model])),
        }
    }
    for (_, members) in &mut groups {
        if newest {
            members.sort_by_key(|model| std::cmp::Reverse(model.modified));
        } else {
            members.sort_by(|a, b| {
                let (left, right) = (parse_model_name(&a.name), parse_model_name(&b.name));
                (a.step.is_none(), a.step)
                    .cmp(&(b.step.is_none(), b.step))
                    .then(left.detail.to_lowercase().cmp(&right.detail.to_lowercase()))
                    .then(left.epochs.cmp(&right.epochs))
            });
        }
    }
    if newest {
        let latest = |members: &[&Model]| members.iter().map(|m| m.modified).max().unwrap_or(0);
        groups.sort_by_key(|(_, members)| std::cmp::Reverse(latest(members)));
    } else {
        groups.sort_by_key(|(voice, _)| voice.to_lowercase());
    }
    groups
}

pub fn format_size(bytes: u64) -> String {
    let megabytes = bytes as f64 / 1_000_000.0;
    if megabytes >= 1000.0 {
        format!("{:.1} GB", megabytes / 1000.0)
    } else {
        format!("{megabytes:.0} MB")
    }
}

/// `YYYY-MM-DD` (UTC) of a Unix time.
pub fn format_date(unix_seconds: i64) -> String {
    // Howard Hinnant's days-to-civil algorithm.
    let days = unix_seconds.div_euclid(86_400) + 719_468;
    let era = days.div_euclid(146_097);
    let day_of_era = days.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 { month_index + 3 } else { month_index - 9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

pub fn page<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let space = cosmic::theme::spacing();
    let newest = app.library_sort == 1;
    let groups = grouped(&state.models, &app.library_query, newest);
    let shown: usize = groups.iter().map(|(_, members)| members.len()).sum();
    let converting_with =
        (state.running && state.function == "vc").then_some(state.settings.model_name.as_str());

    let toolbar = widget::row::with_capacity(4)
        .push(
            widget::search_input(fl!("library-search"), app.library_query.as_str())
                .on_input(Message::LibrarySearch)
                .on_clear(Message::LibrarySearch(String::new()))
                .width(Length::Fill),
        )
        .push(widget::dropdown(
            &app.library_sort_labels,
            Some(app.library_sort),
            Message::LibrarySort,
        ))
        .push(
            widget::button::standard(fl!("reload"))
                .leading_icon(widget::icon::from_name("view-refresh-symbolic"))
                .on_press(Message::ReloadModels),
        )
        .spacing(space.space_xs)
        .align_y(Alignment::Center);

    let mut sections: Vec<Element<'a, Message>> = vec![
        toolbar.into(),
        widget::text::caption(fl!(
            "library-count",
            shown = shown,
            total = state.models.len()
        ))
        .into(),
    ];
    if state.models.is_empty() {
        sections.push(widget::text::body(fl!("models-empty")).into());
    } else if groups.is_empty() {
        sections.push(widget::text::body(fl!("library-no-match")).into());
    }
    for (voice, members) in groups {
        let mut section = settings::section().title(fl!(
            "library-voice",
            voice = voice,
            count = members.len()
        ));
        for model in members {
            section = section.add(model_row(model, state, converting_with));
        }
        sections.push(section.into());
    }
    settings::view_column(sections).into()
}

fn model_row<'a>(
    model: &'a Model,
    state: &'a State,
    converting_with: Option<&str>,
) -> Element<'a, Message> {
    let space = cosmic::theme::spacing();
    let title = checkpoint_label(model);
    let description = format!(
        "{} · {}",
        format_size(model.size_bytes),
        format_date(model.modified)
    );
    let selected = model.name == state.settings.model_name;
    let in_use = converting_with == Some(model.name.as_str());

    let use_button: Element<_> = if selected {
        widget::button::standard(fl!("library-selected")).into()
    } else {
        widget::button::standard(fl!("library-use"))
            .on_press(Message::SelectModelByName(model.name.clone()))
            .into()
    };
    let delete_tip = if in_use {
        fl!("library-delete-in-use")
    } else {
        fl!("library-delete")
    };
    let rename_tip = if in_use {
        fl!("library-rename-in-use")
    } else {
        fl!("library-rename")
    };
    let controls = widget::row::with_capacity(6)
        .push(use_button)
        .push(widget::tooltip(
            widget::button::icon(widget::icon::from_name("document-edit-symbolic"))
                .on_press_maybe((!in_use).then(|| Message::AskRenameModel(model.name.clone()))),
            widget::text::caption(rename_tip),
            widget::tooltip::Position::Top,
        ))
        .push(widget::tooltip(
            widget::button::icon(widget::icon::from_name("document-save-as-symbolic"))
                .on_press(Message::ExportModel(model.name.clone())),
            widget::text::caption(fl!("library-export")),
            widget::tooltip::Position::Top,
        ))
        .push(widget::tooltip(
            widget::button::icon(widget::icon::from_name("folder-open-symbolic"))
                .on_press(Message::OpenModelFolder(model.name.clone())),
            widget::text::caption(fl!("library-open-folder")),
            widget::tooltip::Position::Top,
        ))
        .push(widget::tooltip(
            widget::button::icon(widget::icon::from_name("user-trash-symbolic"))
                .class(cosmic::theme::Button::Destructive)
                .on_press_maybe((!in_use).then(|| Message::AskDeleteModel(model.name.clone()))),
            widget::text::caption(delete_tip),
            widget::tooltip::Position::Top,
        ))
        .spacing(space.space_xxs)
        .align_y(Alignment::Center);

    let mut item = settings::item::builder(title).description(description);
    if selected {
        item = item.icon(widget::icon::from_name("object-select-symbolic").size(16));
    }
    widget::tooltip(
        item.control(controls),
        widget::text::caption(model.name.as_str()),
        widget::tooltip::Position::Bottom,
    )
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str, modified: i64) -> Model {
        Model {
            name: name.to_string(),
            model_file: format!("{name}.causalvc.json"),
            size_bytes: 56_223_809,
            modified,
            merge_family: None,
            has_index: false,
            voice: String::new(),
            step: None,
        }
    }

    fn checkpoint(name: &str, voice: &str, step: Option<u64>) -> Model {
        Model {
            voice: voice.to_string(),
            step,
            ..model(name, 0)
        }
    }

    #[test]
    fn groups_by_the_engines_voice_and_orders_checkpoints_by_step() {
        let models = vec![
            checkpoint("Jett-10000s", "Jett", Some(10000)),
            checkpoint("Jett-ema-4000s", "Jett-ema", Some(4000)),
            checkpoint("Jett-best", "Jett", None),
            checkpoint("Jett-2000s", "Jett", Some(2000)),
        ];
        let groups = grouped(&models, "", false);
        let names: Vec<(&str, Vec<&str>)> = groups
            .iter()
            .map(|(voice, members)| (voice.as_str(), members.iter().map(|m| m.name.as_str()).collect()))
            .collect();
        assert_eq!(
            names,
            vec![
                ("Jett", vec!["Jett-2000s", "Jett-10000s", "Jett-best"]),
                ("Jett-ema", vec!["Jett-ema-4000s"]),
            ]
        );
        // Searching for the voice finds all of its checkpoints.
        assert_eq!(grouped(&models, "jett-ema", false)[0].1.len(), 1);
        assert_eq!(grouped(&models, "jett", false).len(), 2);
    }

    #[test]
    fn labels_checkpoints_within_their_voice() {
        let plain = |text: String| text.replace(['\u{2068}', '\u{2069}'], "");
        assert_eq!(plain(checkpoint_label(&checkpoint("Jett-10000s", "Jett", Some(10000)))), "10000 steps");
        assert_eq!(checkpoint_label(&checkpoint("Jett-best", "Jett", None)), "best");
        assert_eq!(checkpoint_label(&checkpoint("P89", "P89", None)), "P89");
    }

    #[test]
    fn splits_trainer_names_into_voice_detail_and_epochs() {
        assert_eq!(
            parse_model_name("Sakura-shortstream-50m-1.6_200e_14200s"),
            ModelName {
                voice: "Sakura",
                detail: "shortstream-50m-1.6",
                epochs: Some(200)
            }
        );
        assert_eq!(
            parse_model_name("Lex-100Lola50Lydi15-hfg"),
            ModelName {
                voice: "Lex",
                detail: "100Lola50Lydi15-hfg",
                epochs: None
            }
        );
        assert_eq!(
            parse_model_name("MyVoice"),
            ModelName {
                voice: "MyVoice",
                detail: "",
                epochs: None
            }
        );
        // Not an epoch/step suffix: kept in the detail.
        assert_eq!(parse_model_name("Kye_v2_final").detail, "v2_final");
        assert_eq!(parse_model_name("Kye_40e_7600s").epochs, Some(40));
    }

    #[test]
    fn search_needs_every_word() {
        assert!(matches("Sakura-shortstream-50m-1.6_200e_14200s", "sakura 200e"));
        assert!(!matches("Sakura-shortstream-50m-1.6_120e_8520s", "sakura 200e"));
        assert!(matches("Anything", "  "));
    }

    #[test]
    fn groups_checkpoints_of_one_voice_together() {
        let models = vec![
            model("Sakura-shortstream-50m-1.6_200e_14200s", 30),
            model("Jett-comb-longer-1.6_200e_15800s", 10),
            model("sakura-shortstream-50m-1.6_120e_8520s", 20),
        ];
        let by_name = grouped(&models, "", false);
        let names: Vec<(&str, Vec<&str>)> = by_name
            .iter()
            .map(|(voice, members)| (voice.as_str(), members.iter().map(|m| m.name.as_str()).collect()))
            .collect();
        assert_eq!(
            names,
            vec![
                ("Jett", vec!["Jett-comb-longer-1.6_200e_15800s"]),
                (
                    "Sakura",
                    vec![
                        "sakura-shortstream-50m-1.6_120e_8520s",
                        "Sakura-shortstream-50m-1.6_200e_14200s"
                    ]
                ),
            ]
        );
        let newest = grouped(&models, "", true);
        assert_eq!(newest[0].0, "Sakura");
        assert_eq!(newest[0].1[0].modified, 30);
        assert_eq!(grouped(&models, "jett", false).len(), 1);
    }

    #[test]
    fn formats_sizes_and_dates() {
        assert_eq!(format_size(56_223_809), "56 MB");
        assert_eq!(format_size(1_500_000_000), "1.5 GB");
        assert_eq!(format_date(0), "1970-01-01");
        assert_eq!(format_date(1_791_067_045), "2026-10-03");
        assert_eq!(format_date(951_782_400), "2000-02-29");
    }
}
