//! COSMIC front end for building CausalVC voices (`causalvc/trainer.py` does the work).

mod app;
mod i18n;
mod pages;
mod trainer;
mod update;

use cosmic::app::Settings;
use cosmic::iced::Size;

fn main() -> cosmic::iced::Result {
    i18n::init();
    let settings = Settings::default()
        .size(Size::new(900.0, 780.0))
        .size_limits(
            cosmic::iced::Limits::NONE
                .min_width(600.0)
                .min_height(480.0),
        );
    cosmic::app::run::<app::App>(settings, ())
}
