use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

use crate::output::style;

pub struct Spinner {
    bar: ProgressBar,
}

pub fn spinner(msg: impl Into<String>) -> Spinner {
    let bar = ProgressBar::new_spinner();
    bar.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✓"]),
    );
    bar.set_message(msg.into());
    bar.enable_steady_tick(Duration::from_millis(80));
    Spinner { bar }
}

impl Spinner {
    pub fn set_message(&self, msg: impl Into<String>) {
        self.bar.set_message(msg.into());
    }

    pub fn finish_ok(self, msg: impl Into<String>) {
        self.bar.finish_and_clear();
        style::success(msg.into());
    }

    pub fn finish_fail(self, msg: impl Into<String>) {
        self.bar.finish_and_clear();
        style::error(msg.into());
    }
}
