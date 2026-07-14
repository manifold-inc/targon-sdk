use std::io::IsTerminal;
use std::time::Duration;

use colored::Colorize;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::output::{palettes, style};

const TICK_MS: u64 = 80;

fn ticks() -> Vec<String> {
    ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✓"]
        .iter()
        .map(|t| t.color(palettes::ACCENT).to_string())
        .collect()
}

fn spinner_style() -> ProgressStyle {
    let ticks = ticks();
    let ticks: Vec<&str> = ticks.iter().map(String::as_str).collect();
    ProgressStyle::with_template("{spinner} {msg}")
        .unwrap()
        .tick_strings(&ticks)
}

pub struct Spinner {
    bar: ProgressBar,
    quiet: bool,
}

pub fn spinner(msg: impl Into<String>) -> Spinner {
    spinner_if(true, msg)
}

/// A spinner that renders only when `enabled` (pass `!ctx.json()`); indicatif
/// additionally hides it when stderr is not a TTY. Finish messages are
/// suppressed entirely when disabled.
pub fn spinner_if(enabled: bool, msg: impl Into<String>) -> Spinner {
    let bar = if enabled {
        ProgressBar::new_spinner()
    } else {
        ProgressBar::with_draw_target(None, ProgressDrawTarget::hidden())
    };
    bar.set_style(spinner_style());
    bar.set_message(msg.into());
    if enabled {
        bar.enable_steady_tick(Duration::from_millis(TICK_MS));
    }
    Spinner { bar, quiet: !enabled }
}

impl Spinner {
    pub fn set_message(&self, msg: impl Into<String>) {
        self.bar.set_message(msg.into());
    }

    pub fn finish_ok(self, msg: impl Into<String>) {
        self.bar.finish_and_clear();
        if !self.quiet {
            style::success(msg.into());
        }
    }

    pub fn finish_fail(self, msg: impl Into<String>) {
        self.bar.finish_and_clear();
        if !self.quiet {
            style::error(msg.into());
        }
    }
}

/// Live step checklist for provisioning flows: ✓ done, spinner active,
/// ○ pending. Each step collapses to its final line — no scrollback spam.
///
/// On a TTY all steps render up front; when stderr is piped (or in `--json`
/// mode) nothing animates and completed steps are logged as plain lines
/// (or not at all in `--json` mode).
pub struct Checklist {
    bars: Vec<ProgressBar>,
    labels: Vec<String>,
    live: bool,
    quiet: bool,
    _mp: MultiProgress,
}

impl Checklist {
    /// `enabled` should be `!ctx.json()`.
    pub fn new(enabled: bool, labels: &[&str]) -> Self {
        let live = enabled && std::io::stderr().is_terminal();
        let mp = if live {
            MultiProgress::new()
        } else {
            MultiProgress::with_draw_target(ProgressDrawTarget::hidden())
        };
        let pending_style = ProgressStyle::with_template("{msg}").unwrap();
        let bars = labels
            .iter()
            .map(|label| {
                let bar = mp.add(ProgressBar::new_spinner());
                bar.set_style(pending_style.clone());
                bar.set_message(pending_line(label));
                bar.tick();
                bar
            })
            .collect();
        Self {
            bars,
            labels: labels.iter().map(|l| l.to_string()).collect(),
            live,
            quiet: !enabled,
            _mp: mp,
        }
    }

    /// Activate a step: cyan spinner + label, optional dim detail.
    pub fn start(&self, idx: usize, detail: &str) {
        let bar = &self.bars[idx];
        bar.set_style(
            ProgressStyle::with_template("  {spinner} {msg}")
                .unwrap()
                .tick_strings(
                    &ticks().iter().map(String::as_str).collect::<Vec<_>>(),
                ),
        );
        bar.set_message(active_line(&self.labels[idx], detail));
        bar.enable_steady_tick(Duration::from_millis(TICK_MS));
    }

    /// Collapse a step to its ✓ line. `detail` may carry its own colors
    /// (e.g. a cyan UID) and is printed as-is.
    pub fn done(&self, idx: usize, label: &str, detail: &str) {
        let line = format!(
            "  {} {:<15} {}",
            style::TICK.color(palettes::SUCCESS),
            label,
            detail
        );
        self.finish_line(idx, line);
    }

    /// Collapse a step to its ✗ line.
    pub fn fail(&self, idx: usize, label: &str) {
        let line = format!(
            "  {} {}",
            style::CROSS.color(palettes::ERROR).bold(),
            label.color(palettes::ERROR)
        );
        self.finish_line(idx, line);
        // Remaining pending steps stay meaningless after a failure.
        for bar in &self.bars[idx + 1..] {
            bar.finish_and_clear();
        }
    }

    fn finish_line(&self, idx: usize, line: String) {
        let bar = &self.bars[idx];
        if self.live {
            bar.set_style(ProgressStyle::with_template("{msg}").unwrap());
            bar.finish_with_message(line);
        } else {
            bar.finish_and_clear();
            if !self.quiet {
                eprintln!("{line}");
            }
        }
    }
}

fn pending_line(label: &str) -> String {
    format!(
        "  {} {}",
        style::CIRCLE.color(palettes::BORDER),
        label.color(palettes::DIM)
    )
}

fn active_line(label: &str, detail: &str) -> String {
    if detail.is_empty() {
        label.to_string()
    } else {
        format!("{label}   {}", detail.color(palettes::DIM))
    }
}
