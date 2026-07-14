use chrono::{DateTime, Utc};
use colored::{Color, Colorize};
use serde::Serialize;

use crate::error::{CliError, Result};
use crate::output::{palettes, style};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Human,
    Json,
}

impl OutputFormat {
    pub fn is_json(self) -> bool {
        matches!(self, OutputFormat::Json)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateKind {
    Ok,
    Pending,
    Bad,
    Idle,
    Unknown,
}

pub fn classify(status: &str) -> StateKind {
    match status.to_ascii_lowercase().as_str() {
        "running" | "ready" | "active" | "verified" => StateKind::Ok,
        "provisioning" | "registered" | "pending" | "starting" | "creating" => StateKind::Pending,
        "error" | "failed" => StateKind::Bad,
        "suspended" | "deleted" | "stopped" => StateKind::Idle,
        _ => StateKind::Unknown,
    }
}

pub fn state_color(kind: StateKind) -> Color {
    match kind {
        StateKind::Ok => palettes::SUCCESS,
        StateKind::Pending => palettes::WARN,
        StateKind::Bad => palettes::ERROR,
        StateKind::Idle => palettes::WARN,
        StateKind::Unknown => palettes::DIM,
    }
}

/// Colored dot + state word, e.g. `● running` in green.
pub fn state_badge(status: &str) -> String {
    let color = state_color(classify(status));
    format!(
        "{} {}",
        style::DOT.color(color),
        status.to_lowercase().color(color)
    )
}

/// Workload type in its own hue (RENTAL blue, VM magenta).
pub fn type_badge(workload_type: &str) -> String {
    workload_type
        .color(palettes::workload_type_color(workload_type))
        .to_string()
}

/// Credits with semantic thresholds: green normally, yellow under $25,
/// red at zero.
pub fn credits_badge(amount: f64, currency: &str) -> String {
    let color = if amount <= 0.0 {
        palettes::ERROR
    } else if amount < 25.0 {
        palettes::WARN
    } else {
        palettes::SUCCESS
    };
    format!("{amount:.2} {currency}").color(color).bold().to_string()
}

pub fn print_json<T: Serialize>(value: &T) -> Result<()> {
    let rendered = serde_json::to_string_pretty(value)
        .map_err(|e| CliError::Config(format!("failed to encode json: {e}")))?;
    println!("{rendered}");
    Ok(())
}

pub fn cost(value: f64) -> String {
    format!("${value:.2}")
}

pub fn cost_per_hour(value: f64) -> String {
    format!("${value:.2}/hr")
}

pub fn relative_time(t: DateTime<Utc>) -> String {
    let secs = Utc::now().signed_duration_since(t).num_seconds();
    if secs < 0 {
        return "just now".to_string();
    }
    if secs < 60 {
        return format!("{secs}s ago");
    }
    let mins = secs / 60;
    if mins < 60 {
        return format!("{mins}m ago");
    }
    let hours = mins / 60;
    if hours < 24 {
        return format!("{hours}h ago");
    }
    let days = hours / 24;
    if days < 30 {
        return format!("{days}d ago");
    }
    if days < 365 {
        return format!("{}mo ago", days / 30);
    }
    format!("{}y ago", days / 365)
}

pub fn short_uid(uid: &str) -> String {
    let max = 14;
    if uid.chars().count() > max {
        let head: String = uid.chars().take(max).collect();
        format!("{head}…")
    } else {
        uid.to_string()
    }
}

pub fn mib_to_human(mib: u64) -> String {
    if mib >= 1024 {
        format!("{:.0} GiB", mib as f64 / 1024.0)
    } else {
        format!("{mib} MiB")
    }
}

pub fn gpu_spec(gpu_type: Option<&str>, count: u32) -> String {
    match gpu_type {
        Some(t) if !t.is_empty() && count > 0 => format!("{count}x {t}"),
        Some(t) if !t.is_empty() => t.to_string(),
        _ => "-".to_string(),
    }
}
