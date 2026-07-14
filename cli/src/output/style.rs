use colored::Colorize;

use crate::output::palettes;

pub const TICK: &str = "✓";
pub const CROSS: &str = "✗";
pub const ARROW: &str = "→";
pub const BULLET: &str = "•";
pub const DOT: &str = "●";
pub const CIRCLE: &str = "○";
pub const SEP: &str = "·";

pub fn success(msg: impl AsRef<str>) {
    eprintln!("{} {}", TICK.color(palettes::SUCCESS).bold(), msg.as_ref());
}

pub fn error(msg: impl AsRef<str>) {
    eprintln!(
        "{} {}",
        CROSS.color(palettes::ERROR).bold(),
        msg.as_ref().color(palettes::ERROR)
    );
}

/// Recovery line printed under an error: dim arrow + explanation, with the
/// part after the last `: ` (usually the exact command to run) in normal
/// text so the fix reads calm — never red.
pub fn hint(msg: impl AsRef<str>) {
    let msg = msg.as_ref();
    match msg.rsplit_once(": ") {
        Some((prefix, command)) if !command.is_empty() => eprintln!(
            "  {} {} {}",
            ARROW.color(palettes::DIM),
            format!("{prefix}:").color(palettes::DIM),
            command
        ),
        _ => eprintln!(
            "  {} {}",
            ARROW.color(palettes::DIM),
            msg.color(palettes::DIM)
        ),
    }
}

pub fn warn(msg: impl AsRef<str>) {
    eprintln!("{} {}", "!".color(palettes::WARN).bold(), msg.as_ref());
}

pub fn info(msg: impl AsRef<str>) {
    eprintln!("{} {}", ARROW.color(palettes::ACCENT), msg.as_ref());
}

pub fn step(msg: impl AsRef<str>) {
    eprintln!("{} {}", BULLET.color(palettes::DIM), msg.as_ref());
}

pub fn dim(msg: impl AsRef<str>) {
    eprintln!("{}", msg.as_ref().color(palettes::DIM));
}

/// Key/value line for detail panels: dim lowercase key, value as given
/// (callers pre-color values with semantic colors where relevant).
pub fn field(label: &str, value: impl AsRef<str>) {
    println!(
        "  {:<12} {}",
        label.to_lowercase().color(palettes::DIM),
        value.as_ref()
    );
}

/// Indented next-action line under a final success message (endpoint, logs,
/// shell hints). Printed to stderr with the rest of the flow chrome.
pub fn next_action(label: &str, value: impl AsRef<str>) {
    eprintln!(
        "  {:<10} {}",
        label.color(palettes::DIM),
        value.as_ref()
    );
}

/// Rounded summary box shown before spending money:
/// ```text
///   ╭─ rental ─────────────────╮
///   │  name      vllm-infer…   │
///   ╰──────────────────────────╯
/// ```
/// Frame and keys are dim; values keep their own colors.
pub fn summary_box(title: &str, rows: &[(&str, String)]) {
    let key_width = rows
        .iter()
        .map(|(k, _)| k.chars().count())
        .max()
        .unwrap_or(0);
    let value_width = rows
        .iter()
        .map(|(_, v)| visible_width(v))
        .max()
        .unwrap_or(0);
    // 2 spaces + key + 2 spaces + value + 2 spaces
    let inner = (2 + key_width + 2 + value_width + 2).max(title.chars().count() + 4);

    let top = format!(
        "╭─ {title} {}╮",
        "─".repeat(inner.saturating_sub(title.chars().count() + 3))
    );
    eprintln!("  {}", top.color(palettes::BORDER));
    for (key, value) in rows {
        let pad = inner
            .saturating_sub(2 + key_width + 2 + visible_width(value));
        eprintln!(
            "  {}  {:<key_width$}  {}{}{}",
            "│".color(palettes::BORDER),
            key.color(palettes::DIM),
            value,
            " ".repeat(pad),
            "│".color(palettes::BORDER),
        );
    }
    eprintln!(
        "  {}",
        format!("╰{}╯", "─".repeat(inner)).color(palettes::BORDER)
    );
}

/// Display width of a string, ignoring ANSI escape sequences.
pub fn visible_width(s: &str) -> usize {
    let mut width = 0;
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // Skip CSI sequence: ESC [ ... final byte in @-~
            if chars.next() == Some('[') {
                for c in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&c) {
                        break;
                    }
                }
            }
        } else {
            width += 1;
        }
    }
    width
}
