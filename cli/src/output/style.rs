use colored::Colorize;

use crate::output::palettes;

pub const TICK: &str = "✓";
pub const CROSS: &str = "✗";
pub const ARROW: &str = "→";
pub const BULLET: &str = "•";

pub fn success(msg: impl AsRef<str>) {
    eprintln!("{} {}", TICK.color(palettes::SUCCESS).bold(), msg.as_ref());
}

pub fn error(msg: impl AsRef<str>) {
    eprintln!("{} {}", CROSS.color(palettes::ERROR).bold(), msg.as_ref());
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

pub fn field(label: &str, value: impl AsRef<str>) {
    println!("{:<16} {}", label.color(palettes::DIM), value.as_ref());
}
