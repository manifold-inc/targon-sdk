use colored::Color;

/// Truecolor palette from the Targon CLI output design spec.
pub const SUCCESS: Color = rgb(0x3F, 0xD6, 0x8F);
pub const ERROR: Color = rgb(0xF0, 0x71, 0x6B);
pub const WARN: Color = rgb(0xE5, 0xB4, 0x54);
pub const ACCENT: Color = rgb(0x56, 0xD4, 0xDD);
pub const HEADER: Color = rgb(0xF2, 0xF5, 0xF8);
pub const DIM: Color = rgb(0x5C, 0x67, 0x73);
pub const BORDER: Color = rgb(0x33, 0x3C, 0x4A);
pub const RENTAL_BLUE: Color = rgb(0x6C, 0xA4, 0xF8);
pub const VM_MAGENTA: Color = rgb(0xB8, 0x8E, 0xF5);
pub const SRVLESS_AMBER: Color = rgb(0xE5, 0xB4, 0x54);
pub const STORAGE_CYAN: Color = rgb(0x56, 0xD4, 0xDD);

const fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::TrueColor { r, g, b }
}

/// Offering / workload types get their own hues so a mixed list scans instantly.
pub fn workload_type_color(workload_type: &str) -> Color {
    match workload_type.to_ascii_uppercase().as_str() {
        "VM" => VM_MAGENTA,
        "RENTAL" => RENTAL_BLUE,
        "SRVLESS" | "SERVERLESS" => SRVLESS_AMBER,
        "STORAGE" | "VOLUME" => STORAGE_CYAN,
        _ => DIM,
    }
}

/// Whether colored output is currently enabled (respects NO_COLOR and TTY
/// detection via the `colored` crate). Used to gate comfy-table cell colors,
/// which would otherwise emit ANSI codes unconditionally.
pub fn colors_enabled() -> bool {
    colored::control::SHOULD_COLORIZE.should_colorize()
}
