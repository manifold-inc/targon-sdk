use comfy_table::presets::NOTHING;
use comfy_table::{Attribute, Cell, Color, ContentArrangement, Table};

use crate::output::format::{self, StateKind};

pub fn table(headers: &[&str]) -> Table {
    let mut table = Table::new();
    table
        .load_preset(NOTHING)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(
        headers
            .iter()
            .map(|h| Cell::new(h.to_uppercase()).add_attribute(Attribute::Dim)),
    );
    table
}

pub fn state_cell(status: &str) -> Cell {
    let color = match format::classify(status) {
        StateKind::Ok => Color::Green,
        StateKind::Pending => Color::Yellow,
        StateKind::Bad => Color::Red,
        StateKind::Idle => Color::DarkGrey,
        StateKind::Unknown => Color::White,
    };
    Cell::new(status).fg(color)
}

pub fn dim_cell(value: impl Into<String>) -> Cell {
    Cell::new(value.into()).fg(Color::DarkGrey)
}
