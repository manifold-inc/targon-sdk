use std::io::IsTerminal;

use colored::Colorize;
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Attribute, Cell, Color, ContentArrangement, Table, TableComponent};

use crate::output::format::{self, StateKind};
use crate::output::{palettes, style};

/// Boxed list table with rounded unicode borders. On a TTY the frame is
/// rendered dim so the data pops, not the frame; piped output degrades to
/// tab-separated columns with no borders or color (see [`print`]).
pub fn table(headers: &[&str]) -> Table {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic);
    // Clean single-line frame per the design: solid │ verticals, one ─ rule
    // under the header, and no separators between data rows.
    table
        .set_style(TableComponent::VerticalLines, '│')
        .set_style(TableComponent::HeaderLines, '─')
        .set_style(TableComponent::LeftHeaderIntersection, '├')
        .set_style(TableComponent::MiddleHeaderIntersections, '┼')
        .set_style(TableComponent::RightHeaderIntersection, '┤')
        .remove_style(TableComponent::HorizontalLines)
        .remove_style(TableComponent::MiddleIntersections)
        .remove_style(TableComponent::LeftBorderIntersections)
        .remove_style(TableComponent::RightBorderIntersections);
    table.set_header(headers.iter().map(|h| {
        let cell = Cell::new(h.to_uppercase());
        if palettes::colors_enabled() {
            cell.fg(rgb(palettes::HEADER)).add_attribute(Attribute::Bold)
        } else {
            cell
        }
    }));
    table
}

/// Print a table: boxed with a dim frame on a TTY, tab-separated plain
/// columns when piped.
pub fn print(t: &Table) {
    if std::io::stdout().is_terminal() {
        for line in t.lines() {
            println!("{}", dim_frame(&line));
        }
    } else {
        if let Some(header) = t.header() {
            let cells: Vec<String> = header.cell_iter().map(|c| c.content()).collect();
            println!("{}", cells.join("\t"));
        }
        for row in t.row_iter() {
            let cells: Vec<String> = row.cell_iter().map(|c| c.content()).collect();
            println!("{}", cells.join("\t"));
        }
    }
}

/// Dim summary line under a list table, e.g.
/// `  3 workloads · 2 running · $26.91/hr burning`. Skipped when piped.
pub fn summary(msg: impl AsRef<str>) {
    if std::io::stdout().is_terminal() {
        println!("  {}", msg.as_ref().color(palettes::DIM));
    }
}

/// UIDs are the thing you copy — render them in accent cyan.
pub fn uid_cell(uid: impl Into<String>) -> Cell {
    colored_cell(uid.into(), palettes::ACCENT)
}

/// Workload type in its own hue (RENTAL blue, VM magenta).
pub fn type_cell(workload_type: &str) -> Cell {
    colored_cell(
        workload_type.to_string(),
        palettes::workload_type_color(workload_type),
    )
}

/// State as a colored dot + word, e.g. `● running`. Piped output drops the
/// dot so columns stay clean for cut/awk.
pub fn state_cell(status: &str) -> Cell {
    if status.is_empty() {
        return Cell::new("-");
    }
    let status = status.to_lowercase();
    if !std::io::stdout().is_terminal() {
        return Cell::new(status);
    }
    let color = match format::classify(&status) {
        StateKind::Ok => palettes::SUCCESS,
        StateKind::Pending | StateKind::Idle => palettes::WARN,
        StateKind::Bad => palettes::ERROR,
        StateKind::Unknown => palettes::DIM,
    };
    colored_cell(format!("{} {status}", style::DOT), color)
}

pub fn dim_cell(value: impl Into<String>) -> Cell {
    colored_cell(value.into(), palettes::DIM)
}

fn colored_cell(content: String, color: colored::Color) -> Cell {
    let cell = Cell::new(content);
    if palettes::colors_enabled() {
        cell.fg(rgb(color))
    } else {
        cell
    }
}

fn rgb(color: colored::Color) -> Color {
    match color {
        colored::Color::TrueColor { r, g, b } => Color::Rgb { r, g, b },
        _ => Color::White,
    }
}

/// Wrap runs of box-drawing characters in the dim border color so the frame
/// recedes behind the data. comfy-table cannot color the frame itself.
fn dim_frame(line: &str) -> String {
    if !palettes::colors_enabled() {
        return line.to_string();
    }
    let mut out = String::with_capacity(line.len() * 2);
    let mut run = String::new();
    for c in line.chars() {
        if is_frame_char(c) {
            run.push(c);
        } else {
            flush_frame(&mut out, &mut run);
            out.push(c);
        }
    }
    flush_frame(&mut out, &mut run);
    out
}

fn flush_frame(out: &mut String, run: &mut String) {
    if !run.is_empty() {
        out.push_str(&run.color(palettes::BORDER).to_string());
        run.clear();
    }
}

fn is_frame_char(c: char) -> bool {
    matches!(
        c,
        '─' | '│' | '╭' | '╮' | '╰' | '╯' | '├' | '┤' | '┬' | '┴' | '┼'
    )
}
