use std::io::IsTerminal;

use dialoguer::console::{style, Style};
use dialoguer::theme::ColorfulTheme;
use dialoguer::{Confirm, Input, MultiSelect, Password, Select};

use crate::error::{CliError, Result};

// Nearest xterm-256 approximations of the design palette (the console crate
// used by dialoguer has no truecolor support): 80 ≈ #56D4DD accent,
// 242 ≈ #5C6773 dim, 203 ≈ #F0716B err, 78 ≈ #3FD68F ok.
const ACCENT_256: u8 = 80;
const DIM_256: u8 = 242;
const ERR_256: u8 = 203;
const OK_256: u8 = 78;

pub fn is_tty() -> bool {
    std::io::stdin().is_terminal()
}

pub fn require_tty(field: &str) -> Result<()> {
    if is_tty() {
        Ok(())
    } else {
        Err(CliError::NotInteractive(field.to_string()))
    }
}

fn theme() -> ColorfulTheme {
    let accent = Style::new().for_stderr().color256(ACCENT_256);
    let dim = Style::new().for_stderr().color256(DIM_256);
    ColorfulTheme {
        prompt_prefix: style("?".to_string()).for_stderr().color256(ACCENT_256),
        prompt_suffix: style("›".to_string()).for_stderr().color256(DIM_256),
        success_prefix: style("?".to_string()).for_stderr().color256(ACCENT_256),
        success_suffix: style("›".to_string()).for_stderr().color256(DIM_256),
        error_prefix: style("✗".to_string()).for_stderr().color256(ERR_256),
        error_style: Style::new().for_stderr().color256(ERR_256),
        hint_style: dim.clone(),
        defaults_style: dim,
        values_style: accent.clone(),
        active_item_style: accent,
        active_item_prefix: style("❯".to_string()).for_stderr().color256(OK_256),
        checked_item_prefix: style("✓".to_string()).for_stderr().color256(OK_256),
        ..ColorfulTheme::default()
    }
}

pub fn input(prompt: &str) -> Result<String> {
    Input::with_theme(&theme())
        .with_prompt(prompt)
        .interact_text()
        .map_err(map_err)
}

pub fn input_default(prompt: &str, default: &str) -> Result<String> {
    Input::with_theme(&theme())
        .with_prompt(prompt)
        .default(default.to_string())
        .interact_text()
        .map_err(map_err)
}

pub fn optional(prompt: &str) -> Result<Option<String>> {
    let value: String = Input::with_theme(&theme())
        .with_prompt(prompt)
        .allow_empty(true)
        .interact_text()
        .map_err(map_err)?;
    Ok(Some(value).filter(|v| !v.trim().is_empty()))
}

pub fn password(prompt: &str) -> Result<String> {
    Password::with_theme(&theme())
        .with_prompt(prompt)
        .interact()
        .map_err(map_err)
}

pub fn confirm(prompt: &str, default: bool) -> Result<bool> {
    Confirm::with_theme(&theme())
        .with_prompt(prompt)
        .default(default)
        .interact()
        .map_err(map_err)
}

pub fn select(prompt: &str, items: &[String]) -> Result<usize> {
    Select::with_theme(&theme())
        .with_prompt(prompt)
        .items(items)
        .default(0)
        .interact()
        .map_err(map_err)
}

pub fn multi_select(prompt: &str, items: &[String]) -> Result<Vec<usize>> {
    MultiSelect::with_theme(&theme())
        .with_prompt(prompt)
        .items(items)
        .interact()
        .map_err(map_err)
}

fn map_err(e: dialoguer::Error) -> CliError {
    match e {
        dialoguer::Error::IO(io) if io.kind() == std::io::ErrorKind::Interrupted => {
            CliError::Cancelled
        }
        dialoguer::Error::IO(io) => CliError::Io(io),
    }
}
