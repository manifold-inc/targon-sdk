use std::io::IsTerminal;

use dialoguer::theme::ColorfulTheme;
use dialoguer::{Confirm, Input, Password, Select};

use crate::error::{CliError, Result};

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
    ColorfulTheme::default()
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

fn map_err(e: dialoguer::Error) -> CliError {
    match e {
        dialoguer::Error::IO(io) if io.kind() == std::io::ErrorKind::Interrupted => {
            CliError::Cancelled
        }
        dialoguer::Error::IO(io) => CliError::Io(io),
    }
}
