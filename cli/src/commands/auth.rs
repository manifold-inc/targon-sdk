use clap::Subcommand;

use crate::client::DEFAULT_BASE_URL;
use crate::config::{self, default, Config};
use crate::error::{CliError, Result};
use crate::output::{prompt, style};

#[derive(Debug, Subcommand)]
pub enum AuthCommands {
    /// Store an API key for the active profile
    Login,
    /// Remove the stored API key for the active profile
    Logout,
    /// Show authentication status for the active profile
    Status,
    /// Print the resolved API key (for scripting)
    Token,
}

pub async fn handle(
    cmd: &AuthCommands,
    profile_override: Option<&str>,
    base_url_override: Option<&str>,
) -> Result<()> {
    let config = Config::load()?;
    let profile = profile_override
        .map(str::to_string)
        .unwrap_or_else(|| config.current.clone());

    match cmd {
        AuthCommands::Login => login(&profile, base_url_override).await,
        AuthCommands::Logout => logout(&profile),
        AuthCommands::Status => status(&config, &profile, base_url_override),
        AuthCommands::Token => token(&profile),
    }
}

fn token(profile: &str) -> Result<()> {
    let key = std::env::var(default::API_KEY_ENV)
        .ok()
        .filter(|k| !k.is_empty())
        .or_else(|| config::load_api_key(profile))
        .ok_or(CliError::NotAuthenticated)?;
    println!("{key}");
    Ok(())
}

async fn login(profile: &str, base_url_override: Option<&str>) -> Result<()> {
    prompt::require_tty("api-key")?;
    let key = prompt::password("API key")?;
    let key = key.trim();
    if key.is_empty() {
        return Err(CliError::Config("api key cannot be empty".to_string()));
    }
    config::store_api_key(profile, key, base_url_override)?;
    style::success(format!("stored credentials for profile '{profile}'"));
    Ok(())
}

fn logout(profile: &str) -> Result<()> {
    config::clear_api_key(profile)?;
    style::success(format!("cleared credentials for profile '{profile}'"));
    Ok(())
}

fn status(config: &Config, profile: &str, base_url_override: Option<&str>) -> Result<()> {
    let base_url = base_url_override
        .map(str::to_string)
        .or_else(|| config.profile(profile).ok().map(|p| p.base_url.clone()))
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());

    style::field("Profile", profile);
    style::field("Base URL", &base_url);

    if let Ok(key) = std::env::var(default::API_KEY_ENV) {
        if !key.is_empty() {
            style::field("API key", format!("{} (via {})", mask(&key), default::API_KEY_ENV));
            return Ok(());
        }
    }

    match config::load_api_key(profile) {
        Some(key) => style::field("API key", mask(&key)),
        None => style::field("API key", "not set"),
    }
    Ok(())
}

fn mask(key: &str) -> String {
    let count = key.chars().count();
    if count <= 4 {
        return "•".repeat(count.max(1));
    }
    let tail: String = key.chars().skip(count - 4).collect();
    format!("{}{tail}", "•".repeat(4))
}
