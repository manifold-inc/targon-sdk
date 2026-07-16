use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use serde::{Deserialize, Serialize};

use crate::client::DEFAULT_BASE_URL;
use crate::config::default;
use crate::error::{CliError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub base_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub current: String,
    pub profiles: BTreeMap<String, Profile>,
}

impl Default for Config {
    fn default() -> Self {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            default::DEFAULT_PROFILE.to_string(),
            Profile {
                base_url: DEFAULT_BASE_URL.to_string(),
                project: None,
            },
        );
        Self {
            current: default::DEFAULT_PROFILE.to_string(),
            profiles,
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let path = default::config_file();
        match fs::read_to_string(&path) {
            Ok(contents) => toml::from_str(&contents)
                .map_err(|e| CliError::Config(format!("invalid config at {}: {e}", path.display()))),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(CliError::Io(e)),
        }
    }

    pub fn save(&self) -> Result<()> {
        fs::create_dir_all(default::config_dir())?;
        let contents =
            toml::to_string_pretty(self).map_err(|e| CliError::Config(e.to_string()))?;
        fs::write(default::config_file(), contents)?;
        Ok(())
    }

    pub fn profile(&self, name: &str) -> Result<&Profile> {
        self.profiles
            .get(name)
            .ok_or_else(|| CliError::Config(format!("unknown profile '{name}'")))
    }
}

pub struct Resolved {
    pub profile: String,
    pub base_url: String,
    pub project: Option<String>,
    pub api_key: Option<String>,
}

pub fn resolve(profile_override: Option<&str>, base_url_override: Option<&str>) -> Result<Resolved> {
    let config = Config::load()?;
    let profile_name = profile_override
        .map(str::to_string)
        .unwrap_or_else(|| config.current.clone());

    // Credentials may exist without a config entry (e.g. login before this was fixed).
    if config.profiles.get(&profile_name).is_none() && load_api_key(&profile_name).is_some() {
        ensure_profile(&profile_name, base_url_override)?;
    }

    let config = Config::load()?;
    let profile = config.profile(&profile_name)?;

    let base_url = base_url_override
        .map(str::to_string)
        .unwrap_or_else(|| profile.base_url.clone());

    let api_key = env_api_key().or_else(|| load_api_key(&profile_name));

    Ok(Resolved {
        profile: profile_name,
        base_url,
        project: profile.project.clone(),
        api_key,
    })
}

/// Ensure `name` exists in config.toml, creating it with the given (or default) base URL.
pub fn ensure_profile(name: &str, base_url: Option<&str>) -> Result<()> {
    let mut config = Config::load()?;
    match config.profiles.get_mut(name) {
        Some(profile) => {
            if let Some(url) = base_url {
                profile.base_url = url.to_string();
                config.save()?;
            }
        }
        None => {
            config.profiles.insert(
                name.to_string(),
                Profile {
                    base_url: base_url.unwrap_or(DEFAULT_BASE_URL).to_string(),
                    project: None,
                },
            );
            config.save()?;
        }
    }
    Ok(())
}

pub fn set_project(profile_override: Option<&str>, project: Option<String>) -> Result<String> {
    let mut config = Config::load()?;
    let profile_name = profile_override
        .map(str::to_string)
        .unwrap_or_else(|| config.current.clone());
    config.profile(&profile_name)?;
    if let Some(profile) = config.profiles.get_mut(&profile_name) {
        profile.project = project;
    }
    config.save()?;
    Ok(profile_name)
}

fn env_api_key() -> Option<String> {
    std::env::var(default::API_KEY_ENV)
        .ok()
        .filter(|k| !k.is_empty())
}

pub fn load_api_key(profile: &str) -> Option<String> {
    fs::read_to_string(default::credentials_file(profile))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub fn store_api_key(profile: &str, api_key: &str, base_url: Option<&str>) -> Result<()> {
    fs::create_dir_all(default::config_dir())?;
    let path = default::credentials_file(profile);
    fs::write(&path, api_key)?;
    #[cfg(unix)]
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    ensure_profile(profile, base_url)?;
    Ok(())
}

pub fn clear_api_key(profile: &str) -> Result<()> {
    let path = default::credentials_file(profile);
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}
