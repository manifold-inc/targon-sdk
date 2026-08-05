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
    pub org: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub projects: BTreeMap<String, String>,
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
                org: None,
                projects: BTreeMap::new(),
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
            Ok(contents) => toml::from_str(&contents).map_err(|e| {
                CliError::Config(format!("invalid config at {}: {e}", path.display()))
            }),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(CliError::Io(e)),
        }
    }

    pub fn save(&self) -> Result<()> {
        fs::create_dir_all(default::config_dir())?;
        let contents = toml::to_string_pretty(self).map_err(|e| CliError::Config(e.to_string()))?;
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
    pub org: Option<String>,
    pub project: Option<String>,
    pub api_key: Option<String>,
}

pub fn resolve(
    profile_override: Option<&str>,
    base_url_override: Option<&str>,
    org_override: Option<&str>,
) -> Result<Resolved> {
    let config = Config::load()?;
    let profile_name = profile_override
        .map(str::to_string)
        .unwrap_or_else(|| config.current.clone());

    // Credentials may exist without a config entry (e.g. login before this was fixed).
    if !config.profiles.contains_key(&profile_name) && load_api_key(&profile_name).is_some() {
        ensure_profile(&profile_name, base_url_override)?;
    }

    let config = Config::load()?;
    let profile = config.profile(&profile_name)?;

    let base_url = base_url_override
        .map(str::to_string)
        .unwrap_or_else(|| profile.base_url.clone());

    let api_key = env_api_key().or_else(|| load_api_key(&profile_name));
    let org = resolve_org(profile, org_override, env_org().as_deref());
    let project = project_for_org(profile, org.as_deref());

    Ok(Resolved {
        profile: profile_name,
        base_url,
        org,
        project,
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
                    org: None,
                    projects: BTreeMap::new(),
                },
            );
            config.save()?;
        }
    }
    Ok(())
}

/// Persist the active organization for a profile.
pub fn set_org(profile_override: Option<&str>, org: Option<String>) -> Result<String> {
    let mut config = Config::load()?;
    let profile_name = selected_profile_name(&config, profile_override);
    config.profile(&profile_name)?;
    if let Some(profile) = config.profiles.get_mut(&profile_name) {
        profile.org = org;
    }
    config.save()?;
    Ok(profile_name)
}

/// Persist (or clear) the default project for one organization in a profile.
pub fn set_project(
    profile_override: Option<&str>,
    org: &str,
    project: Option<String>,
) -> Result<String> {
    let mut config = Config::load()?;
    let profile_name = selected_profile_name(&config, profile_override);
    config.profile(&profile_name)?;
    if let Some(profile) = config.profiles.get_mut(&profile_name) {
        match project {
            Some(project) => {
                profile.projects.insert(org.to_string(), project);
            }
            None => {
                profile.projects.remove(org);
            }
        }
    }
    config.save()?;
    Ok(profile_name)
}

/// Move locally persisted context when an organization slug is renamed.
pub fn rename_org_slug(
    profile_override: Option<&str>,
    old_slug: &str,
    new_slug: &str,
) -> Result<String> {
    let mut config = Config::load()?;
    let profile_name = selected_profile_name(&config, profile_override);
    config.profile(&profile_name)?;
    if let Some(profile) = config.profiles.get_mut(&profile_name) {
        rename_org_context(profile, old_slug, new_slug);
    }
    config.save()?;
    Ok(profile_name)
}

/// Remove an organization's remembered project and clear it if it is active.
pub fn clear_org_context(profile_override: Option<&str>, org: &str) -> Result<String> {
    let mut config = Config::load()?;
    let profile_name = selected_profile_name(&config, profile_override);
    config.profile(&profile_name)?;
    if let Some(profile) = config.profiles.get_mut(&profile_name) {
        clear_org(profile, org);
    }
    config.save()?;
    Ok(profile_name)
}

fn selected_profile_name(config: &Config, profile_override: Option<&str>) -> String {
    profile_override
        .map(str::to_string)
        .unwrap_or_else(|| config.current.clone())
}

fn resolve_org(
    profile: &Profile,
    org_override: Option<&str>,
    org_from_env: Option<&str>,
) -> Option<String> {
    org_override
        .filter(|org| !org.is_empty())
        .or_else(|| org_from_env.filter(|org| !org.is_empty()))
        .map(str::to_string)
        .or_else(|| profile.org.clone().filter(|org| !org.is_empty()))
}

fn project_for_org(profile: &Profile, org: Option<&str>) -> Option<String> {
    org.and_then(|org_slug| profile.projects.get(org_slug))
        .cloned()
}

fn rename_org_context(profile: &mut Profile, old_slug: &str, new_slug: &str) {
    if old_slug == new_slug {
        return;
    }
    if profile.org.as_deref() == Some(old_slug) {
        profile.org = Some(new_slug.to_string());
    }
    if let Some(project) = profile.projects.remove(old_slug) {
        profile.projects.insert(new_slug.to_string(), project);
    }
}

fn clear_org(profile: &mut Profile, org: &str) {
    if profile.org.as_deref() == Some(org) {
        profile.org = None;
    }
    profile.projects.remove(org);
}

fn env_api_key() -> Option<String> {
    std::env::var(default::API_KEY_ENV)
        .ok()
        .filter(|k| !k.is_empty())
}

fn env_org() -> Option<String> {
    std::env::var(default::ORG_ENV)
        .ok()
        .filter(|org| !org.is_empty())
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
