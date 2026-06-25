use std::path::PathBuf;

pub const DEFAULT_PROFILE: &str = "default";
pub const CONFIG_FILE: &str = "config.toml";
pub const API_KEY_ENV: &str = "TARGON_API_KEY";

pub fn config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".targon")
}

pub fn config_file() -> PathBuf {
    config_dir().join(CONFIG_FILE)
}

pub fn credentials_file(profile: &str) -> PathBuf {
    config_dir().join(format!("credentials-{profile}"))
}
