use std::time::Duration;

pub const DEFAULT_BASE_URL: &str = "https://api.targon.com";
pub const API_VERSION: &str = "/tha/v2";
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub base_url: String,
    pub api_key: String,
    pub timeout: Duration,
}

impl ClientConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            timeout: DEFAULT_TIMEOUT,
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{}{}", self.base_url.trim_end_matches('/'), API_VERSION, path)
    }
}
