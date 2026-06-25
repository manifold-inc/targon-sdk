pub mod api;
pub mod config;
pub mod error;
pub mod http;
pub mod pagination;
pub mod types;

pub use config::{ClientConfig, DEFAULT_BASE_URL};
pub use error::{ClientError, Result};
pub use pagination::{List, Page};

use std::time::Duration;

use api::{InventoryApi, Projects, SshKeys, User, VersionApi, Volumes, Workloads};
use http::HttpClient;

#[derive(Debug, Clone)]
pub struct Client {
    http: HttpClient,
}

impl Client {
    pub fn new(config: ClientConfig) -> Result<Self> {
        Ok(Self {
            http: HttpClient::new(config)?,
        })
    }

    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    pub fn workloads(&self) -> Workloads {
        Workloads::new(self.http.clone())
    }

    pub fn volumes(&self) -> Volumes {
        Volumes::new(self.http.clone())
    }

    pub fn ssh_keys(&self) -> SshKeys {
        SshKeys::new(self.http.clone())
    }

    pub fn projects(&self) -> Projects {
        Projects::new(self.http.clone())
    }

    pub fn inventory(&self) -> InventoryApi {
        InventoryApi::new(self.http.clone())
    }

    pub fn user(&self) -> User {
        User::new(self.http.clone())
    }

    pub fn version(&self) -> VersionApi {
        VersionApi::new(self.http.clone())
    }
}

#[derive(Debug, Default)]
pub struct ClientBuilder {
    base_url: Option<String>,
    api_key: Option<String>,
    timeout: Option<Duration>,
}

impl ClientBuilder {
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn build(self) -> Result<Client> {
        let api_key = self
            .api_key
            .ok_or_else(|| ClientError::InvalidConfig("api key is required".to_string()))?;

        let mut config = ClientConfig::new(api_key);
        if let Some(base_url) = self.base_url {
            config.base_url = base_url;
        }
        if let Some(timeout) = self.timeout {
            config.timeout = timeout;
        }

        Client::new(config)
    }
}
