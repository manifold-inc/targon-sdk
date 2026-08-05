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

use api::{
    ApiTokens, CreditsApi, InventoryApi, Members, Orgs, Projects, ServiceTokens, SshKeys,
    VersionApi, Volumes, WalletApi, Workloads,
};
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

    pub fn workloads(&self, org: impl Into<String>) -> Workloads {
        Workloads::new(self.http.clone(), org)
    }

    pub fn volumes(&self, org: impl Into<String>) -> Volumes {
        Volumes::new(self.http.clone(), org)
    }

    pub fn ssh_keys(&self, org: impl Into<String>) -> SshKeys {
        SshKeys::new(self.http.clone(), org)
    }

    pub fn projects(&self, org: impl Into<String>) -> Projects {
        Projects::new(self.http.clone(), org)
    }

    pub fn orgs(&self) -> Orgs {
        Orgs::new(self.http.clone())
    }

    pub fn members(&self, org: impl Into<String>) -> Members {
        Members::new(self.http.clone(), org)
    }

    pub fn service_tokens(&self, org: impl Into<String>) -> ServiceTokens {
        ServiceTokens::new(self.http.clone(), org)
    }

    pub fn api_tokens(&self) -> ApiTokens {
        ApiTokens::new(self.http.clone())
    }

    pub fn wallet(&self, org: impl Into<String>) -> WalletApi {
        WalletApi::new(self.http.clone(), org)
    }

    pub fn credits(&self, org: impl Into<String>) -> CreditsApi {
        CreditsApi::new(self.http.clone(), org)
    }

    pub fn inventory(&self) -> InventoryApi {
        InventoryApi::new(self.http.clone())
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
