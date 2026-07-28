use crate::client::error::Result;
use crate::client::http::HttpClient;
use crate::client::pagination::{List, Page};
use crate::client::types::{CreateSshKeyRequest, SshKey, UpdateSshKeyRequest};

#[derive(Debug, Clone)]
pub struct SshKeys {
    http: HttpClient,
    base_path: String,
}

impl SshKeys {
    pub(crate) fn new(http: HttpClient, org: impl Into<String>) -> Self {
        Self {
            http,
            base_path: format!("/orgs/{}/ssh-keys", org.into()),
        }
    }

    pub async fn list(&self, page: &Page) -> Result<List<SshKey>> {
        self.http.get_query(&self.base_path, &page.query()).await
    }

    pub async fn get(&self, uid: &str) -> Result<SshKey> {
        self.http.get(&format!("{}/{uid}", self.base_path)).await
    }

    pub async fn create(&self, req: &CreateSshKeyRequest) -> Result<SshKey> {
        self.http.post(&self.base_path, req).await
    }

    pub async fn update(&self, uid: &str, req: &UpdateSshKeyRequest) -> Result<SshKey> {
        self.http
            .patch(&format!("{}/{uid}", self.base_path), req)
            .await
    }

    pub async fn delete(&self, uid: &str) -> Result<()> {
        self.http.delete(&format!("{}/{uid}", self.base_path)).await
    }
}
