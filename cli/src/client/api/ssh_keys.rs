use crate::client::error::Result;
use crate::client::http::HttpClient;
use crate::client::pagination::{List, Page};
use crate::client::types::{CreateSshKeyRequest, SshKey, UpdateSshKeyRequest};

#[derive(Debug, Clone)]
pub struct SshKeys {
    http: HttpClient,
}

impl SshKeys {
    pub(crate) fn new(http: HttpClient) -> Self {
        Self { http }
    }

    pub async fn list(&self, page: &Page) -> Result<List<SshKey>> {
        self.http.get_query("/ssh-keys", &page.query()).await
    }

    pub async fn get(&self, uid: &str) -> Result<SshKey> {
        self.http.get(&format!("/ssh-keys/{uid}")).await
    }

    pub async fn create(&self, req: &CreateSshKeyRequest) -> Result<SshKey> {
        self.http.post("/ssh-keys", req).await
    }

    pub async fn update(&self, uid: &str, req: &UpdateSshKeyRequest) -> Result<SshKey> {
        self.http.patch(&format!("/ssh-keys/{uid}"), req).await
    }

    pub async fn delete(&self, uid: &str) -> Result<()> {
        self.http.delete(&format!("/ssh-keys/{uid}")).await
    }
}
