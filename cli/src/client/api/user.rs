use crate::client::error::Result;
use crate::client::http::HttpClient;
use crate::client::pagination::{List, Page};
use crate::client::types::{ApiKey, CreateApiKeyRequest, Credits, UpdateApiKeyRequest, Wallet};

#[derive(Debug, Clone)]
pub struct User {
    http: HttpClient,
}

impl User {
    pub(crate) fn new(http: HttpClient) -> Self {
        Self { http }
    }

    pub async fn wallet(&self) -> Result<Wallet> {
        self.http.get("/me/wallet").await
    }

    pub async fn credits(&self) -> Result<Credits> {
        self.http.get("/me/credits").await
    }

    pub async fn api_keys(&self, page: &Page) -> Result<List<ApiKey>> {
        self.http.get_query("/me/api-keys", &page.query()).await
    }

    pub async fn create_api_key(&self, req: &CreateApiKeyRequest) -> Result<ApiKey> {
        self.http.post("/me/api-keys", req).await
    }

    pub async fn update_api_key(&self, uid: &str, req: &UpdateApiKeyRequest) -> Result<ApiKey> {
        self.http.patch(&format!("/me/api-keys/{uid}"), req).await
    }

    pub async fn delete_api_key(&self, uid: &str) -> Result<()> {
        self.http.delete(&format!("/me/api-keys/{uid}")).await
    }

    pub async fn rotate_api_key(&self, uid: &str) -> Result<ApiKey> {
        self.http.post_empty(&format!("/me/api-keys/{uid}:roll")).await
    }
}
