use crate::client::error::Result;
use crate::client::http::HttpClient;
use crate::client::pagination::{List, Page};
use crate::client::types::{
    ApiToken, CreateApiTokenRequest, CreateServiceTokenRequest, ServiceToken, UpdateApiTokenRequest,
};

#[derive(Debug, Clone)]
pub struct ApiTokens {
    http: HttpClient,
}

impl ApiTokens {
    pub(crate) fn new(http: HttpClient) -> Self {
        Self { http }
    }

    pub async fn list(&self, page: &Page) -> Result<List<ApiToken>> {
        self.http.get_query("/me/api-tokens", &page.query()).await
    }

    pub async fn create(&self, req: &CreateApiTokenRequest) -> Result<ApiToken> {
        self.http.post("/me/api-tokens", req).await
    }

    pub async fn update(&self, uid: &str, req: &UpdateApiTokenRequest) -> Result<ApiToken> {
        self.http.patch(&format!("/me/api-tokens/{uid}"), req).await
    }

    pub async fn delete(&self, uid: &str) -> Result<()> {
        self.http.delete(&format!("/me/api-tokens/{uid}")).await
    }
}

#[derive(Debug, Clone)]
pub struct ServiceTokens {
    http: HttpClient,
    base_path: String,
}

impl ServiceTokens {
    pub(crate) fn new(http: HttpClient, org: impl Into<String>) -> Self {
        Self {
            http,
            base_path: format!("/orgs/{}/tokens", org.into()),
        }
    }

    pub async fn list(&self, page: &Page) -> Result<List<ServiceToken>> {
        self.http.get_query(&self.base_path, &page.query()).await
    }

    pub async fn create(&self, req: &CreateServiceTokenRequest) -> Result<ServiceToken> {
        self.http.post(&self.base_path, req).await
    }

    pub async fn delete(&self, uid: &str) -> Result<()> {
        self.http.delete(&format!("{}/{uid}", self.base_path)).await
    }
}
