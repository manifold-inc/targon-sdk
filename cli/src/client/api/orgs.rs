use crate::client::error::Result;
use crate::client::http::HttpClient;
use crate::client::pagination::{List, Page};
use crate::client::types::{
    CreateOrgRequest, Credits, ListMembersParams, Membership, Org, UpdateMemberRequest,
    UpdateOrgRequest, Wallet,
};

#[derive(Debug, Clone)]
pub struct Orgs {
    http: HttpClient,
}

impl Orgs {
    pub(crate) fn new(http: HttpClient) -> Self {
        Self { http }
    }

    pub async fn list(&self, page: &Page) -> Result<List<Org>> {
        self.http.get_query("/orgs", &page.query()).await
    }

    pub async fn get(&self, slug: &str) -> Result<Org> {
        self.http.get(&format!("/orgs/{slug}")).await
    }

    pub async fn create(&self, req: &CreateOrgRequest) -> Result<Org> {
        self.http.post("/orgs", req).await
    }

    pub async fn update(&self, slug: &str, req: &UpdateOrgRequest) -> Result<Org> {
        self.http.patch(&format!("/orgs/{slug}"), req).await
    }

    pub async fn delete(&self, slug: &str) -> Result<()> {
        self.http.delete(&format!("/orgs/{slug}")).await
    }
}

#[derive(Debug, Clone)]
pub struct Members {
    http: HttpClient,
    base_path: String,
}

impl Members {
    pub(crate) fn new(http: HttpClient, org: impl Into<String>) -> Self {
        Self {
            http,
            base_path: format!("/orgs/{}/members", org.into()),
        }
    }

    pub async fn list(&self, params: &ListMembersParams) -> Result<List<Membership>> {
        let mut query = params.page.query();
        if let Some(role) = params.role {
            query.push(("role", role.as_str().to_string()));
        }
        if let Some(status) = params.status {
            query.push(("status", status.as_str().to_string()));
        }
        self.http.get_query(&self.base_path, &query).await
    }

    pub async fn get(&self, username: &str) -> Result<Membership> {
        self.http
            .get(&format!("{}/{username}", self.base_path))
            .await
    }

    pub async fn update(&self, username: &str, req: &UpdateMemberRequest) -> Result<Membership> {
        self.http
            .patch(&format!("{}/{username}", self.base_path), req)
            .await
    }

    pub async fn delete(&self, username: &str) -> Result<()> {
        self.http
            .delete(&format!("{}/{username}", self.base_path))
            .await
    }
}

#[derive(Debug, Clone)]
pub struct WalletApi {
    http: HttpClient,
    path: String,
}

impl WalletApi {
    pub(crate) fn new(http: HttpClient, org: impl Into<String>) -> Self {
        Self {
            http,
            path: format!("/orgs/{}/wallet", org.into()),
        }
    }

    pub async fn get(&self) -> Result<Wallet> {
        self.http.get(&self.path).await
    }
}

#[derive(Debug, Clone)]
pub struct CreditsApi {
    http: HttpClient,
    path: String,
}

impl CreditsApi {
    pub(crate) fn new(http: HttpClient, org: impl Into<String>) -> Self {
        Self {
            http,
            path: format!("/orgs/{}/credits", org.into()),
        }
    }

    pub async fn get(&self) -> Result<Credits> {
        self.http.get(&self.path).await
    }
}
