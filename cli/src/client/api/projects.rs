use crate::client::error::Result;
use crate::client::http::HttpClient;
use crate::client::pagination::{List, Page};
use crate::client::types::{CreateProjectRequest, Project, UpdateProjectRequest};

#[derive(Debug, Clone)]
pub struct Projects {
    http: HttpClient,
    base_path: String,
}

impl Projects {
    pub(crate) fn new(http: HttpClient, org: impl Into<String>) -> Self {
        Self {
            http,
            base_path: format!("/orgs/{}/projects", org.into()),
        }
    }

    pub async fn list(&self, page: &Page) -> Result<List<Project>> {
        self.http.get_query(&self.base_path, &page.query()).await
    }

    pub async fn get(&self, uid: &str) -> Result<Project> {
        self.http.get(&format!("{}/{uid}", self.base_path)).await
    }

    pub async fn create(&self, req: &CreateProjectRequest) -> Result<Project> {
        self.http.post(&self.base_path, req).await
    }

    pub async fn update(&self, uid: &str, req: &UpdateProjectRequest) -> Result<Project> {
        self.http
            .patch(&format!("{}/{uid}", self.base_path), req)
            .await
    }

    pub async fn delete(&self, uid: &str) -> Result<()> {
        self.http.delete(&format!("{}/{uid}", self.base_path)).await
    }
}
