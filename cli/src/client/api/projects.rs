use crate::client::error::Result;
use crate::client::http::HttpClient;
use crate::client::pagination::{List, Page};
use crate::client::types::{CreateProjectRequest, Project, UpdateProjectRequest};

#[derive(Debug, Clone)]
pub struct Projects {
    http: HttpClient,
}

impl Projects {
    pub(crate) fn new(http: HttpClient) -> Self {
        Self { http }
    }

    pub async fn list(&self, page: &Page) -> Result<List<Project>> {
        self.http.get_query("/projects", &page.query()).await
    }

    pub async fn get(&self, uid: &str) -> Result<Project> {
        self.http.get(&format!("/projects/{uid}")).await
    }

    pub async fn create(&self, req: &CreateProjectRequest) -> Result<Project> {
        self.http.post("/projects", req).await
    }

    pub async fn update(&self, uid: &str, req: &UpdateProjectRequest) -> Result<Project> {
        self.http.patch(&format!("/projects/{uid}"), req).await
    }

    pub async fn delete(&self, uid: &str) -> Result<()> {
        self.http.delete(&format!("/projects/{uid}")).await
    }
}
