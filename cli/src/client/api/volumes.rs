use crate::client::error::Result;
use crate::client::http::HttpClient;
use crate::client::pagination::{List, Page};
use crate::client::types::{
    CreateVolumeRequest, ListVolumesParams, UpdateVolumeRequest, Volume, VolumeEvent,
    VolumeOperationResponse, VolumeStateResponse,
};

#[derive(Debug, Clone)]
pub struct Volumes {
    http: HttpClient,
    base_path: String,
}

impl Volumes {
    pub(crate) fn new(http: HttpClient, org: impl Into<String>) -> Self {
        Self {
            http,
            base_path: format!("/orgs/{}/volumes", org.into()),
        }
    }

    pub async fn list(&self, params: &ListVolumesParams) -> Result<List<Volume>> {
        let mut query = params.page.query();
        if let Some(workload_uid) = &params.workload_uid {
            query.push(("workload_uid", workload_uid.clone()));
        }
        self.http.get_query(&self.base_path, &query).await
    }

    pub async fn get(&self, uid: &str) -> Result<Volume> {
        self.http.get(&format!("{}/{uid}", self.base_path)).await
    }

    pub async fn create(&self, req: &CreateVolumeRequest) -> Result<VolumeOperationResponse> {
        self.http.post(&self.base_path, req).await
    }

    pub async fn update(&self, uid: &str, req: &UpdateVolumeRequest) -> Result<Volume> {
        self.http
            .patch(&format!("{}/{uid}", self.base_path), req)
            .await
    }

    pub async fn delete(&self, uid: &str) -> Result<()> {
        self.http.delete(&format!("{}/{uid}", self.base_path)).await
    }

    pub async fn state(&self, uid: &str) -> Result<VolumeStateResponse> {
        self.http
            .get(&format!("{}/{uid}/state", self.base_path))
            .await
    }

    pub async fn events(&self, uid: &str, page: &Page) -> Result<List<VolumeEvent>> {
        self.http
            .get_query(&format!("{}/{uid}/events", self.base_path), &page.query())
            .await
    }
}
