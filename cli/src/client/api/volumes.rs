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
}

impl Volumes {
    pub(crate) fn new(http: HttpClient) -> Self {
        Self { http }
    }

    pub async fn list(&self, params: &ListVolumesParams) -> Result<List<Volume>> {
        let mut query = params.page.query();
        if let Some(workload_uid) = &params.workload_uid {
            query.push(("workload_uid", workload_uid.clone()));
        }
        self.http.get_query("/volumes", &query).await
    }

    pub async fn get(&self, uid: &str) -> Result<Volume> {
        self.http.get(&format!("/volumes/{uid}")).await
    }

    pub async fn create(&self, req: &CreateVolumeRequest) -> Result<VolumeOperationResponse> {
        self.http.post("/volumes", req).await
    }

    pub async fn update(&self, uid: &str, req: &UpdateVolumeRequest) -> Result<Volume> {
        self.http.patch(&format!("/volumes/{uid}"), req).await
    }

    pub async fn delete(&self, uid: &str) -> Result<()> {
        self.http.delete(&format!("/volumes/{uid}")).await
    }

    pub async fn delete_deployment(&self, uid: &str) -> Result<VolumeOperationResponse> {
        self.http.post_empty(&format!("/volumes/{uid}/delete")).await
    }

    pub async fn state(&self, uid: &str) -> Result<VolumeStateResponse> {
        self.http.get(&format!("/volumes/{uid}/state")).await
    }

    pub async fn events(&self, uid: &str, page: &Page) -> Result<List<VolumeEvent>> {
        self.http
            .get_query(&format!("/volumes/{uid}/events"), &page.query())
            .await
    }
}
