use bytes::Bytes;
use futures_util::Stream;

use crate::client::error::Result;
use crate::client::http::HttpClient;
use crate::client::pagination::{List, Page};
use crate::client::types::{
    AttachVolumeRequest, CreateWorkloadRequest, ListWorkloadsParams, LogOptions,
    UpdateWorkloadRequest, VerifyWorkloadRequest, VerifyWorkloadResponse, VmImage, Workload,
    WorkloadEvent, WorkloadOperationResponse, WorkloadSshKeyAttachment, WorkloadStateResponse,
    WorkloadVolume,
};

#[derive(Debug, Clone)]
pub struct Workloads {
    http: HttpClient,
    base_path: String,
}

impl Workloads {
    pub(crate) fn new(http: HttpClient, org: impl Into<String>) -> Self {
        Self {
            http,
            base_path: format!("/orgs/{}/workloads", org.into()),
        }
    }

    pub async fn list(
        &self,
        params: &ListWorkloadsParams,
    ) -> Result<List<WorkloadOperationResponse>> {
        let mut query = params.page.query();
        if let Some(workload_type) = &params.workload_type {
            query.push(("type", workload_type.clone()));
        }
        if let Some(status) = &params.status {
            query.push(("status", status.clone()));
        }
        if let Some(project_id) = &params.project_id {
            query.push(("project_id", project_id.clone()));
        }
        if let Some(name) = &params.name {
            query.push(("name", name.clone()));
        }
        self.http.get_query(&self.base_path, &query).await
    }

    pub async fn get(&self, uid: &str) -> Result<Workload> {
        self.http.get(&format!("{}/{uid}", self.base_path)).await
    }

    pub async fn create(&self, req: &CreateWorkloadRequest) -> Result<Workload> {
        self.http.post(&self.base_path, req).await
    }

    pub async fn update(&self, uid: &str, req: &UpdateWorkloadRequest) -> Result<Workload> {
        self.http
            .patch(&format!("{}/{uid}", self.base_path), req)
            .await
    }

    pub async fn delete(&self, uid: &str) -> Result<()> {
        self.http.delete(&format!("{}/{uid}", self.base_path)).await
    }

    pub async fn deploy(&self, uid: &str) -> Result<WorkloadOperationResponse> {
        self.http
            .post_empty(&format!("{}/{uid}/deploy", self.base_path))
            .await
    }

    pub async fn suspend(&self, uid: &str) -> Result<WorkloadOperationResponse> {
        self.http
            .post_empty(&format!("{}/{uid}/suspend", self.base_path))
            .await
    }

    pub async fn reboot(&self, uid: &str) -> Result<WorkloadOperationResponse> {
        self.http
            .post_empty(&format!("{}/{uid}/reboot", self.base_path))
            .await
    }

    pub async fn state(&self, uid: &str) -> Result<WorkloadStateResponse> {
        self.http
            .get(&format!("{}/{uid}/state", self.base_path))
            .await
    }

    pub async fn events(&self, uid: &str, page: &Page) -> Result<List<WorkloadEvent>> {
        self.http
            .get_query(&format!("{}/{uid}/events", self.base_path), &page.query())
            .await
    }

    pub async fn logs(&self, uid: &str, opts: &LogOptions) -> Result<String> {
        self.http
            .get_text(
                &format!("{}/{uid}/logs", self.base_path),
                &log_query(opts, false),
            )
            .await
    }

    pub async fn logs_stream(
        &self,
        uid: &str,
        opts: &LogOptions,
    ) -> Result<impl Stream<Item = reqwest::Result<Bytes>>> {
        self.http
            .stream(
                &format!("{}/{uid}/logs", self.base_path),
                &log_query(opts, true),
            )
            .await
    }

    pub async fn exec(
        &self,
        uid: &str,
        command: &[String],
    ) -> Result<impl Stream<Item = reqwest::Result<Bytes>>> {
        let query: Vec<(&str, String)> =
            command.iter().map(|arg| ("command", arg.clone())).collect();
        self.http
            .post_stream(&format!("{}/{uid}/exec", self.base_path), &query)
            .await
    }

    pub async fn vm_images(&self) -> Result<Vec<VmImage>> {
        self.http
            .get(&format!("{}/vm-images", self.base_path))
            .await
    }

    pub async fn verify(&self, req: &VerifyWorkloadRequest) -> Result<VerifyWorkloadResponse> {
        self.http
            .post(&format!("{}/verify", self.base_path), req)
            .await
    }

    pub async fn attach_volume(
        &self,
        uid: &str,
        volume_uid: &str,
        req: &AttachVolumeRequest,
    ) -> Result<WorkloadVolume> {
        self.http
            .put(
                &format!("{}/{uid}/volumes/{volume_uid}", self.base_path),
                req,
            )
            .await
    }

    pub async fn detach_volume(&self, uid: &str, volume_uid: &str) -> Result<()> {
        self.http
            .delete(&format!("{}/{uid}/volumes/{volume_uid}", self.base_path))
            .await
    }

    pub async fn attach_ssh_key(
        &self,
        uid: &str,
        ssh_key_uid: &str,
    ) -> Result<WorkloadSshKeyAttachment> {
        self.http
            .put_empty(&format!("{}/{uid}/ssh-keys/{ssh_key_uid}", self.base_path))
            .await
    }

    pub async fn detach_ssh_key(&self, uid: &str, ssh_key_uid: &str) -> Result<()> {
        self.http
            .delete(&format!("{}/{uid}/ssh-keys/{ssh_key_uid}", self.base_path))
            .await
    }
}

fn log_query(opts: &LogOptions, follow: bool) -> Vec<(&'static str, String)> {
    let mut query = Vec::new();
    if let Some(since) = &opts.since {
        query.push(("since", since.clone()));
    }
    if let Some(tail) = opts.tail {
        query.push(("tail", tail.to_string()));
    }
    if opts.previous {
        query.push(("previous", "true".to_string()));
    }
    if let Some(log_type) = &opts.log_type {
        query.push(("type", log_type.clone()));
    }
    if follow {
        query.push(("follow", "true".to_string()));
    }
    query
}
