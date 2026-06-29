use bytes::Bytes;
use futures_util::Stream;

use crate::client::error::Result;
use crate::client::http::HttpClient;
use crate::client::pagination::{List, Page};
use crate::client::types::{
    AttachVolumeRequest, CreateWorkloadRequest, ListWorkloadsParams, LogOptions,
    UpdateWorkloadRequest, VerifyWorkloadRequest, VerifyWorkloadResponse, Workload, WorkloadEvent,
    WorkloadSshKeyAttachment, WorkloadStateResponse, WorkloadSummary, WorkloadVolume,
};

#[derive(Debug, Clone)]
pub struct Workloads {
    http: HttpClient,
}

impl Workloads {
    pub(crate) fn new(http: HttpClient) -> Self {
        Self { http }
    }

    pub async fn list(&self, params: &ListWorkloadsParams) -> Result<List<WorkloadSummary>> {
        let mut query = params.page.query();
        query.push(("type", "RENTAL".to_string()));
        if let Some(status) = &params.status {
            query.push(("status", status.clone()));
        }
        if let Some(project_id) = &params.project_id {
            query.push(("project_id", project_id.clone()));
        }
        if let Some(name) = &params.name {
            query.push(("name", name.clone()));
        }
        self.http.get_query("/workloads", &query).await
    }

    pub async fn get(&self, uid: &str) -> Result<Workload> {
        self.http.get(&format!("/workloads/{uid}")).await
    }

    pub async fn create(&self, req: &CreateWorkloadRequest) -> Result<Workload> {
        self.http.post("/workloads", req).await
    }

    pub async fn update(&self, uid: &str, req: &UpdateWorkloadRequest) -> Result<Workload> {
        self.http.patch(&format!("/workloads/{uid}"), req).await
    }

    pub async fn delete(&self, uid: &str) -> Result<()> {
        self.http.delete(&format!("/workloads/{uid}")).await
    }

    pub async fn deploy(&self, uid: &str) -> Result<WorkloadSummary> {
        self.http.post_empty(&format!("/workloads/{uid}/deploy")).await
    }

    pub async fn suspend(&self, uid: &str) -> Result<WorkloadSummary> {
        self.http.post_empty(&format!("/workloads/{uid}/suspend")).await
    }

    pub async fn reboot(&self, uid: &str) -> Result<WorkloadSummary> {
        self.http.post_empty(&format!("/workloads/{uid}/reboot")).await
    }

    pub async fn state(&self, uid: &str) -> Result<WorkloadStateResponse> {
        self.http.get(&format!("/workloads/{uid}/state")).await
    }

    pub async fn events(&self, uid: &str, page: &Page) -> Result<List<WorkloadEvent>> {
        self.http
            .get_query(&format!("/workloads/{uid}/events"), &page.query())
            .await
    }

    pub async fn logs(&self, uid: &str, opts: &LogOptions) -> Result<String> {
        self.http
            .get_text(&format!("/workloads/{uid}/logs"), &log_query(opts, false))
            .await
    }

    pub async fn logs_stream(
        &self,
        uid: &str,
        opts: &LogOptions,
    ) -> Result<impl Stream<Item = reqwest::Result<Bytes>>> {
        self.http
            .stream(&format!("/workloads/{uid}/logs"), &log_query(opts, true))
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
            .post_stream(&format!("/workloads/{uid}/exec"), &query)
            .await
    }

    pub async fn verify(&self, uid: &str, digest: &str) -> Result<VerifyWorkloadResponse> {
        let req = VerifyWorkloadRequest {
            uid: uid.to_string(),
            digest: digest.to_string(),
        };
        self.http.post("/workloads/verify", &req).await
    }

    pub async fn attach_volume(
        &self,
        uid: &str,
        volume_uid: &str,
        req: &AttachVolumeRequest,
    ) -> Result<WorkloadVolume> {
        self.http
            .put(&format!("/workloads/{uid}/volumes/{volume_uid}"), req)
            .await
    }

    pub async fn detach_volume(&self, uid: &str, volume_uid: &str) -> Result<()> {
        self.http
            .delete(&format!("/workloads/{uid}/volumes/{volume_uid}"))
            .await
    }

    pub async fn attach_ssh_key(
        &self,
        uid: &str,
        ssh_key_uid: &str,
    ) -> Result<WorkloadSshKeyAttachment> {
        self.http
            .put_empty(&format!("/workloads/{uid}/ssh-keys/{ssh_key_uid}"))
            .await
    }

    pub async fn detach_ssh_key(&self, uid: &str, ssh_key_uid: &str) -> Result<()> {
        self.http
            .delete(&format!("/workloads/{uid}/ssh-keys/{ssh_key_uid}"))
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
    if follow {
        query.push(("follow", "true".to_string()));
    }
    query
}
