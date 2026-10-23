use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::common::{Port, State};
use crate::client::pagination::Page;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum WorkloadType {
    #[default]
    Vm,
    Bm,
}

impl WorkloadType {
    pub fn as_str(self) -> &'static str {
        match self {
            WorkloadType::Vm => "VM",
            WorkloadType::Bm => "BM",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VmConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BmConfigInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmConfig {
    #[serde(default)]
    pub region: String,
    #[serde(default)]
    pub hostname: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmImage {
    pub name: String,
    pub display_name: String,
    #[serde(default)]
    pub os_family: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmImage {
    pub name: String,
    pub display_name: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadUrl {
    pub port: u16,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadResource {
    pub name: String,
    pub display_name: String,
    #[serde(default)]
    pub gpu_type: Option<String>,
    #[serde(default)]
    pub gpu_count: Option<u32>,
    #[serde(default)]
    pub disk_size_mib: Option<u64>,
    #[serde(default)]
    pub network_mode: Option<String>,

    #[serde(default)]
    pub vcpu: u32,
    #[serde(default)]
    pub memory: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadState {
    pub status: State,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub urls: Vec<WorkloadUrl>,
    #[serde(default)]
    pub public_ip: Option<String>,
    #[serde(default)]
    pub ssh_port: Option<u16>,
    pub ready_replicas: u32,
    pub total_replicas: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadSshKey {
    pub uid: String,
    pub name: String,
    #[serde(rename = "public_key_raw")]
    pub public_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workload {
    pub uid: String,
    pub name: String,
    #[serde(rename = "type")]
    pub workload_type: String,
    #[serde(default)]
    pub image: Option<String>,
    #[serde(default)]
    pub resource_name: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub ports: Vec<Port>,
    #[serde(default)]
    pub ssh_keys: Vec<WorkloadSshKey>,
    #[serde(default)]
    pub state: Option<WorkloadState>,
    #[serde(default)]
    pub resource: Option<WorkloadResource>,
    #[serde(default)]
    pub vm_config: Option<VmConfig>,
    #[serde(default)]
    pub bm_config: Option<BmConfig>,
    #[serde(default)]
    pub reservation_uid: Option<String>,
    #[serde(default)]
    pub cost_per_hour: Option<f64>,
    #[serde(default)]
    pub revision: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadOperationResponse {
    pub uid: String,
    pub name: String,
    #[serde(rename = "type")]
    pub workload_type: String,
    #[serde(default)]
    pub image: Option<String>,
    #[serde(default)]
    pub state: Option<WorkloadState>,
    #[serde(default)]
    pub resource: Option<WorkloadResource>,
    #[serde(default)]
    pub reservation_uid: Option<String>,
    #[serde(default)]
    pub cost_per_hour: Option<f64>,
    #[serde(default)]
    pub revision: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadStateResponse {
    pub uid: String,
    pub workload_type: String,
    pub status: State,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub urls: Vec<WorkloadUrl>,
    #[serde(default)]
    pub public_ip: Option<String>,
    #[serde(default)]
    pub ssh_port: Option<u16>,
    pub ready_replicas: u32,
    pub total_replicas: u32,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadEvent {
    pub workload_uid: String,
    pub workload_type: String,
    pub event_type: String,
    #[serde(default)]
    pub new_status: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub display_message: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub resource_name: Option<String>,
    #[serde(default)]
    pub pod_name: Option<String>,
    #[serde(default)]
    pub container_name: Option<String>,
    #[serde(default)]
    pub container_image: Option<String>,
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub replica_count: Option<u32>,
    #[serde(default)]
    pub old_replica_count: Option<u32>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateWorkloadRequest {
    pub name: String,
    pub image: String,
    pub resource_name: String,
    #[serde(rename = "type")]
    pub workload_type: WorkloadType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ports: Vec<Port>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ssh_keys: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vm_config: Option<VmConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reservation_uid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm_config: Option<BmConfigInput>,
}

impl CreateWorkloadRequest {
    pub fn new(
        workload_type: WorkloadType,
        name: impl Into<String>,
        image: impl Into<String>,
        resource_name: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            image: image.into(),
            resource_name: resource_name.into(),
            workload_type,
            project_id: None,
            ports: Vec::new(),
            ssh_keys: Vec::new(),
            vm_config: None,
            region: None,
            reservation_uid: None,
            bm_config: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ReinstallWorkloadRequest {
    pub image: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ssh_keys: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct UpdateWorkloadRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ports: Option<Vec<Port>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssh_keys: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default)]
pub struct ListWorkloadsParams {
    pub page: Page,
    pub workload_type: Option<String>,
    pub status: Option<String>,
    pub project_id: Option<String>,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct LogOptions {
    pub since: Option<String>,
    pub tail: Option<u32>,
    pub previous: bool,
    pub log_type: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VerifyWorkloadRequest {
    pub uid: String,
    pub digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyWorkloadResponse {
    pub verified: bool,
}
