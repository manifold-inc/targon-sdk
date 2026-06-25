use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::common::State;
use crate::client::pagination::Page;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeState {
    pub status: State,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Volume {
    pub uid: String,
    pub name: String,
    pub resource_name: String,
    pub size: u64,
    #[serde(default)]
    pub cost_per_hour: Option<f64>,
    #[serde(default)]
    pub pvc_name: Option<String>,
    #[serde(default)]
    pub last_backup_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub state: Option<VolumeState>,
    #[serde(default)]
    pub workload_uid: Option<String>,
    #[serde(default)]
    pub mount_path: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeStateResponse {
    pub uid: String,
    pub status: State,
    #[serde(default)]
    pub message: String,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeOperationResponse {
    pub uid: String,
    #[serde(default)]
    pub state: Option<VolumeState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeEvent {
    pub volume_uid: String,
    pub event_type: String,
    #[serde(default)]
    pub old_status: Option<String>,
    #[serde(default)]
    pub new_status: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub resource_name: Option<String>,
    #[serde(default)]
    pub pvc_name: Option<String>,
    #[serde(default)]
    pub requested_size: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateVolumeRequest {
    pub name: String,
    pub resource_name: String,
    pub size_in_mb: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateVolumeRequest {
    pub name: String,
}

#[derive(Debug, Clone, Default)]
pub struct ListVolumesParams {
    pub page: Page,
    pub workload_uid: Option<String>,
}
