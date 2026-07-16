use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshKey {
    pub uid: String,
    pub name: String,
    #[serde(rename = "public_key_raw")]
    pub public_key: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateSshKeyRequest {
    pub name: String,
    pub ssh_key: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateSshKeyRequest {
    pub name: String,
}
