use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct Project {
    pub uid: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateProjectRequest {
    pub name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateProjectRequest {
    pub name: String,
}
