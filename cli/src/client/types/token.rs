use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiToken {
    pub uid: String,
    pub name: String,
    #[serde(default)]
    pub token: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateApiTokenRequest {
    pub name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateApiTokenRequest {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenCreator {
    pub username: String,
    pub email: String,
    pub first_name: String,
    pub last_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceToken {
    pub uid: String,
    pub name: String,
    #[serde(default)]
    pub token: Option<String>,
    pub created_by: TokenCreator,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateServiceTokenRequest {
    pub name: String,
}
