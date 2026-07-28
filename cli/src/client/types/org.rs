use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::client::pagination::Page;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Org {
    pub uid: String,
    pub slug: String,
    pub name: String,
    pub org_type: String,
    pub billing_email: String,
    pub credits: i64,
    pub overage: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateOrgRequest {
    pub slug: String,
    pub name: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct UpdateOrgRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slug: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub billing_email: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum OrgRole {
    Owner,
    Admin,
    Member,
}

impl OrgRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Owner => "OWNER",
            Self::Admin => "ADMIN",
            Self::Member => "MEMBER",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum MembershipStatus {
    Active,
    Invited,
    Suspended,
}

impl MembershipStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "ACTIVE",
            Self::Invited => "INVITED",
            Self::Suspended => "SUSPENDED",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembershipUser {
    pub username: String,
    pub email: String,
    pub first_name: String,
    pub last_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Membership {
    pub uid: String,
    pub user: MembershipUser,
    #[serde(default)]
    pub invited_by_user: Option<MembershipUser>,
    pub role: OrgRole,
    pub status: MembershipStatus,
    #[serde(default)]
    pub invited_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub joined_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default)]
pub struct ListMembersParams {
    pub page: Page,
    pub role: Option<OrgRole>,
    pub status: Option<MembershipStatus>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateMemberRequest {
    pub role: OrgRole,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wallet {
    pub address: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credits {
    pub credits: f64,
    pub currency: String,
}
