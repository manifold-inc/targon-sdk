use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct State(pub String);

impl State {
    pub const PROVISIONING: &str = "provisioning";
    pub const RUNNING: &str = "running";
    pub const ERROR: &str = "error";
    pub const SUSPENDED: &str = "suspended";
    pub const DELETED: &str = "deleted";
    pub const REGISTERED: &str = "registered";

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PortProtocol {
    #[default]
    Tcp,
    Udp,
    Sctp,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PortRouting {
    #[default]
    Proxied,
    Direct,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Port {
    pub port: u16,
    #[serde(default)]
    pub protocol: PortProtocol,
    #[serde(default)]
    pub routing: PortRouting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Version {
    pub name: String,
    pub version: String,
    #[serde(rename = "gitHash")]
    pub git_hash: String,
    #[serde(rename = "buildDate")]
    pub build_date: String,
}
