use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct InventorySpec {
    #[serde(default)]
    pub gpu_type: Option<String>,
    #[serde(default)]
    pub gpu_count: u32,
    #[serde(default)]
    pub vcpu: u32,
    #[serde(default)]
    pub memory: u32,
    #[serde(default)]
    pub storage: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Inventory {
    pub name: String,
    pub display_name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "type")]
    pub resource_type: String,
    pub gpu: bool,
    pub spec: InventorySpec,
    pub cost_per_hour: f64,
    pub available: i32,
}

#[derive(Debug, Clone, Default)]
pub struct InventoryFilter {
    pub resource_type: Option<String>,
    pub gpu: Option<bool>,
}
