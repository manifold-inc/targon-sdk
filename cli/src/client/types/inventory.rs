use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InventorySpec {
    #[serde(default)]
    pub gpu_vendor: Option<String>,
    #[serde(default)]
    pub gpu_model: Option<String>,
    #[serde(default)]
    pub gpu_type: Option<String>,
    #[serde(default)]
    pub gpu_count: u32,
    #[serde(default)]
    pub cpu_millicores: u32,
    #[serde(default)]
    pub cpu_vendor: Option<String>,
    #[serde(default)]
    pub cpu_model: Option<String>,
    #[serde(default)]
    pub cpu_sockets: Option<u32>,
    #[serde(default)]
    pub memory_mib: u64,
    #[serde(default)]
    pub disk_size_mib: u64,
    #[serde(default)]
    pub disk_label: Option<String>,
    #[serde(default)]
    pub network_mode: Option<String>,
    #[serde(default)]
    pub cc_enabled: Option<bool>,

    #[serde(default)]
    pub vcpu: u32,
    #[serde(default)]
    pub memory: u64,
    #[serde(default)]
    pub storage: u64,
}

/// `vcpu`, `memory` and `storage` are deprecated aliases the API drops after
/// its compatibility window, so the canonical fields win and the aliases are
/// only consulted while the canonical ones are still absent.
impl InventorySpec {
    pub fn cpu(&self) -> u32 {
        if self.cpu_millicores > 0 {
            self.cpu_millicores
        } else {
            self.vcpu
        }
    }

    pub fn mem(&self) -> u64 {
        if self.memory_mib > 0 {
            self.memory_mib
        } else {
            self.memory
        }
    }

    pub fn disk(&self) -> u64 {
        if self.disk_size_mib > 0 {
            self.disk_size_mib
        } else {
            self.storage
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryRegion {
    pub uid: String,
    pub display_name: String,
    #[serde(default)]
    pub country: String,
    #[serde(default)]
    pub continent: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryReservation {
    pub uid: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub workload_uid: Option<String>,
    #[serde(default)]
    pub reservation_end_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryReserved {
    #[serde(default)]
    pub count: i32,
    #[serde(default)]
    pub reservations: Vec<InventoryReservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    #[serde(default)]
    pub region: Option<InventoryRegion>,
    #[serde(default)]
    pub reserved: Option<InventoryReserved>,
}

impl Inventory {
    pub fn region_uid(&self) -> Option<&str> {
        self.region.as_ref().map(|r| r.uid.as_str())
    }
}

#[derive(Debug, Clone, Default)]
pub struct InventoryFilter {
    pub resource_type: Option<String>,
    pub gpu: Option<bool>,
    pub region: Option<String>,
}
