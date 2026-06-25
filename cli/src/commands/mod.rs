pub mod auth;
pub mod inventory;
pub mod project;
pub mod ssh_key;
pub mod user;
pub mod version;
pub mod volume;
pub mod workload;

use crate::client::types::InventoryFilter;
use crate::client::Client;
use crate::error::{CliError, Result};
use crate::output::{format, prompt, OutputFormat};

pub struct Context {
    pub client: Client,
    pub format: OutputFormat,
}

impl Context {
    pub fn new(client: Client, format: OutputFormat) -> Self {
        Self { client, format }
    }

    pub fn json(&self) -> bool {
        self.format.is_json()
    }
}

pub async fn select_resource(ctx: &Context) -> Result<String> {
    let items = ctx
        .client
        .inventory()
        .list(&InventoryFilter::default())
        .await?;
    if items.is_empty() {
        return Err(CliError::Config(
            "no inventory available to select".to_string(),
        ));
    }
    let labels: Vec<String> = items
        .iter()
        .map(|i| {
            format!(
                "{:<24} {:>11}   {}",
                i.display_name,
                format::cost_per_hour(i.cost_per_hour),
                format::gpu_spec(i.spec.gpu_type.as_deref(), i.spec.gpu_count),
            )
        })
        .collect();
    let idx = prompt::select("Select a resource", &labels)?;
    Ok(items[idx].name.clone())
}
