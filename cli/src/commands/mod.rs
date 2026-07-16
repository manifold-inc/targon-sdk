pub mod auth;
pub mod inventory;
pub mod project;
pub mod rental;
pub mod ssh_key;
pub mod version;
pub mod vm;
pub mod volume;
pub mod whoami;
pub mod workload;

use crate::client::pagination::Page;
use crate::client::types::{Inventory, InventoryFilter, Workload};
use crate::client::Client;
use crate::error::{CliError, Result};
use crate::output::{format, prompt, style, OutputFormat};

pub struct Context {
    pub client: Client,
    pub format: OutputFormat,
    pub profile: String,
    pub base_url: String,
    pub project: Option<String>,
}

impl Context {
    pub fn new(
        client: Client,
        format: OutputFormat,
        profile: String,
        base_url: String,
        project: Option<String>,
    ) -> Self {
        Self {
            client,
            format,
            profile,
            base_url,
            project,
        }
    }

    pub fn json(&self) -> bool {
        self.format.is_json()
    }

    pub fn project(&self, flag: Option<String>) -> Option<String> {
        flag.or_else(|| self.project.clone())
    }
}

pub async fn select_resource(ctx: &Context, resource_type: &str) -> Result<String> {
    let items: Vec<_> = ctx
        .client
        .inventory()
        .list(&InventoryFilter {
            resource_type: Some(resource_type.to_string()),
            ..Default::default()
        })
        .await?
        .into_iter()
        .filter(|i| i.available > 0)
        .collect();
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

/// Best-effort inventory lookup so confirmation prompts can show the exact
/// hourly cost before spending money. Failures are swallowed — pricing is
/// display-only here.
pub(crate) async fn resource_pricing(
    ctx: &Context,
    resource_name: &str,
    resource_type: &str,
) -> Option<Inventory> {
    ctx.client
        .inventory()
        .list(&InventoryFilter {
            resource_type: Some(resource_type.to_string()),
            ..Default::default()
        })
        .await
        .ok()?
        .into_iter()
        .find(|i| i.name == resource_name)
}

pub(crate) fn prompt_list(label: &str) -> Result<Vec<String>> {
    Ok(prompt::optional(label)?
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default())
}

/// Multi-select registered SSH keys for VM deploy/create. Returns an empty
/// list when the account has no keys (password login still works).
pub async fn select_ssh_keys(ctx: &Context) -> Result<Vec<String>> {
    let keys = ctx.client.ssh_keys().list(&Page::default()).await?;
    if keys.items.is_empty() {
        style::dim("no ssh keys registered — continuing without (add with: targon key add)");
        return Ok(vec![]);
    }
    let labels: Vec<String> = keys
        .items
        .iter()
        .map(|k| format!("{:<24} {}", k.name, format::short_uid(&k.uid)))
        .collect();
    let selected = prompt::multi_select("SSH keys (space to toggle, enter to confirm)", &labels)?;
    Ok(selected
        .into_iter()
        .filter_map(|i| keys.items.get(i).map(|k| k.uid.clone()))
        .collect())
}

pub(crate) async fn ensure_rental(ctx: &Context, uid: &str, verb: &str) -> Result<Workload> {
    let workload = ctx.client.workloads().get(uid).await?;
    if workload.workload_type == "VM" {
        // First line is the red fact; subsequent lines render as dim → hints.
        let hint = match verb {
            "exec" => format!(
                "{uid} is a VM — exec is not supported\nconnect over SSH instead: targon workload get {uid}"
            ),
            "start" => format!(
                "{uid} is a VM — start applies only to rentals\nstart it with: targon vm start {uid}"
            ),
            _ => format!("{uid} is a VM — {verb} applies only to rentals"),
        };
        return Err(CliError::Config(hint));
    }
    Ok(workload)
}

pub(crate) async fn ensure_vm(ctx: &Context, uid: &str, verb: &str) -> Result<Workload> {
    let workload = ctx.client.workloads().get(uid).await?;
    if workload.workload_type != "VM" {
        let workload_type = &workload.workload_type;
        let hint = match verb {
            "start" => format!(
                "{uid} is a {workload_type} — start applies only to VMs\nstart it with: targon rental start {uid}"
            ),
            _ => format!("{uid} is a {workload_type} — {verb} applies only to VMs"),
        };
        return Err(CliError::Config(hint));
    }
    Ok(workload)
}
