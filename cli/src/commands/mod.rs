pub mod api_token;
pub mod auth;
pub mod bm;
pub mod inventory;
pub mod member;
pub mod org;
pub mod project;
pub mod region;
pub mod service_token;
pub mod ssh_key;
pub mod version;
pub mod vm;
pub mod whoami;
pub mod workload;

use colored::Colorize;

use crate::client::pagination::Page;
use crate::client::types::{Inventory, InventoryFilter, Workload};
use crate::client::Client;
use crate::error::{CliError, Result};
use crate::output::{format, palettes, prompt, style, OutputFormat};

pub struct Context {
    pub client: Client,
    pub format: OutputFormat,
    pub profile: String,
    pub base_url: String,
    pub org: Option<String>,
    pub project: Option<String>,
}

impl Context {
    pub fn new(
        client: Client,
        format: OutputFormat,
        profile: String,
        base_url: String,
        org: Option<String>,
        project: Option<String>,
    ) -> Self {
        Self {
            client,
            format,
            profile,
            base_url,
            org,
            project,
        }
    }

    pub fn json(&self) -> bool {
        self.format.is_json()
    }

    pub fn project(&self, flag: Option<String>) -> Option<String> {
        flag.or_else(|| self.project.clone())
    }

    pub fn org(&self) -> Result<&str> {
        self.org.as_deref().ok_or_else(|| {
            CliError::Config(
                "organization is required\nselect one with: targon org use <slug> (or pass --org)"
                    .to_string(),
            )
        })
    }
}

pub async fn select_resource(ctx: &Context, resource_type: &str) -> Result<String> {
    select_resource_in_region(ctx, resource_type, None).await
}

pub async fn select_resource_in_region(
    ctx: &Context,
    resource_type: &str,
    region: Option<&str>,
) -> Result<String> {
    let items: Vec<_> = ctx
        .client
        .inventory()
        .list(&InventoryFilter {
            resource_type: Some(resource_type.to_string()),
            region: region.map(str::to_string),
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
    resource_pricing_in_region(ctx, resource_name, resource_type, None).await
}

pub(crate) async fn resource_pricing_in_region(
    ctx: &Context,
    resource_name: &str,
    resource_type: &str,
    region: Option<&str>,
) -> Option<Inventory> {
    ctx.client
        .inventory()
        .list(&InventoryFilter {
            resource_type: Some(resource_type.to_string()),
            region: region.map(str::to_string),
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
    let keys = ctx
        .client
        .ssh_keys(ctx.org()?)
        .list(&Page::default())
        .await?;
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

/// Bare metal has no password path, so at least one key must be chosen.
pub async fn select_ssh_keys_required(ctx: &Context) -> Result<Vec<String>> {
    let keys = ctx
        .client
        .ssh_keys(ctx.org()?)
        .list(&Page::default())
        .await?;
    if keys.items.is_empty() {
        return Err(CliError::Config(
            "bare metal requires at least one ssh key\nregister one with: targon key add"
                .to_string(),
        ));
    }
    let labels: Vec<String> = keys
        .items
        .iter()
        .map(|k| format!("{:<24} {}", k.name, format::short_uid(&k.uid)))
        .collect();
    let selected = prompt::multi_select("SSH keys (space to toggle, enter to confirm)", &labels)?;
    let chosen: Vec<String> = selected
        .into_iter()
        .filter_map(|i| keys.items.get(i).map(|k| k.uid.clone()))
        .collect();
    if chosen.is_empty() {
        return Err(CliError::Config(
            "bare metal requires at least one ssh key".to_string(),
        ));
    }
    Ok(chosen)
}

pub(crate) fn resource_value(
    resource_name: &str,
    pricing: Option<&Inventory>,
) -> String {
    match pricing {
        Some(p) => format!(
            "{} {} {}",
            p.display_name,
            style::SEP,
            format::cost_per_hour(p.cost_per_hour).color(palettes::SUCCESS)
        ),
        None => resource_name.to_string(),
    }
}

/// RFC-1123: dot-separated labels of lowercase alphanumerics and hyphens,
/// each starting and ending alphanumeric. ntha caps the whole name at 63.
pub(crate) fn validate_hostname(raw: &str) -> Result<String> {
    let hostname = raw.trim().to_ascii_lowercase();
    let invalid = || {
        CliError::Config(format!(
            "invalid --hostname '{raw}'\nuse lowercase letters, digits, hyphens and dots, at most 63 characters"
        ))
    };
    if hostname.is_empty() || hostname.chars().count() > 63 {
        return Err(invalid());
    }
    for label in hostname.split('.') {
        if label.is_empty() || label.len() > 63 {
            return Err(invalid());
        }
        if !label
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            return Err(invalid());
        }
        if label.starts_with('-') || label.ends_with('-') {
            return Err(invalid());
        }
    }
    Ok(hostname)
}

pub(crate) async fn ensure_vm(ctx: &Context, uid: &str, verb: &str) -> Result<Workload> {
    ensure_type(ctx, uid, verb, "VM").await
}

pub(crate) async fn ensure_bm(ctx: &Context, uid: &str, verb: &str) -> Result<Workload> {
    ensure_type(ctx, uid, verb, "BM").await
}

async fn ensure_type(ctx: &Context, uid: &str, verb: &str, want: &str) -> Result<Workload> {
    let workload = ctx.client.workloads(ctx.org()?).get(uid).await?;
    if workload.workload_type != want {
        let workload_type = &workload.workload_type;
        let hint = format!(
            "{uid} is a {workload_type} — {verb} applies only to {want}s\nrun it with: targon {} {verb} {uid}",
            family(workload_type)
        );
        return Err(CliError::Config(hint));
    }
    Ok(workload)
}

/// Command family that owns a workload type, for "run it with:" hints.
fn family(workload_type: &str) -> &'static str {
    match workload_type {
        "VM" => "vm",
        "BM" => "bm",
        _ => "workload",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_org_has_an_actionable_error() {
        let client = Client::builder()
            .api_key("test")
            .build()
            .expect("test client");
        let ctx = Context::new(
            client,
            OutputFormat::Human,
            "default".to_string(),
            "https://example.test".to_string(),
            None,
            None,
        );

        let error = ctx.org().expect_err("org should be required").to_string();
        assert!(error.contains("targon org use <slug>"));
    }
}
