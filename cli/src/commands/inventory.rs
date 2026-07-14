use clap::Subcommand;
use comfy_table::Cell;

use crate::client::types::{Inventory, InventoryFilter};
use crate::commands::{workload, Context};
use crate::error::{CliError, Result};
use crate::output::{format, style, table};

#[derive(Debug, Subcommand)]
pub enum InventoryCommands {
    /// List available inventory
    List {
        /// Filter by resource type
        #[arg(long = "type")]
        resource_type: Option<String>,
        /// Only show GPU resources
        #[arg(long)]
        gpu: bool,
    },
    /// Show one SKU
    Get { sku: String },
}

pub async fn handle(ctx: &Context, cmd: &InventoryCommands) -> Result<()> {
    match cmd {
        InventoryCommands::List { resource_type, gpu } => {
            list(ctx, resource_type.clone(), *gpu).await
        }
        InventoryCommands::Get { sku } => get(ctx, sku).await,
    }
}

async fn list(ctx: &Context, resource_type: Option<String>, gpu: bool) -> Result<()> {
    let filter = InventoryFilter {
        resource_type,
        gpu: gpu.then_some(true),
    };
    let items = ctx.client.inventory().list(&filter).await?;

    if ctx.json() {
        return format::print_json(&items);
    }

    if items.is_empty() {
        style::dim("no inventory available");
        return Ok(());
    }

    let mut t = table::table(&["SKU", "DISPLAY", "GPU", "VCPU", "MEMORY", "PRICE", "AVAIL"]);
    for item in &items {
        t.add_row(vec![
            table::uid_cell(&item.name),
            Cell::new(&item.display_name),
            Cell::new(format::gpu_spec(
                item.spec.gpu_type.as_deref(),
                item.spec.gpu_count,
            )),
            Cell::new(item.spec.vcpu.to_string()),
            Cell::new(format::mib_to_human(item.spec.memory as u64)),
            Cell::new(format::cost_per_hour(item.cost_per_hour)),
            Cell::new(item.available.to_string()),
        ]);
    }
    table::print(&t);
    table::summary(workload::plural(items.len(), "resource"));
    Ok(())
}

async fn get(ctx: &Context, sku: &str) -> Result<()> {
    let items = ctx.client.inventory().list(&InventoryFilter::default()).await?;
    let item: &Inventory = items
        .iter()
        .find(|i| i.name == sku)
        .ok_or_else(|| CliError::Config(format!("unknown sku '{sku}'")))?;

    if ctx.json() {
        return format::print_json(item);
    }

    style::field("SKU", &item.name);
    style::field("Display", &item.display_name);
    if !item.description.is_empty() {
        style::field("Description", &item.description);
    }
    style::field("Type", &item.resource_type);
    style::field(
        "GPU",
        format::gpu_spec(item.spec.gpu_type.as_deref(), item.spec.gpu_count),
    );
    style::field("vCPU", item.spec.vcpu.to_string());
    style::field("Memory", format::mib_to_human(item.spec.memory as u64));
    style::field("Storage", format::mib_to_human(item.spec.storage as u64));
    style::field("Price", format::cost_per_hour(item.cost_per_hour));
    style::field("Available", item.available.to_string());
    Ok(())
}
