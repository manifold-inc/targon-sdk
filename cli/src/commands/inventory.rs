use clap::Args;

use crate::client::types::InventoryFilter;
use crate::commands::Context;
use crate::error::Result;
use crate::output::{format, style, table};

#[derive(Debug, Args)]
pub struct InventoryArgs {
    /// Filter by resource type
    #[arg(long = "type")]
    resource_type: Option<String>,
    /// Only show GPU resources
    #[arg(long)]
    gpu: bool,
}

pub async fn handle(ctx: &Context, args: &InventoryArgs) -> Result<()> {
    let filter = InventoryFilter {
        resource_type: args.resource_type.clone(),
        gpu: args.gpu.then_some(true),
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
            comfy_table::Cell::new(&item.name),
            comfy_table::Cell::new(&item.display_name),
            comfy_table::Cell::new(format::gpu_spec(
                item.spec.gpu_type.as_deref(),
                item.spec.gpu_count,
            )),
            comfy_table::Cell::new(item.spec.vcpu.to_string()),
            comfy_table::Cell::new(format::mib_to_human(item.spec.memory as u64)),
            comfy_table::Cell::new(format::cost_per_hour(item.cost_per_hour)),
            comfy_table::Cell::new(item.available.to_string()),
        ]);
    }
    println!("{t}");
    Ok(())
}
