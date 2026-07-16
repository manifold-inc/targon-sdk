use std::collections::BTreeMap;
use std::io::IsTerminal;

use clap::Subcommand;
use colored::Colorize;

use crate::client::types::{Inventory, InventoryFilter};
use crate::commands::Context;
use crate::error::{CliError, Result};
use crate::output::{format, palettes, style};

/// Types fetched for the default mixed inventory view.
const ALL_TYPES: &[&str] = &["rental", "vm", "storage"];

const COL_SKU: usize = 17;
const COL_TYPE: usize = 8;
const COL_GPU: usize = 4;
const COL_VCPU: usize = 7;
const COL_MEMORY: usize = 8;
const COL_PRICE: usize = 10;
const COL_AVAIL: usize = 5;

#[derive(Debug, Subcommand)]
pub enum InventoryCommands {
    /// List available inventory
    List {
        /// Filter by offering type: rental, vm, storage
        #[arg(long = "type", value_name = "TYPE")]
        resource_type: Option<String>,
        /// Only show GPU resources
        #[arg(long)]
        gpu: bool,
        /// Hide sold-out SKUs
        #[arg(long)]
        available: bool,
    },
    /// Show one SKU
    Get { sku: String },
}

pub async fn handle(ctx: &Context, cmd: &InventoryCommands) -> Result<()> {
    match cmd {
        InventoryCommands::List {
            resource_type,
            gpu,
            available,
        } => list(ctx, resource_type.clone(), *gpu, *available).await,
        InventoryCommands::Get { sku } => get(ctx, sku).await,
    }
}

async fn list(
    ctx: &Context,
    resource_type: Option<String>,
    gpu: bool,
    available_only: bool,
) -> Result<()> {
    let mut items = fetch_inventory(ctx, resource_type.as_deref(), gpu).await?;

    if available_only {
        items.retain(|i| i.available > 0);
    }

    if ctx.json() {
        return format::print_json(&items);
    }

    if items.is_empty() {
        style::dim("no inventory available");
        return Ok(());
    }

    if std::io::stdout().is_terminal() {
        print_grouped(&items);
    } else {
        print_plain(&items);
    }
    Ok(())
}

async fn fetch_inventory(
    ctx: &Context,
    resource_type: Option<&str>,
    gpu: bool,
) -> Result<Vec<Inventory>> {
    let gpu = gpu.then_some(true);
    if let Some(resource_type) = resource_type {
        return ctx
            .client
            .inventory()
            .list(&InventoryFilter {
                resource_type: Some(resource_type.to_string()),
                gpu,
            })
            .await
            .map_err(Into::into);
    }

    // Default mixed view: fan-out per offering type and merge.
    let mut items = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for ty in ALL_TYPES {
        let batch = ctx
            .client
            .inventory()
            .list(&InventoryFilter {
                resource_type: Some((*ty).to_string()),
                gpu,
            })
            .await?;
        for item in batch {
            let key = (item.name.clone(), item.resource_type.to_ascii_lowercase());
            if seen.insert(key) {
                items.push(item);
            }
        }
    }
    Ok(items)
}

fn print_grouped(items: &[Inventory]) {
    let groups = group_by_family(items);
    let width = table_width();

    println!();
    println!(
        "  {} {} {} {} {} {} {}",
        pad_header("SKU", COL_SKU),
        pad_header("TYPE", COL_TYPE),
        pad_header("GPU", COL_GPU),
        pad_header("VCPU", COL_VCPU),
        pad_header("MEMORY", COL_MEMORY),
        pad_header("PRICE", COL_PRICE),
        pad_header("AVAIL", COL_AVAIL),
    );
    println!();

    for (family, rows) in &groups {
        print_family_header(family, width);
        for item in rows {
            print_row(item);
        }
        println!();
    }

    let total = items.len();
    let in_stock = items.iter().filter(|i| i.available > 0).count();
    let sku_word = if total == 1 { "SKU" } else { "SKUs" };
    let avail = if palettes::colors_enabled() {
        in_stock.to_string().color(palettes::SUCCESS).to_string()
    } else {
        in_stock.to_string()
    };
    println!(
        "  {} {} {} {} {} {} {}",
        total.to_string().color(palettes::DIM),
        sku_word.color(palettes::DIM),
        style::SEP.color(palettes::DIM),
        avail,
        "available now".color(palettes::DIM),
        style::SEP.color(palettes::DIM),
        "--type rental|vm|storage · --available hides sold-out".color(palettes::DIM),
    );
}

fn print_plain(items: &[Inventory]) {
    println!("SKU\tTYPE\tGPU\tVCPU\tMEMORY\tPRICE\tAVAIL\tFAMILY");
    for item in items {
        let ty = format::inventory_type_label(&item.resource_type);
        let (gpu, vcpu, memory) = row_specs(item);
        let avail = if item.available > 0 {
            item.available.to_string()
        } else {
            "0".to_string()
        };
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            item.name,
            ty,
            gpu,
            vcpu,
            memory,
            format::cost_per_hour(item.cost_per_hour),
            avail,
            hardware_family(item),
        );
    }
}

fn print_family_header(family: &str, width: usize) {
    let fill = width.saturating_sub(family.chars().count() + 1);
    if palettes::colors_enabled() {
        println!(
            "  {} {}",
            family.color(palettes::HEADER).bold(),
            "─".repeat(fill).color(palettes::BORDER)
        );
    } else {
        println!("  {family} {}", "─".repeat(fill));
    }
}

fn print_row(item: &Inventory) {
    let sold_out = item.available <= 0;
    let ty = format::inventory_type_label(&item.resource_type);
    let (gpu, vcpu, memory) = row_specs(item);
    let price = format::cost_per_hour(item.cost_per_hour);
    let avail = if sold_out {
        "–".to_string()
    } else {
        item.available.to_string()
    };

    if sold_out {
        println!(
            "  {} {} {} {} {} {} {}",
            dim_pad(&item.name, COL_SKU),
            dim_pad(&ty, COL_TYPE),
            dim_pad(&gpu, COL_GPU),
            dim_pad(&vcpu, COL_VCPU),
            dim_pad(&memory, COL_MEMORY),
            dim_pad(&price, COL_PRICE),
            dim_pad(&avail, COL_AVAIL),
        );
        return;
    }

    let type_colored = {
        let padded = format!("{ty:<COL_TYPE$}");
        if palettes::colors_enabled() {
            padded.color(palettes::workload_type_color(&ty)).to_string()
        } else {
            padded
        }
    };
    let avail_colored = {
        let padded = format!("{avail:<COL_AVAIL$}");
        if palettes::colors_enabled() {
            padded.color(palettes::SUCCESS).to_string()
        } else {
            padded
        }
    };

    println!(
        "  {} {} {} {} {} {} {}",
        format!("{:<COL_SKU$}", item.name),
        type_colored,
        format!("{gpu:<COL_GPU$}"),
        format!("{vcpu:<COL_VCPU$}"),
        format!("{memory:<COL_MEMORY$}"),
        format!("{price:<COL_PRICE$}"),
        avail_colored,
    );
}

fn row_specs(item: &Inventory) -> (String, String, String) {
    let is_storage = matches!(
        item.resource_type.to_ascii_lowercase().as_str(),
        "storage" | "volume"
    );
    if is_storage {
        return ("–".into(), "–".into(), "–".into());
    }
    let gpu = format::gpu_count(item.spec.gpu_count);
    let vcpu = if item.spec.vcpu == 0 {
        "–".into()
    } else {
        item.spec.vcpu.to_string()
    };
    let memory = format::mib_to_human(item.spec.memory as u64);
    (gpu, vcpu, memory)
}

fn group_by_family(items: &[Inventory]) -> Vec<(String, Vec<&Inventory>)> {
    let mut map: BTreeMap<(u8, String), Vec<&Inventory>> = BTreeMap::new();
    for item in items {
        let family = hardware_family(item);
        let key = family_sort_key(&family);
        map.entry(key).or_default().push(item);
    }
    map.into_iter()
        .map(|((_, family), mut rows)| {
            rows.sort_by(|a, b| {
                a.name
                    .cmp(&b.name)
                    .then_with(|| a.resource_type.cmp(&b.resource_type))
            });
            (family, rows)
        })
        .collect()
}

fn hardware_family(item: &Inventory) -> String {
    let ty = item.resource_type.to_ascii_lowercase();
    if matches!(ty.as_str(), "storage" | "volume") {
        return "STORAGE".to_string();
    }

    if let Some(gpu) = item.spec.gpu_type.as_deref().filter(|g| !g.is_empty()) {
        return normalize_gpu_family(gpu);
    }

    let name = item.name.to_ascii_lowercase();
    if name.contains("storage") {
        return "STORAGE".to_string();
    }
    if let Some(family) = family_from_name(&name) {
        return family;
    }
    if !item.gpu || item.spec.gpu_count == 0 {
        return "CPU".to_string();
    }
    "OTHER".to_string()
}

fn normalize_gpu_family(gpu: &str) -> String {
    let g = gpu.to_ascii_uppercase().replace('_', " ");
    if g.contains("B200") {
        return "B200".to_string();
    }
    if g.contains("H200") {
        return "H200".to_string();
    }
    if g.contains("H100") {
        return "H100".to_string();
    }
    if g.contains("A100") {
        return "A100".to_string();
    }
    if g.contains("4090") {
        return "RTX 4090".to_string();
    }
    if g.contains("3090") {
        return "RTX 3090".to_string();
    }
    if g.contains("A6000") {
        return "A6000".to_string();
    }
    // Fall back to a cleaned display of the GPU string.
    gpu.split_whitespace()
        .last()
        .unwrap_or(gpu)
        .to_string()
}

fn family_from_name(name: &str) -> Option<String> {
    if name.contains("b200") {
        Some("B200".into())
    } else if name.contains("h200") {
        Some("H200".into())
    } else if name.contains("h100") {
        Some("H100".into())
    } else if name.contains("a100") {
        Some("A100".into())
    } else if name.contains("4090") || name.contains("rtx4090") {
        Some("RTX 4090".into())
    } else if name.starts_with("cpu") || name.contains("cpu-") {
        Some("CPU".into())
    } else {
        None
    }
}

fn family_sort_key(family: &str) -> (u8, String) {
    let rank = match family {
        "CPU" => 0,
        "RTX 3090" => 10,
        "RTX 4090" => 11,
        "A6000" => 20,
        "A100" => 30,
        "H100" => 40,
        "H200" => 41,
        "B200" => 42,
        "STORAGE" => 200,
        "OTHER" => 180,
        _ => 100,
    };
    (rank, family.to_string())
}

fn table_width() -> usize {
    COL_SKU + 1 + COL_TYPE + 1 + COL_GPU + 1 + COL_VCPU + 1 + COL_MEMORY + 1 + COL_PRICE + 1 + COL_AVAIL
}

fn pad_header(label: &str, width: usize) -> String {
    let s = format!("{label:<width$}");
    if palettes::colors_enabled() {
        s.color(palettes::DIM).to_string()
    } else {
        s
    }
}

fn dim_pad(value: &str, width: usize) -> String {
    let s = format!("{value:<width$}");
    if palettes::colors_enabled() {
        s.color(palettes::DIM).to_string()
    } else {
        s
    }
}

async fn get(ctx: &Context, sku: &str) -> Result<()> {
    let items = fetch_inventory(ctx, None, false).await?;
    let matches: Vec<&Inventory> = items.iter().filter(|i| i.name == sku).collect();
    if matches.is_empty() {
        return Err(CliError::Config(format!("unknown sku '{sku}'")));
    }

    if ctx.json() {
        if matches.len() == 1 {
            return format::print_json(matches[0]);
        }
        return format::print_json(&matches);
    }

    for (idx, item) in matches.iter().enumerate() {
        if idx > 0 {
            println!();
        }
        let ty = format::inventory_type_label(&item.resource_type);
        style::field("SKU", &item.name);
        style::field("Display", &item.display_name);
        if !item.description.is_empty() {
            style::field("Description", &item.description);
        }
        style::field(
            "Type",
            ty.color(palettes::workload_type_color(&ty)).to_string(),
        );
        style::field("Family", hardware_family(item));
        style::field(
            "GPU",
            format::gpu_spec(item.spec.gpu_type.as_deref(), item.spec.gpu_count),
        );
        style::field("vCPU", item.spec.vcpu.to_string());
        style::field("Memory", format::mib_to_human(item.spec.memory as u64));
        style::field("Storage", format::mib_to_human(item.spec.storage as u64));
        style::field("Price", format::cost_per_hour(item.cost_per_hour));
        let avail = if item.available > 0 {
            item.available
                .to_string()
                .color(palettes::SUCCESS)
                .to_string()
        } else {
            "–".color(palettes::DIM).to_string()
        };
        style::field("Available", avail);
    }
    Ok(())
}
