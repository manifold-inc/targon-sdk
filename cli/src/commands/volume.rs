use clap::Subcommand;
use colored::Colorize;
use comfy_table::Cell;

use crate::client::pagination::Page;
use crate::client::types::CreateVolumeRequest;
use crate::commands::{self, workload, Context};
use crate::error::Result;
use crate::output::{format, palettes, prompt, style, table};

#[derive(Debug, Subcommand)]
pub enum VolumeCommands {
    /// Create a volume
    Create {
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        resource: Option<String>,
        /// Size in GiB
        #[arg(long)]
        size: Option<u64>,
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// List volumes
    List {
        #[arg(long = "workload")]
        workload_uid: Option<String>,
    },
    /// Show a volume
    Get { uid: String },
    /// Delete a volume
    Delete {
        uid: String,
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Show volume state
    State { uid: String },
    /// Show volume events
    Events {
        uid: String,
        #[arg(long, default_value_t = 20)]
        limit: u32,
    },
}

pub async fn handle(ctx: &Context, cmd: &VolumeCommands) -> Result<()> {
    match cmd {
        VolumeCommands::Create {
            name,
            resource,
            size,
            yes,
        } => create(ctx, name.clone(), resource.clone(), *size, *yes).await,
        VolumeCommands::List { workload_uid } => list(ctx, workload_uid.clone()).await,
        VolumeCommands::Get { uid } => get(ctx, uid).await,
        VolumeCommands::Delete { uid, yes } => delete(ctx, uid, *yes).await,
        VolumeCommands::State { uid } => state(ctx, uid).await,
        VolumeCommands::Events { uid, limit } => events(ctx, uid, *limit).await,
    }
}

async fn create(
    ctx: &Context,
    name: Option<String>,
    resource: Option<String>,
    size: Option<u64>,
    yes: bool,
) -> Result<()> {
    let name = match name {
        Some(name) => name,
        None => {
            prompt::require_tty("--name")?;
            prompt::input("Volume name")?
        }
    };
    let resource_name = match resource {
        Some(resource) => resource,
        None => {
            prompt::require_tty("--resource")?;
            commands::select_resource(ctx, "storage").await?
        }
    };
    let size_gib = match size {
        Some(size) => size,
        None => {
            prompt::require_tty("--size")?;
            prompt::input_default("Size (GiB)", "10")?
                .trim()
                .parse()
                .map_err(|_| crate::error::CliError::Config("invalid size".to_string()))?
        }
    };

    if prompt::is_tty() && !yes {
        eprintln!();
        style::summary_box(
            "volume",
            &[
                ("name", name.clone()),
                ("resource", resource_name.clone()),
                ("size", format!("{size_gib} GiB")),
            ],
        );
        if !prompt::confirm("Create this volume?", false)? {
            return Err(crate::error::CliError::Cancelled);
        }
    }

    let req = CreateVolumeRequest {
        name,
        resource_name,
        size_in_mb: size_gib * 1024,
    };
    let volume = ctx.client.volumes().create(&req).await?;
    if ctx.json() {
        return format::print_json(&volume);
    }
    style::success(format!("created volume {}", volume.uid));
    Ok(())
}

async fn list(ctx: &Context, workload_uid: Option<String>) -> Result<()> {
    let params = crate::client::types::ListVolumesParams {
        page: Page::default(),
        workload_uid,
    };
    let volumes = ctx.client.volumes().list(&params).await?;
    if ctx.json() {
        return format::print_json(&volumes);
    }
    if volumes.items.is_empty() {
        style::dim("no volumes");
        return Ok(());
    }
    let mut t = table::table(&["UID", "NAME", "RESOURCE", "SIZE", "STATE", "ATTACHED"]);
    let mut attached = 0usize;
    for volume in &volumes.items {
        let status = volume
            .state
            .as_ref()
            .map(|s| s.status.to_string())
            .unwrap_or_default();
        if volume.workload_uid.is_some() {
            attached += 1;
        }
        t.add_row(vec![
            table::uid_cell(&volume.uid),
            Cell::new(&volume.name),
            Cell::new(&volume.resource_name),
            Cell::new(format::mib_to_human(volume.size)),
            table::state_cell(&status),
            table::dim_cell(volume.workload_uid.clone().unwrap_or_else(|| "-".to_string())),
        ]);
    }
    table::print(&t);
    table::summary(format!(
        "{} {} {attached} attached",
        workload::plural(volumes.items.len(), "volume"),
        style::SEP
    ));
    Ok(())
}

async fn get(ctx: &Context, uid: &str) -> Result<()> {
    let volume = ctx.client.volumes().get(uid).await?;
    if ctx.json() {
        return format::print_json(&volume);
    }
    style::field("UID", volume.uid.color(palettes::ACCENT).to_string());
    style::field("Name", &volume.name);
    style::field("Resource", &volume.resource_name);
    style::field("Size", format::mib_to_human(volume.size));
    if let Some(state) = &volume.state {
        style::field("State", format::state_badge(state.status.as_str()));
    }
    if let Some(cost) = volume.cost_per_hour {
        style::field("Cost", format::cost_per_hour(cost));
    }
    if let Some(workload) = &volume.workload_uid {
        style::field("Attached to", workload);
    }
    style::field("Created", format::relative_time(volume.created_at));
    Ok(())
}

async fn delete(ctx: &Context, uid: &str, yes: bool) -> Result<()> {
    if prompt::is_tty() && !yes && !prompt::confirm(&format!("Delete volume {uid}?"), false)? {
        return Err(crate::error::CliError::Cancelled);
    }
    ctx.client.volumes().delete(uid).await?;
    style::success(format!("deleted volume {uid}"));
    Ok(())
}

async fn state(ctx: &Context, uid: &str) -> Result<()> {
    let state = ctx.client.volumes().state(uid).await?;
    if ctx.json() {
        return format::print_json(&state);
    }
    style::field("UID", state.uid.color(palettes::ACCENT).to_string());
    style::field("Status", format::state_badge(state.status.as_str()));
    if !state.message.is_empty() {
        style::field("Message", &state.message);
    }
    style::field("Updated", format::relative_time(state.updated_at));
    Ok(())
}

async fn events(ctx: &Context, uid: &str, limit: u32) -> Result<()> {
    let page = Page {
        limit: Some(limit),
        cursor: None,
    };
    let events = ctx.client.volumes().events(uid, &page).await?;
    if ctx.json() {
        return format::print_json(&events);
    }
    if events.items.is_empty() {
        style::dim("no events");
        return Ok(());
    }
    let mut t = table::table(&["TIME", "TYPE", "STATUS", "REASON"]);
    for event in &events.items {
        t.add_row(vec![
            table::dim_cell(format::relative_time(event.created_at)),
            Cell::new(&event.event_type),
            table::state_cell(&event.new_status.clone().unwrap_or_default()),
            table::dim_cell(event.reason.clone().unwrap_or_default()),
        ]);
    }
    table::print(&t);
    table::summary(workload::plural(events.items.len(), "event"));
    Ok(())
}
