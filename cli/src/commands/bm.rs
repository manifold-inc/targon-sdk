use clap::{Args, Subcommand};
use colored::Colorize;
use comfy_table::Cell;

use crate::client::types::{
    BmConfigInput, CreateWorkloadRequest, ReinstallWorkloadRequest, WorkloadOperationResponse,
    WorkloadType,
};
use crate::commands::{self, workload, Context};
use crate::error::{CliError, Result};
use crate::output::{format, palettes, progress, prompt, style, table};

#[derive(Debug, Clone, Args)]
pub struct BmSpec {
    /// Server name
    #[arg(long)]
    pub name: Option<String>,
    /// OS image from the catalog (see `targon bm images`)
    #[arg(long)]
    pub image: Option<String>,
    /// Hardware class (see `targon inventory --type bm`)
    #[arg(long)]
    pub resource: Option<String>,
    /// Region the server is provisioned in (see `targon region`)
    #[arg(long)]
    pub region: Option<String>,
    /// SSH key (repeatable; at least one is required)
    #[arg(long = "ssh-key", value_name = "UID")]
    pub ssh_key: Vec<String>,
    /// RFC-1123 hostname set via cloud-init (default: the server name)
    #[arg(long)]
    pub hostname: Option<String>,
    /// Place on one of the organization's reservations instead of on-demand
    #[arg(long, value_name = "UID")]
    pub reservation: Option<String>,
    /// Project (default: active project from `project use`)
    #[arg(long)]
    pub project: Option<String>,
}

#[derive(Debug, Subcommand)]
pub enum BmCommands {
    /// Claim and provision a bare metal server
    #[command(
        override_usage = "targon bm deploy --name <NAME> --image <IMAGE> --resource <RESOURCE> --region <REGION> --ssh-key <UID> [OPTIONS]"
    )]
    Deploy {
        #[command(flatten)]
        spec: BmSpec,
        /// Skip confirmation
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Register a server without claiming hardware
    #[command(
        override_usage = "targon bm create --name <NAME> --image <IMAGE> --resource <RESOURCE> --region <REGION> --ssh-key <UID> [OPTIONS]"
    )]
    Create {
        #[command(flatten)]
        spec: BmSpec,
        /// Skip confirmation
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Claim hardware for a registered server
    Start { uid: String },
    /// Wipe the disks and install a fresh OS (destroys all data)
    Reinstall {
        uid: String,
        /// OS image to install (see `targon bm images`)
        #[arg(long)]
        image: Option<String>,
        /// SSH key (repeatable; default: the keys already attached)
        #[arg(long = "ssh-key", value_name = "UID")]
        ssh_key: Vec<String>,
        /// RFC-1123 hostname (default: the current hostname)
        #[arg(long)]
        hostname: Option<String>,
        /// Skip confirmation
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Power cycle the chassis out of band
    Reboot { uid: String },
    /// Power on the chassis
    #[command(name = "power-on")]
    PowerOn { uid: String },
    /// Power off the chassis (the server stays claimed and billed)
    #[command(name = "power-off")]
    PowerOff {
        uid: String,
        /// Skip confirmation
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// List bare metal servers
    List {
        /// Filter by status
        #[arg(long, visible_alias = "state")]
        status: Option<String>,
        /// Filter by project
        #[arg(long)]
        project: Option<String>,
        /// Filter by name
        #[arg(long)]
        name: Option<String>,
        /// Max results
        #[arg(long, default_value_t = 50)]
        limit: u32,
        /// Continue listing from this cursor
        #[arg(long)]
        cursor: Option<String>,
    },
    /// List installable OS images
    Images,
}

pub async fn handle(ctx: &Context, cmd: &BmCommands) -> Result<()> {
    match cmd {
        BmCommands::Deploy { spec, yes } => deploy(ctx, spec.clone(), *yes).await,
        BmCommands::Create { spec, yes } => create(ctx, spec.clone(), *yes).await,
        BmCommands::Start { uid } => start(ctx, uid).await,
        BmCommands::Reinstall {
            uid,
            image,
            ssh_key,
            hostname,
            yes,
        } => {
            reinstall(
                ctx,
                uid,
                image.clone(),
                ssh_key.clone(),
                hostname.clone(),
                *yes,
            )
            .await
        }
        BmCommands::Reboot { uid } => power(ctx, uid, PowerOp::Reboot).await,
        BmCommands::PowerOn { uid } => power(ctx, uid, PowerOp::On).await,
        BmCommands::PowerOff { uid, yes } => {
            if prompt::is_tty()
                && !yes
                && !prompt::confirm(
                    &format!("Power off {uid}? The server stays claimed and billed."),
                    false,
                )?
            {
                return Err(CliError::Cancelled);
            }
            power(ctx, uid, PowerOp::Off).await
        }
        BmCommands::List {
            status,
            project,
            name,
            limit,
            cursor,
        } => {
            workload::list(
                ctx,
                Some("BM".to_string()),
                status.clone(),
                project.clone(),
                name.clone(),
                *limit,
                cursor.clone(),
            )
            .await
        }
        BmCommands::Images => images(ctx).await,
    }
}

async fn deploy(ctx: &Context, spec: BmSpec, yes: bool) -> Result<()> {
    let req = build_request(ctx, spec, yes, "Deploy").await?;
    workload::deploy_flow(ctx, &req, workload::DeployKind::Bm).await
}

async fn create(ctx: &Context, spec: BmSpec, yes: bool) -> Result<()> {
    let req = build_request(ctx, spec, yes, "Create").await?;
    let created = workload::register(ctx, &req).await?;
    if ctx.json() {
        return format::print_json(&created);
    }
    style::next_action("start", format!("targon bm start {}", created.uid));
    Ok(())
}

async fn start(ctx: &Context, uid: &str) -> Result<()> {
    commands::ensure_bm(ctx, uid, "start").await?;
    workload::start_workload(ctx, uid).await
}

#[derive(Clone, Copy)]
enum PowerOp {
    Reboot,
    On,
    Off,
}

impl PowerOp {
    fn verb(self) -> &'static str {
        match self {
            PowerOp::Reboot => "reboot",
            PowerOp::On => "power-on",
            PowerOp::Off => "power-off",
        }
    }

    fn progress(self) -> &'static str {
        match self {
            PowerOp::Reboot => "Rebooting",
            PowerOp::On => "Powering on",
            PowerOp::Off => "Powering off",
        }
    }

    fn done(self) -> &'static str {
        match self {
            PowerOp::Reboot => "Reboot queued",
            PowerOp::On => "Power on queued",
            PowerOp::Off => "Power off queued",
        }
    }
}

async fn power(ctx: &Context, uid: &str, op: PowerOp) -> Result<()> {
    commands::ensure_bm(ctx, uid, op.verb()).await?;
    let spinner = progress::spinner_if(
        !ctx.json(),
        format!("{} {}…", op.progress(), format::short_uid(uid)),
    );
    let workloads = ctx.client.workloads(ctx.org()?);
    let result = match op {
        PowerOp::Reboot => workloads.reboot(uid).await,
        PowerOp::On => workloads.power_on(uid).await,
        PowerOp::Off => workloads.power_off(uid).await,
    };
    match result {
        Ok(workload) => {
            spinner.finish_ok(format!("{} {}", op.done(), workload.uid));
            finish(ctx, &workload)
        }
        Err(e) => {
            spinner.finish_fail(format!("{} failed", op.progress()));
            Err(e.into())
        }
    }
}

async fn reinstall(
    ctx: &Context,
    uid: &str,
    image: Option<String>,
    ssh_keys: Vec<String>,
    hostname: Option<String>,
    yes: bool,
) -> Result<()> {
    let workload = commands::ensure_bm(ctx, uid, "reinstall").await?;
    let image = match image {
        Some(image) => image,
        None => {
            prompt::require_tty("--image")?;
            select_image(ctx).await?
        }
    };
    let hostname = match hostname {
        Some(hostname) => Some(commands::validate_hostname(&hostname)?),
        None => None,
    };

    if prompt::is_tty() && !yes {
        let mut rows = vec![
            ("server", format!("{} ({uid})", workload.name)),
            ("image", image.clone()),
        ];
        if let Some(hostname) = &hostname {
            rows.push(("hostname", hostname.clone()));
        }
        if !ssh_keys.is_empty() {
            rows.push(("ssh keys", ssh_keys.join(", ")));
        }
        eprintln!();
        style::summary_box("reinstall", &rows);
        style::warn("every disk on this server is wiped — all data is destroyed");
        if !prompt::confirm("Reinstall this server?", false)? {
            return Err(CliError::Cancelled);
        }
    }

    let req = ReinstallWorkloadRequest {
        image,
        ssh_keys,
        hostname,
    };
    let spinner = progress::spinner_if(
        !ctx.json(),
        format!("Reinstalling {}…", format::short_uid(uid)),
    );
    match ctx.client.workloads(ctx.org()?).reinstall(uid, &req).await {
        Ok(workload) => {
            spinner.finish_ok(format!("Reinstall queued for {}", workload.uid));
            if ctx.json() {
                return format::print_json(&workload);
            }
            style::next_action("watch", format!("targon workload state {}", workload.uid));
            style::next_action("events", format!("targon workload events {}", workload.uid));
            Ok(())
        }
        Err(e) => {
            spinner.finish_fail("Reinstall failed");
            Err(e.into())
        }
    }
}

fn finish(ctx: &Context, workload: &WorkloadOperationResponse) -> Result<()> {
    if ctx.json() {
        return format::print_json(workload);
    }
    if let Some(state) = &workload.state {
        style::next_action("state", format::state_badge(state.status.as_str()));
    }
    style::next_action("watch", format!("targon workload state {}", workload.uid));
    Ok(())
}

async fn images(ctx: &Context) -> Result<()> {
    let images = ctx.client.workloads(ctx.org()?).bm_images().await?;
    if ctx.json() {
        return format::print_json(&images);
    }
    if images.is_empty() {
        style::dim("no bm images available");
        return Ok(());
    }
    let mut t = table::table(&["NAME", "DISPLAY", "OS", "DESCRIPTION"]);
    for image in &images {
        t.add_row(vec![
            Cell::new(&image.name),
            Cell::new(&image.display_name),
            table::dim_cell(image.os_family.clone()),
            table::dim_cell(image.description.clone()),
        ]);
    }
    table::print(&t);
    table::summary(workload::plural(images.len(), "image"));
    Ok(())
}

async fn build_request(
    ctx: &Context,
    spec: BmSpec,
    yes: bool,
    verb: &str,
) -> Result<CreateWorkloadRequest> {
    let name = match spec.name {
        Some(name) => name,
        None => {
            prompt::require_tty("--name")?;
            prompt::input("Server name")?
        }
    };
    let region = match spec.region {
        Some(region) => region,
        None => {
            prompt::require_tty("--region")?;
            select_region(ctx).await?
        }
    };
    let image = match spec.image {
        Some(image) => image,
        None => {
            prompt::require_tty("--image")?;
            select_image(ctx).await?
        }
    };
    let resource_name = match spec.resource {
        Some(resource) => resource,
        None => {
            prompt::require_tty("--resource")?;
            commands::select_resource_in_region(ctx, "bm", Some(&region)).await?
        }
    };
    let ssh_keys = if !spec.ssh_key.is_empty() {
        spec.ssh_key
    } else if prompt::is_tty() {
        commands::select_ssh_keys_required(ctx).await?
    } else {
        return Err(CliError::NotInteractive("--ssh-key".to_string()));
    };
    let hostname = match spec.hostname {
        Some(hostname) => Some(commands::validate_hostname(&hostname)?),
        None => None,
    };

    let mut req = CreateWorkloadRequest::new(WorkloadType::Bm, name, image, resource_name);
    req.project_id = ctx.project(spec.project);
    req.ssh_keys = ssh_keys;
    req.region = Some(region.clone());
    req.reservation_uid = spec.reservation;
    if hostname.is_some() {
        req.bm_config = Some(BmConfigInput { hostname });
    }

    if prompt::is_tty() && !yes {
        let pricing =
            commands::resource_pricing_in_region(ctx, &req.resource_name, "bm", Some(&region))
                .await;
        let mut rows = vec![
            ("name", req.name.clone()),
            ("image", req.image.clone()),
            (
                "resource",
                commands::resource_value(&req.resource_name, pricing.as_ref()),
            ),
            ("region", region.clone()),
        ];
        if let Some(hostname) = req.bm_config.as_ref().and_then(|c| c.hostname.clone()) {
            rows.push(("hostname", hostname));
        }
        if let Some(reservation) = &req.reservation_uid {
            rows.push(("reservation", reservation.clone()));
        } else {
            rows.push(("placement", "on-demand".color(palettes::DIM).to_string()));
        }
        if let Some(project) = &req.project_id {
            rows.push(("project", project.clone()));
        }
        rows.push(("ssh keys", req.ssh_keys.join(", ")));
        eprintln!();
        style::summary_box("bare metal", &rows);
        let confirm_msg = match &pricing {
            Some(p) => format!("{verb} for {}?", format::cost_per_hour(p.cost_per_hour)),
            None => format!("{verb} this server?"),
        };
        if !prompt::confirm(&confirm_msg, false)? {
            return Err(CliError::Cancelled);
        }
    }
    Ok(req)
}

async fn select_image(ctx: &Context) -> Result<String> {
    let images = ctx.client.workloads(ctx.org()?).bm_images().await?;
    if images.is_empty() {
        return Err(CliError::Config("no bm images available".to_string()));
    }
    let labels: Vec<String> = images
        .iter()
        .map(|i| format!("{:<28} {}", i.display_name, i.description))
        .collect();
    let idx = prompt::select("Select an image", &labels)?;
    Ok(images[idx].name.clone())
}

async fn select_region(ctx: &Context) -> Result<String> {
    let regions = ctx.client.inventory().regions().await?;
    if regions.is_empty() {
        return Err(CliError::Config("no active regions".to_string()));
    }
    let labels: Vec<String> = regions
        .iter()
        .map(|r| format!("{:<20} {:<4} {}", r.display_name, r.country, r.continent))
        .collect();
    let idx = prompt::select("Select a region", &labels)?;
    Ok(regions[idx].uid.clone())
}
