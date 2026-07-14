use std::io::Write;

use clap::{Args, Subcommand};
use colored::Colorize;
use futures_util::{pin_mut, StreamExt};

use crate::client::types::{
    AttachVolumeRequest, CreateWorkloadRequest, RegistryAuth, WorkloadType,
};
use crate::client::ClientError;
use crate::commands::{self, workload, Context};
use crate::error::{CliError, Result};
use crate::output::{format, palettes, prompt, style};

#[derive(Debug, Clone, Args)]
pub struct RentalSpec {
    /// Rental name
    #[arg(long)]
    pub name: Option<String>,
    /// Container image reference
    #[arg(long)]
    pub image: Option<String>,
    /// Resource SKU (see `targon inventory list`)
    #[arg(long)]
    pub resource: Option<String>,
    /// Environment variable (repeatable)
    #[arg(long = "env", value_name = "KEY=VAL")]
    pub env: Vec<String>,
    /// Port to expose (repeatable, comma-separated ok)
    #[arg(long = "port", value_name = "PORT[/PROTO[/ROUTING]]", value_delimiter = ',')]
    pub port: Vec<String>,
    /// Volume mount (repeatable)
    #[arg(long = "volume", value_name = "UID:/path[:ro]")]
    pub volume: Vec<String>,
    /// SSH key to attach (repeatable)
    #[arg(long = "ssh-key", value_name = "UID")]
    pub ssh_key: Vec<String>,
    /// Container command (repeatable)
    #[arg(long = "command")]
    pub command: Vec<String>,
    /// Container argument (repeatable)
    #[arg(long = "arg")]
    pub arg: Vec<String>,
    /// Project (default: active project from `project use`)
    #[arg(long)]
    pub project: Option<String>,
    /// Private registry server
    #[arg(long = "registry-server")]
    pub registry_server: Option<String>,
    /// Private registry username
    #[arg(long = "registry-user")]
    pub registry_user: Option<String>,
    /// Private registry password
    #[arg(long = "registry-pass")]
    pub registry_pass: Option<String>,
}

#[derive(Debug, Subcommand)]
pub enum RentalCommands {
    /// Register and start a rental
    #[command(override_usage = "targon rental deploy --name <NAME> --image <IMAGE> --resource <RESOURCE> [OPTIONS]")]
    Deploy {
        #[command(flatten)]
        spec: RentalSpec,
        /// Skip confirmation
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Register a rental without starting it
    #[command(override_usage = "targon rental create --name <NAME> --image <IMAGE> --resource <RESOURCE> [OPTIONS]")]
    Create {
        #[command(flatten)]
        spec: RentalSpec,
        /// Skip confirmation
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Start a registered or suspended rental
    Start { uid: String },
    /// List rentals
    List {
        /// Filter by state
        #[arg(long)]
        state: Option<String>,
        /// Filter by project
        #[arg(long)]
        project: Option<String>,
        /// Filter by name
        #[arg(long)]
        name: Option<String>,
        /// Max results
        #[arg(long, default_value_t = 50)]
        limit: u32,
    },
    /// Run a command inside a rental
    Exec {
        uid: String,
        /// Command to run
        #[arg(last = true, required = true)]
        command: Vec<String>,
    },
    /// Suspend a rental
    Suspend { uid: String },
    /// Attach a volume
    AttachVolume {
        uid: String,
        volume_uid: String,
        #[arg(long = "mount-path")]
        mount_path: String,
        #[arg(long = "read-only")]
        read_only: bool,
    },
    /// Detach a volume
    DetachVolume { uid: String, volume_uid: String },
    /// Attach an SSH key
    AttachSshKey { uid: String, ssh_key_uid: String },
    /// Detach an SSH key
    DetachSshKey { uid: String, ssh_key_uid: String },
}

pub async fn handle(ctx: &Context, cmd: &RentalCommands) -> Result<()> {
    match cmd {
        RentalCommands::Deploy { spec, yes } => deploy(ctx, spec.clone(), *yes).await,
        RentalCommands::Create { spec, yes } => create(ctx, spec.clone(), *yes).await,
        RentalCommands::Start { uid } => start(ctx, uid).await,
        RentalCommands::List {
            state,
            project,
            name,
            limit,
        } => {
            workload::list(
                ctx,
                Some("RENTAL".to_string()),
                state.clone(),
                project.clone(),
                name.clone(),
                *limit,
            )
            .await
        }
        RentalCommands::Exec { uid, command } => exec(ctx, uid, command).await,
        RentalCommands::Suspend { uid } => suspend(ctx, uid).await,
        RentalCommands::AttachVolume {
            uid,
            volume_uid,
            mount_path,
            read_only,
        } => attach_volume(ctx, uid, volume_uid, mount_path.clone(), *read_only).await,
        RentalCommands::DetachVolume { uid, volume_uid } => {
            detach_volume(ctx, uid, volume_uid).await
        }
        RentalCommands::AttachSshKey { uid, ssh_key_uid } => {
            attach_ssh_key(ctx, uid, ssh_key_uid).await
        }
        RentalCommands::DetachSshKey { uid, ssh_key_uid } => {
            detach_ssh_key(ctx, uid, ssh_key_uid).await
        }
    }
}

async fn deploy(ctx: &Context, spec: RentalSpec, yes: bool) -> Result<()> {
    let req = build_request(ctx, spec, yes, "Deploy").await?;
    workload::deploy_flow(ctx, &req, workload::DeployKind::Rental).await
}

async fn create(ctx: &Context, spec: RentalSpec, yes: bool) -> Result<()> {
    let req = build_request(ctx, spec, yes, "Create").await?;
    let created = workload::register(ctx, &req).await?;
    if ctx.json() {
        return format::print_json(&created);
    }
    style::next_action("start", format!("targon rental start {}", created.uid));
    Ok(())
}

async fn start(ctx: &Context, uid: &str) -> Result<()> {
    commands::ensure_rental(ctx, uid, "start").await?;
    workload::start_workload(ctx, uid).await
}

async fn exec(ctx: &Context, uid: &str, command: &[String]) -> Result<()> {
    commands::ensure_rental(ctx, uid, "exec").await?;
    let stream = ctx.client.workloads().exec(uid, command).await?;
    pin_mut!(stream);
    let mut stdout = std::io::stdout();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(ClientError::from)?;
        stdout.write_all(&bytes)?;
        stdout.flush()?;
    }
    Ok(())
}

async fn suspend(ctx: &Context, uid: &str) -> Result<()> {
    commands::ensure_rental(ctx, uid, "suspend").await?;
    let spinner = crate::output::progress::spinner_if(
        !ctx.json(),
        format!("Suspending {}…", format::short_uid(uid)),
    );
    match ctx.client.workloads().suspend(uid).await {
        Ok(workload) => {
            spinner.finish_ok(format!(
                "Suspended {} (resume with `targon rental start {}`)",
                workload.uid, workload.uid
            ));
            if ctx.json() {
                return format::print_json(&workload);
            }
            Ok(())
        }
        Err(e) => {
            spinner.finish_fail("Suspend failed");
            Err(e.into())
        }
    }
}

async fn attach_volume(
    ctx: &Context,
    uid: &str,
    volume_uid: &str,
    mount_path: String,
    read_only: bool,
) -> Result<()> {
    commands::ensure_rental(ctx, uid, "attach-volume").await?;
    let req = AttachVolumeRequest {
        mount_path,
        read_only,
    };
    let result = ctx
        .client
        .workloads()
        .attach_volume(uid, volume_uid, &req)
        .await?;
    if ctx.json() {
        return format::print_json(&result);
    }
    style::success(format!(
        "attached volume {volume_uid} to {uid} at {}",
        result.mount_path
    ));
    Ok(())
}

async fn detach_volume(ctx: &Context, uid: &str, volume_uid: &str) -> Result<()> {
    commands::ensure_rental(ctx, uid, "detach-volume").await?;
    ctx.client.workloads().detach_volume(uid, volume_uid).await?;
    style::success(format!("detached volume {volume_uid} from {uid}"));
    Ok(())
}

async fn attach_ssh_key(ctx: &Context, uid: &str, ssh_key_uid: &str) -> Result<()> {
    commands::ensure_rental(ctx, uid, "attach-ssh-key").await?;
    let result = ctx.client.workloads().attach_ssh_key(uid, ssh_key_uid).await?;
    if ctx.json() {
        return format::print_json(&result);
    }
    style::success(format!("attached ssh key {ssh_key_uid} to {uid}"));
    Ok(())
}

async fn detach_ssh_key(ctx: &Context, uid: &str, ssh_key_uid: &str) -> Result<()> {
    commands::ensure_rental(ctx, uid, "detach-ssh-key").await?;
    ctx.client
        .workloads()
        .detach_ssh_key(uid, ssh_key_uid)
        .await?;
    style::success(format!("detached ssh key {ssh_key_uid} from {uid}"));
    Ok(())
}

async fn build_request(
    ctx: &Context,
    spec: RentalSpec,
    yes: bool,
    verb: &str,
) -> Result<CreateWorkloadRequest> {
    let name = match spec.name {
        Some(name) => name,
        None => {
            prompt::require_tty("--name")?;
            prompt::input("Rental name")?
        }
    };
    let image = match spec.image {
        Some(image) => image,
        None => {
            prompt::require_tty("--image")?;
            prompt::input("Container image")?
        }
    };
    let resource_name = match spec.resource {
        Some(resource) => resource,
        None => {
            prompt::require_tty("--resource")?;
            commands::select_resource(ctx).await?
        }
    };

    let ports = if spec.port.is_empty() && prompt::is_tty() {
        commands::prompt_list("Ports to expose (comma-separated, blank for none)")?
    } else {
        spec.port
    };
    let envs = if spec.env.is_empty() && prompt::is_tty() {
        commands::prompt_list("Env vars KEY=VAL (comma-separated, blank for none)")?
    } else {
        spec.env
    };

    let mut req = CreateWorkloadRequest::new(WorkloadType::Rental, name, image, resource_name);
    req.project_id = ctx.project(spec.project);
    for raw in &envs {
        req.envs.push(workload::parse_env(raw)?);
    }
    for raw in &ports {
        req.ports.push(workload::parse_port(raw)?);
    }
    for raw in &spec.volume {
        req.volumes.push(workload::parse_volume(raw)?);
    }
    req.ssh_keys = spec.ssh_key;
    req.commands = spec.command;
    req.args = spec.arg;
    if let Some(server) = spec.registry_server {
        req.registry_auth = Some(RegistryAuth {
            server,
            username: spec.registry_user.unwrap_or_default(),
            password: spec.registry_pass.unwrap_or_default(),
        });
    }

    if prompt::is_tty() && !yes {
        let pricing = commands::resource_pricing(ctx, &req.resource_name).await;
        eprintln!();
        style::summary_box("rental", &summary_rows(&req, pricing.as_ref()));
        let confirm_msg = match &pricing {
            Some(p) => format!("{verb} for {}?", format::cost_per_hour(p.cost_per_hour)),
            None => format!("{verb} this rental?"),
        };
        if !prompt::confirm(&confirm_msg, false)? {
            return Err(CliError::Cancelled);
        }
    }
    Ok(req)
}

fn summary_rows(
    req: &CreateWorkloadRequest,
    pricing: Option<&crate::client::types::Inventory>,
) -> Vec<(&'static str, String)> {
    let mut rows = vec![
        ("name", req.name.clone()),
        ("image", req.image.clone()),
        ("resource", resource_value(&req.resource_name, pricing)),
    ];
    if let Some(project) = &req.project_id {
        rows.push(("project", project.clone()));
    }
    if !req.ports.is_empty() {
        let ports = req
            .ports
            .iter()
            .map(|p| p.port.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        rows.push(("ports", ports));
    }
    if !req.envs.is_empty() {
        let names = req
            .envs
            .iter()
            .map(|e| e.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        rows.push(("env", format!("{names} ({})", req.envs.len())));
    }
    if !req.volumes.is_empty() {
        rows.push(("volumes", req.volumes.len().to_string()));
    }
    if !req.ssh_keys.is_empty() {
        rows.push(("ssh keys", req.ssh_keys.len().to_string()));
    }
    rows
}

pub(crate) fn resource_value(
    resource_name: &str,
    pricing: Option<&crate::client::types::Inventory>,
) -> String {
    match pricing {
        Some(p) => format!(
            "{} {} {}",
            p.display_name,
            style::SEP,
            format::cost_per_hour(p.cost_per_hour)
                .color(palettes::SUCCESS)
        ),
        None => resource_name.to_string(),
    }
}
