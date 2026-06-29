use std::io::Write;

use clap::{Args, Subcommand};
use comfy_table::Cell;
use futures_util::{pin_mut, StreamExt};

use crate::client::pagination::Page;
use crate::client::types::{
    AttachVolumeRequest, CreateWorkloadRequest, EnvVar, ListWorkloadsParams, LogOptions, Port,
    PortProtocol, PortRouting, RegistryAuth, VolumeMount, Workload,
};
use crate::client::ClientError;
use crate::commands::{self, Context};
use crate::error::{CliError, Result};
use crate::output::{format, progress, prompt, style, table};

#[derive(Debug, Clone, Args)]
pub struct WorkloadSpec {
    #[arg(long)]
    pub name: Option<String>,
    #[arg(long)]
    pub image: Option<String>,
    #[arg(long)]
    pub resource: Option<String>,
    #[arg(long = "env", value_name = "KEY=VAL")]
    pub env: Vec<String>,
    #[arg(long = "port", value_name = "PORT[/PROTO[/ROUTING]]")]
    pub port: Vec<String>,
    #[arg(long = "volume", value_name = "UID:/path[:ro]")]
    pub volume: Vec<String>,
    #[arg(long = "ssh-key", value_name = "UID")]
    pub ssh_key: Vec<String>,
    #[arg(long = "command")]
    pub command: Vec<String>,
    #[arg(long = "arg")]
    pub arg: Vec<String>,
    #[arg(long)]
    pub project: Option<String>,
    #[arg(long = "registry-server")]
    pub registry_server: Option<String>,
    #[arg(long = "registry-user")]
    pub registry_user: Option<String>,
    #[arg(long = "registry-pass")]
    pub registry_pass: Option<String>,
}

#[derive(Debug, Subcommand)]
pub enum WorkloadCommands {
    /// Register and start a workload, or start an existing one by UID
    Deploy {
        uid: Option<String>,
        #[command(flatten)]
        spec: WorkloadSpec,
        /// Register without starting
        #[arg(long = "no-start")]
        no_start: bool,
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Register a workload without starting it
    Create {
        #[command(flatten)]
        spec: WorkloadSpec,
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// List workloads
    List {
        #[arg(long)]
        status: Option<String>,
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        name: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: u32,
    },
    /// Show a workload
    Get { uid: String },
    /// Delete a workload
    Delete {
        uid: String,
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Show workload logs
    Logs {
        uid: String,
        #[arg(long)]
        since: Option<String>,
        #[arg(long)]
        tail: Option<u32>,
        #[arg(long)]
        previous: bool,
        #[arg(long, short = 'f')]
        follow: bool,
    },
    /// Show workload events
    Events {
        uid: String,
        #[arg(long, default_value_t = 20)]
        limit: u32,
    },
    /// Run a command inside a workload
    Exec {
        uid: String,
        #[arg(
            trailing_var_arg = true,
            allow_hyphen_values = true,
            required = true,
            value_name = "COMMAND"
        )]
        command: Vec<String>,
    },
    /// Attach a volume to a workload
    AttachVolume {
        uid: String,
        volume_uid: String,
        #[arg(long = "mount-path")]
        mount_path: String,
        #[arg(long = "read-only")]
        read_only: bool,
    },
    /// Detach a volume from a workload
    DetachVolume { uid: String, volume_uid: String },
    /// Attach an SSH key to a workload
    AttachSshKey { uid: String, ssh_key_uid: String },
    /// Detach an SSH key from a workload
    DetachSshKey { uid: String, ssh_key_uid: String },
    /// Suspend a workload
    Suspend { uid: String },
    /// Reboot a workload
    Reboot { uid: String },
}

pub async fn handle(ctx: &Context, cmd: &WorkloadCommands) -> Result<()> {
    match cmd {
        WorkloadCommands::Deploy {
            uid,
            spec,
            no_start,
            yes,
        } => deploy(ctx, uid.clone(), spec.clone(), *no_start, *yes).await,
        WorkloadCommands::Create { spec, yes } => create(ctx, spec.clone(), *yes).await,
        WorkloadCommands::List {
            status,
            project,
            name,
            limit,
        } => list(ctx, status.clone(), project.clone(), name.clone(), *limit).await,
        WorkloadCommands::Get { uid } => get(ctx, uid).await,
        WorkloadCommands::Delete { uid, yes } => delete(ctx, uid, *yes).await,
        WorkloadCommands::Logs {
            uid,
            since,
            tail,
            previous,
            follow,
        } => logs(ctx, uid, since.clone(), *tail, *previous, *follow).await,
        WorkloadCommands::Events { uid, limit } => events(ctx, uid, *limit).await,
        WorkloadCommands::Exec { uid, command } => exec(ctx, uid, command).await,
        WorkloadCommands::AttachVolume {
            uid,
            volume_uid,
            mount_path,
            read_only,
        } => attach_volume(ctx, uid, volume_uid, mount_path.clone(), *read_only).await,
        WorkloadCommands::DetachVolume { uid, volume_uid } => {
            detach_volume(ctx, uid, volume_uid).await
        }
        WorkloadCommands::AttachSshKey { uid, ssh_key_uid } => {
            attach_ssh_key(ctx, uid, ssh_key_uid).await
        }
        WorkloadCommands::DetachSshKey { uid, ssh_key_uid } => {
            detach_ssh_key(ctx, uid, ssh_key_uid).await
        }
        WorkloadCommands::Suspend { uid } => suspend(ctx, uid).await,
        WorkloadCommands::Reboot { uid } => reboot(ctx, uid).await,
    }
}

async fn deploy(
    ctx: &Context,
    uid: Option<String>,
    spec: WorkloadSpec,
    no_start: bool,
    yes: bool,
) -> Result<()> {
    if let Some(uid) = uid {
        return deploy_existing(ctx, &uid).await;
    }
    let req = build_request(ctx, spec, yes).await?;
    let workload = register(ctx, &req).await?;
    if no_start {
        if ctx.json() {
            return format::print_json(&workload);
        }
        style::info(format!("registered without starting: {}", workload.uid));
        return Ok(());
    }
    deploy_existing(ctx, &workload.uid).await
}

async fn create(ctx: &Context, spec: WorkloadSpec, yes: bool) -> Result<()> {
    let req = build_request(ctx, spec, yes).await?;
    let workload = register(ctx, &req).await?;
    if ctx.json() {
        return format::print_json(&workload);
    }
    Ok(())
}

async fn build_request(
    ctx: &Context,
    spec: WorkloadSpec,
    yes: bool,
) -> Result<CreateWorkloadRequest> {
    let name = match spec.name {
        Some(name) => name,
        None => {
            prompt::require_tty("--name")?;
            prompt::input("Workload name")?
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

    let mut req = CreateWorkloadRequest::new(name, image, resource_name);
    req.project_id = spec.project;
    for raw in &spec.env {
        req.envs.push(parse_env(raw)?);
    }
    for raw in &spec.port {
        req.ports.push(parse_port(raw)?);
    }
    for raw in &spec.volume {
        req.volumes.push(parse_volume(raw)?);
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
        print_summary(&req);
        if !prompt::confirm("Deploy this workload?", true)? {
            return Err(CliError::Cancelled);
        }
    }
    Ok(req)
}

fn print_summary(req: &CreateWorkloadRequest) {
    style::field("Name", &req.name);
    style::field("Image", &req.image);
    style::field("Resource", &req.resource_name);
    if let Some(project) = &req.project_id {
        style::field("Project", project);
    }
    if !req.ports.is_empty() {
        let ports = req
            .ports
            .iter()
            .map(|p| p.port.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        style::field("Ports", ports);
    }
    if !req.volumes.is_empty() {
        style::field("Volumes", req.volumes.len().to_string());
    }
}

async fn register(ctx: &Context, req: &CreateWorkloadRequest) -> Result<Workload> {
    let spinner = progress::spinner(format!("Registering {}…", req.name));
    match ctx.client.workloads().create(req).await {
        Ok(workload) => {
            spinner.finish_ok(format!("Registered {} ({})", workload.name, workload.uid));
            Ok(workload)
        }
        Err(e) => {
            spinner.finish_fail("Registration failed");
            Err(e.into())
        }
    }
}

async fn deploy_existing(ctx: &Context, uid: &str) -> Result<()> {
    let spinner = progress::spinner(format!("Deploying {}…", format::short_uid(uid)));
    match ctx.client.workloads().deploy(uid).await {
        Ok(workload) => {
            let status = workload
                .state
                .as_ref()
                .map(|s| s.status.to_string())
                .unwrap_or_else(|| "provisioning".to_string());
            spinner.finish_ok(format!("Deployed {} ({status})", workload.uid));
            if ctx.json() {
                return format::print_json(&workload);
            }
            Ok(())
        }
        Err(e) => {
            spinner.finish_fail("Deploy failed");
            Err(e.into())
        }
    }
}

async fn list(
    ctx: &Context,
    status: Option<String>,
    project: Option<String>,
    name: Option<String>,
    limit: u32,
) -> Result<()> {
    let params = ListWorkloadsParams {
        page: Page {
            limit: Some(limit),
            cursor: None,
        },
        status,
        project_id: project,
        name,
    };
    let workloads = ctx.client.workloads().list(&params).await?;
    if ctx.json() {
        return format::print_json(&workloads);
    }
    if workloads.items.is_empty() {
        style::dim("no workloads");
        return Ok(());
    }
    let mut t = table::table(&["UID", "NAME", "IMAGE", "RESOURCE", "STATE", "COST", "AGE"]);
    for workload in &workloads.items {
        let status = workload
            .state
            .as_ref()
            .map(|s| s.status.to_string())
            .unwrap_or_default();
        let resource = workload
            .resource
            .as_ref()
            .map(|r| r.display_name.clone())
            .unwrap_or_default();
        let cost = workload
            .cost_per_hour
            .map(format::cost_per_hour)
            .unwrap_or_else(|| "-".to_string());
        t.add_row(vec![
            Cell::new(&workload.uid),
            Cell::new(&workload.name),
            Cell::new(workload.image.clone().unwrap_or_default()),
            Cell::new(resource),
            table::state_cell(&status),
            table::dim_cell(cost),
            table::dim_cell(format::relative_time(workload.created_at)),
        ]);
    }
    println!("{t}");
    Ok(())
}

async fn get(ctx: &Context, uid: &str) -> Result<()> {
    let workload = ctx.client.workloads().get(uid).await?;
    if ctx.json() {
        return format::print_json(&workload);
    }
    style::field("UID", &workload.uid);
    style::field("Name", &workload.name);
    if let Some(image) = &workload.image {
        style::field("Image", image);
    }
    if let Some(resource) = &workload.resource {
        style::field("Resource", &resource.display_name);
        style::field(
            "GPU",
            format::gpu_spec(resource.gpu_type.as_deref(), resource.gpu_count.unwrap_or(0)),
        );
    }
    if let Some(state) = &workload.state {
        style::field("State", format::state_badge(state.status.as_str()));
        if !state.message.is_empty() {
            style::field("Message", &state.message);
        }
        if let Some(ip) = &state.public_ip {
            style::field("Public IP", ip);
        }
        if let Some(port) = state.ssh_port {
            style::field("SSH port", port.to_string());
        }
        for url in &state.urls {
            style::field("URL", format!("{} {} {}", url.port, style::ARROW, url.url));
        }
    }
    if let Some(cost) = workload.cost_per_hour {
        style::field("Cost", format::cost_per_hour(cost));
    }
    if let Some(project) = &workload.project_id {
        style::field("Project", project);
    }
    for volume in &workload.volumes {
        style::field("Volume", format!("{} {} {}", volume.uid, style::ARROW, volume.mount_path));
    }
    for key in &workload.ssh_keys {
        style::field("SSH key", format!("{} ({})", key.name, key.uid));
    }
    style::field("Created", format::relative_time(workload.created_at));
    Ok(())
}

async fn delete(ctx: &Context, uid: &str, yes: bool) -> Result<()> {
    if prompt::is_tty() && !yes && !prompt::confirm(&format!("Delete workload {uid}?"), false)? {
        return Err(CliError::Cancelled);
    }
    ctx.client.workloads().delete(uid).await?;
    style::success(format!("deleted workload {uid}"));
    Ok(())
}

async fn logs(
    ctx: &Context,
    uid: &str,
    since: Option<String>,
    tail: Option<u32>,
    previous: bool,
    follow: bool,
) -> Result<()> {
    let opts = LogOptions {
        since,
        tail,
        previous,
    };
    if follow {
        let stream = ctx.client.workloads().logs_stream(uid, &opts).await?;
        pin_mut!(stream);
        let mut stdout = std::io::stdout();
        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(ClientError::from)?;
            stdout.write_all(&bytes)?;
            stdout.flush()?;
        }
    } else {
        let text = ctx.client.workloads().logs(uid, &opts).await?;
        print!("{text}");
    }
    Ok(())
}

async fn events(ctx: &Context, uid: &str, limit: u32) -> Result<()> {
    let page = Page {
        limit: Some(limit),
        cursor: None,
    };
    let events = ctx.client.workloads().events(uid, &page).await?;
    if ctx.json() {
        return format::print_json(&events);
    }
    if events.items.is_empty() {
        style::dim("no events");
        return Ok(());
    }
    let mut t = table::table(&["TIME", "TYPE", "STATUS", "MESSAGE"]);
    for event in &events.items {
        let message = event
            .display_message
            .clone()
            .or_else(|| event.message.clone())
            .unwrap_or_default();
        t.add_row(vec![
            table::dim_cell(format::relative_time(event.created_at)),
            Cell::new(&event.event_type),
            Cell::new(event.new_status.clone().unwrap_or_default()),
            Cell::new(message),
        ]);
    }
    println!("{t}");
    Ok(())
}

async fn exec(ctx: &Context, uid: &str, command: &[String]) -> Result<()> {
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

async fn attach_volume(
    ctx: &Context,
    uid: &str,
    volume_uid: &str,
    mount_path: String,
    read_only: bool,
) -> Result<()> {
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
    ctx.client.workloads().detach_volume(uid, volume_uid).await?;
    style::success(format!("detached volume {volume_uid} from {uid}"));
    Ok(())
}

async fn attach_ssh_key(ctx: &Context, uid: &str, ssh_key_uid: &str) -> Result<()> {
    let result = ctx.client.workloads().attach_ssh_key(uid, ssh_key_uid).await?;
    if ctx.json() {
        return format::print_json(&result);
    }
    style::success(format!("attached ssh key {ssh_key_uid} to {uid}"));
    Ok(())
}

async fn detach_ssh_key(ctx: &Context, uid: &str, ssh_key_uid: &str) -> Result<()> {
    ctx.client
        .workloads()
        .detach_ssh_key(uid, ssh_key_uid)
        .await?;
    style::success(format!("detached ssh key {ssh_key_uid} from {uid}"));
    Ok(())
}

async fn suspend(ctx: &Context, uid: &str) -> Result<()> {
    let spinner = progress::spinner(format!("Suspending {}…", format::short_uid(uid)));
    match ctx.client.workloads().suspend(uid).await {
        Ok(workload) => {
            spinner.finish_ok(format!("Suspended {}", workload.uid));
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

async fn reboot(ctx: &Context, uid: &str) -> Result<()> {
    let spinner = progress::spinner(format!("Rebooting {}…", format::short_uid(uid)));
    match ctx.client.workloads().reboot(uid).await {
        Ok(workload) => {
            spinner.finish_ok(format!("Rebooted {}", workload.uid));
            if ctx.json() {
                return format::print_json(&workload);
            }
            Ok(())
        }
        Err(e) => {
            spinner.finish_fail("Reboot failed");
            Err(e.into())
        }
    }
}

fn parse_env(raw: &str) -> Result<EnvVar> {
    let (name, value) = raw
        .split_once('=')
        .ok_or_else(|| CliError::Config(format!("invalid --env '{raw}', expected KEY=VAL")))?;
    Ok(EnvVar {
        name: name.to_string(),
        value: value.to_string(),
    })
}

fn parse_port(raw: &str) -> Result<Port> {
    let mut parts = raw.split('/');
    let port: u16 = parts
        .next()
        .unwrap_or_default()
        .parse()
        .map_err(|_| CliError::Config(format!("invalid --port '{raw}'")))?;
    let protocol = match parts.next() {
        None => PortProtocol::default(),
        Some(p) => match p.to_ascii_uppercase().as_str() {
            "TCP" => PortProtocol::Tcp,
            "UDP" => PortProtocol::Udp,
            "SCTP" => PortProtocol::Sctp,
            other => return Err(CliError::Config(format!("invalid port protocol '{other}'"))),
        },
    };
    let routing = match parts.next() {
        None => PortRouting::default(),
        Some(r) => match r.to_ascii_uppercase().as_str() {
            "PROXIED" => PortRouting::Proxied,
            "DIRECT" => PortRouting::Direct,
            other => return Err(CliError::Config(format!("invalid port routing '{other}'"))),
        },
    };
    Ok(Port {
        port,
        protocol,
        routing,
    })
}

fn parse_volume(raw: &str) -> Result<VolumeMount> {
    let (uid, rest) = raw
        .split_once(':')
        .ok_or_else(|| CliError::Config(format!("invalid --volume '{raw}', expected UID:/path[:ro]")))?;
    let (mount_path, read_only) = match rest.rsplit_once(':') {
        Some((path, "ro")) => (path.to_string(), true),
        Some((path, "rw")) => (path.to_string(), false),
        _ => (rest.to_string(), false),
    };
    if uid.is_empty() || mount_path.is_empty() {
        return Err(CliError::Config(format!("invalid --volume '{raw}'")));
    }
    Ok(VolumeMount {
        uid: uid.to_string(),
        name: None,
        mount_path,
        read_only,
        last_backup_at: None,
    })
}
