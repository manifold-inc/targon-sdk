use std::io::Write;
use std::time::Instant;

use clap::{Subcommand, ValueEnum};
use colored::Colorize;
use comfy_table::Cell;
use futures_util::{pin_mut, StreamExt};

use crate::client::pagination::Page;
use crate::client::types::{
    CreateWorkloadRequest, EnvVar, ListWorkloadsParams, LogOptions, Port, PortProtocol,
    PortRouting, VolumeMount, Workload,
};
use crate::client::ClientError;
use crate::commands::Context;
use crate::error::{CliError, Result};
use crate::output::{format, palettes, progress, style, table};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TypeFilter {
    Rental,
    Vm,
}

impl TypeFilter {
    pub fn api_value(self) -> String {
        match self {
            TypeFilter::Rental => "RENTAL".to_string(),
            TypeFilter::Vm => "VM".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LogType {
    Serial,
    Qemu,
}

impl LogType {
    fn api_value(self) -> String {
        match self {
            LogType::Serial => "serial".to_string(),
            LogType::Qemu => "qemu".to_string(),
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum WorkloadCommands {
    /// List workloads
    List {
        /// Filter by workload type
        #[arg(long = "type", value_enum)]
        workload_type: Option<TypeFilter>,
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
    /// Show a workload
    Get { uid: String },
    /// Print a workload's state
    State { uid: String },
    /// Delete a workload
    Delete {
        uid: String,
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Show workload logs
    Logs {
        uid: String,
        /// Stream logs
        #[arg(long, short = 'f')]
        follow: bool,
        /// Show last N lines
        #[arg(long, value_name = "N")]
        tail: Option<u32>,
        /// Logs since duration/timestamp
        #[arg(long)]
        since: Option<String>,
        /// Logs from the previous run
        #[arg(long)]
        previous: bool,
        /// VM log stream
        #[arg(long = "log-type", value_enum)]
        log_type: Option<LogType>,
    },
    /// Show workload events
    Events {
        uid: String,
        #[arg(long, default_value_t = 20)]
        limit: u32,
    },
}

pub async fn handle(ctx: &Context, cmd: &WorkloadCommands) -> Result<()> {
    match cmd {
        WorkloadCommands::List {
            workload_type,
            state,
            project,
            name,
            limit,
        } => {
            list(
                ctx,
                workload_type.map(TypeFilter::api_value),
                state.clone(),
                project.clone(),
                name.clone(),
                *limit,
            )
            .await
        }
        WorkloadCommands::Get { uid } => get(ctx, uid).await,
        WorkloadCommands::State { uid } => state(ctx, uid).await,
        WorkloadCommands::Delete { uid, yes } => delete(ctx, uid, *yes).await,
        WorkloadCommands::Logs {
            uid,
            follow,
            tail,
            since,
            previous,
            log_type,
        } => {
            logs(
                ctx,
                uid,
                since.clone(),
                *tail,
                *previous,
                log_type.map(LogType::api_value),
                *follow,
            )
            .await
        }
        WorkloadCommands::Events { uid, limit } => events(ctx, uid, *limit).await,
    }
}

pub(crate) async fn list(
    ctx: &Context,
    workload_type: Option<String>,
    state: Option<String>,
    project: Option<String>,
    name: Option<String>,
    limit: u32,
) -> Result<()> {
    let params = ListWorkloadsParams {
        page: Page {
            limit: Some(limit),
            cursor: None,
        },
        workload_type,
        status: state,
        project_id: ctx.project(project),
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
    let mut t = table::table(&["UID", "NAME", "TYPE", "STATE", "RESOURCE", "COST", "AGE"]);
    let mut running = 0usize;
    let mut burning = 0.0f64;
    for workload in &workloads.items {
        let status = workload
            .state
            .as_ref()
            .map(|s| s.status.to_string())
            .unwrap_or_default();
        if format::classify(&status) == format::StateKind::Ok {
            running += 1;
            burning += workload.cost_per_hour.unwrap_or_default();
        }
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
            table::uid_cell(&workload.uid),
            Cell::new(&workload.name),
            table::type_cell(&workload.workload_type),
            table::state_cell(&status),
            Cell::new(resource),
            Cell::new(cost),
            table::dim_cell(format::relative_time(workload.created_at)),
        ]);
    }
    table::print(&t);
    let mut parts = vec![plural(workloads.items.len(), "workload")];
    parts.push(format!("{running} running"));
    if burning > 0.0 {
        parts.push(format!("{} burning", format::cost_per_hour(burning)));
    }
    table::summary(parts.join(&format!(" {} ", style::SEP)));
    Ok(())
}

pub(crate) fn plural(count: usize, noun: &str) -> String {
    if count == 1 {
        format!("{count} {noun}")
    } else {
        format!("{count} {noun}s")
    }
}

async fn get(ctx: &Context, uid: &str) -> Result<()> {
    let workload = ctx.client.workloads().get(uid).await?;
    if ctx.json() {
        return format::print_json(&workload);
    }
    style::field("UID", workload.uid.color(palettes::ACCENT).to_string());
    style::field("Name", &workload.name);
    style::field("Type", format::type_badge(&workload.workload_type));
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
            style::field(
                "URL",
                format!(
                    "{} {} :{}",
                    url.url.color(palettes::ACCENT).underline(),
                    style::ARROW.color(palettes::DIM),
                    url.port
                ),
            );
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

async fn state(ctx: &Context, uid: &str) -> Result<()> {
    let state = ctx.client.workloads().state(uid).await?;
    if ctx.json() {
        return format::print_json(&state);
    }
    style::field("UID", state.uid.color(palettes::ACCENT).to_string());
    style::field("Type", format::type_badge(&state.workload_type));
    style::field("Status", format::state_badge(state.status.as_str()));
    if !state.message.is_empty() {
        style::field("Message", &state.message);
    }
    style::field(
        "Replicas",
        format!("{}/{}", state.ready_replicas, state.total_replicas),
    );
    if let Some(ip) = &state.public_ip {
        style::field("Public IP", ip);
    }
    if let Some(port) = state.ssh_port {
        style::field("SSH port", port.to_string());
    }
    style::field("Updated", format::relative_time(state.updated_at));
    Ok(())
}

async fn delete(ctx: &Context, uid: &str, yes: bool) -> Result<()> {
    if crate::output::prompt::is_tty()
        && !yes
        && !crate::output::prompt::confirm(&format!("Delete workload {uid}?"), false)?
    {
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
    log_type: Option<String>,
    follow: bool,
) -> Result<()> {
    if log_type.is_some() {
        let workload = ctx.client.workloads().get(uid).await?;
        if workload.workload_type != "VM" {
            return Err(CliError::Config(format!(
                "{uid} is a {} — --log-type applies only to VMs",
                workload.workload_type
            )));
        }
    }
    let opts = LogOptions {
        since,
        tail,
        previous,
        log_type,
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
            table::state_cell(&event.new_status.clone().unwrap_or_default()),
            Cell::new(message),
        ]);
    }
    table::print(&t);
    table::summary(plural(events.items.len(), "event"));
    Ok(())
}

pub(crate) async fn register(ctx: &Context, req: &CreateWorkloadRequest) -> Result<Workload> {
    let spinner = progress::spinner_if(!ctx.json(), format!("Registering {}…", req.name));
    match ctx.client.workloads().create(req).await {
        Ok(workload) => {
            spinner.finish_ok(format!(
                "Registered {} {}",
                workload.name,
                workload.uid.color(palettes::ACCENT)
            ));
            Ok(workload)
        }
        Err(e) => {
            spinner.finish_fail("Registration failed");
            Err(e.into())
        }
    }
}

pub(crate) async fn start_workload(ctx: &Context, uid: &str) -> Result<()> {
    let spinner = progress::spinner_if(!ctx.json(), format!("Starting {}…", format::short_uid(uid)));
    match ctx.client.workloads().deploy(uid).await {
        Ok(workload) => {
            let status = workload
                .state
                .as_ref()
                .map(|s| s.status.to_string())
                .unwrap_or_else(|| "provisioning".to_string());
            spinner.finish_ok(format!(
                "Started {} {}",
                workload.uid.color(palettes::ACCENT),
                format!("{} {}", style::SEP, status.to_lowercase()).color(palettes::DIM)
            ));
            if ctx.json() {
                return format::print_json(&workload);
            }
            Ok(())
        }
        Err(e) => {
            spinner.finish_fail("Start failed");
            Err(e.into())
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum DeployKind {
    Rental,
    Vm,
}

/// Full deploy flow with a live step checklist, ending with the deployed
/// summary and copy-pasteable next actions.
pub(crate) async fn deploy_flow(
    ctx: &Context,
    req: &CreateWorkloadRequest,
    kind: DeployKind,
) -> Result<()> {
    let started_at = Instant::now();
    let checklist = progress::Checklist::new(!ctx.json(), &["Registering", "Starting"]);

    checklist.start(0, &req.name);
    let created = match ctx.client.workloads().create(req).await {
        Ok(workload) => {
            checklist.done(
                0,
                "Registered",
                &workload.uid.color(palettes::ACCENT).to_string(),
            );
            workload
        }
        Err(e) => {
            checklist.fail(0, "Registration failed");
            return Err(e.into());
        }
    };

    checklist.start(1, &created.uid);
    let workload = match ctx.client.workloads().deploy(&created.uid).await {
        Ok(workload) => {
            let status = workload
                .state
                .as_ref()
                .map(|s| s.status.to_string().to_lowercase())
                .unwrap_or_else(|| "provisioning".to_string());
            checklist.done(1, "Started", &status.color(palettes::DIM).to_string());
            workload
        }
        Err(e) => {
            checklist.fail(1, "Start failed");
            return Err(e.into());
        }
    };

    if ctx.json() {
        return format::print_json(&workload);
    }

    let status = workload
        .state
        .as_ref()
        .map(|s| s.status.to_string().to_lowercase())
        .unwrap_or_else(|| "provisioning".to_string());
    let sep = style::SEP.color(palettes::DIM);
    eprintln!();
    eprintln!(
        "{} {sep} {} {sep} {}",
        format!("{} Deployed {}", style::TICK, workload.uid)
            .color(palettes::SUCCESS)
            .bold(),
        status.color(format::state_color(format::classify(&status))),
        elapsed(started_at).color(palettes::DIM),
    );
    if let Some(state) = &workload.state {
        for url in &state.urls {
            style::next_action(
                "endpoint",
                format!(
                    "{} {} :{}",
                    url.url.color(palettes::ACCENT).underline(),
                    style::ARROW.color(palettes::DIM),
                    url.port
                ),
            );
        }
    }
    style::next_action("logs", format!("targon workload logs -f {}", workload.uid));
    match kind {
        DeployKind::Rental => {
            style::next_action(
                "shell",
                format!("targon rental exec {} -- bash", workload.uid),
            );
        }
        DeployKind::Vm => {
            style::next_action(
                "connect",
                format!("targon workload get {}", workload.uid),
            );
        }
    }
    Ok(())
}

fn elapsed(from: Instant) -> String {
    let secs = from.elapsed().as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else {
        format!("{}m {}s", secs / 60, secs % 60)
    }
}

pub(crate) fn parse_env(raw: &str) -> Result<EnvVar> {
    let (name, value) = raw
        .split_once('=')
        .ok_or_else(|| CliError::Config(format!("invalid --env '{raw}', expected KEY=VAL")))?;
    Ok(EnvVar {
        name: name.to_string(),
        value: value.to_string(),
    })
}

pub(crate) fn parse_port(raw: &str) -> Result<Port> {
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

pub(crate) fn parse_volume(raw: &str) -> Result<VolumeMount> {
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
