use std::io::Read;

use clap::{Args, Subcommand};
use comfy_table::Cell;

use crate::client::types::{CreateWorkloadRequest, VmConfig, WorkloadType};
use crate::commands::{self, workload, Context};
use crate::error::{CliError, Result};
use crate::output::{format, progress, prompt, style, table};

#[derive(Debug, Clone, Args)]
pub struct VmSpec {
    /// VM name
    #[arg(long)]
    pub name: Option<String>,
    /// Image from the catalog (see `targon vm images`)
    #[arg(long)]
    pub image: Option<String>,
    /// Resource SKU (see `targon inventory`)
    #[arg(long)]
    pub resource: Option<String>,
    /// SSH key (repeatable; cannot be changed after boot)
    #[arg(long = "ssh-key", value_name = "UID")]
    pub ssh_key: Vec<String>,
    /// Port to expose (repeatable, comma-separated ok)
    #[arg(long = "port", value_name = "PORT", value_delimiter = ',')]
    pub port: Vec<String>,
    /// Project (default: active project from `project use`)
    #[arg(long)]
    pub project: Option<String>,
    /// Read the VM password from stdin (for CI)
    #[arg(long = "password-stdin")]
    pub password_stdin: bool,
}

#[derive(Debug, Subcommand)]
pub enum VmCommands {
    /// Create and start a VM from the image catalog
    #[command(override_usage = "targon vm deploy --name <NAME> --image <IMAGE> --resource <RESOURCE> [OPTIONS]")]
    Deploy {
        #[command(flatten)]
        spec: VmSpec,
        /// Skip confirmation
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Register a VM without starting it
    #[command(override_usage = "targon vm create --name <NAME> --image <IMAGE> --resource <RESOURCE> [OPTIONS]")]
    Create {
        #[command(flatten)]
        spec: VmSpec,
        /// Skip confirmation
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Start a registered VM
    Start { uid: String },
    /// Reboot a VM
    Reboot { uid: String },
    /// List VMs
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
    /// List available VM images
    Images,
}

pub async fn handle(ctx: &Context, cmd: &VmCommands) -> Result<()> {
    match cmd {
        VmCommands::Deploy { spec, yes } => deploy(ctx, spec.clone(), *yes).await,
        VmCommands::Create { spec, yes } => create(ctx, spec.clone(), *yes).await,
        VmCommands::Start { uid } => start(ctx, uid).await,
        VmCommands::Reboot { uid } => reboot(ctx, uid).await,
        VmCommands::List {
            state,
            project,
            name,
            limit,
        } => {
            workload::list(
                ctx,
                Some("VM".to_string()),
                state.clone(),
                project.clone(),
                name.clone(),
                *limit,
            )
            .await
        }
        VmCommands::Images => images(ctx).await,
    }
}

async fn deploy(ctx: &Context, spec: VmSpec, yes: bool) -> Result<()> {
    let req = build_request(ctx, spec, yes, "Deploy").await?;
    workload::deploy_flow(ctx, &req, workload::DeployKind::Vm).await
}

async fn create(ctx: &Context, spec: VmSpec, yes: bool) -> Result<()> {
    let req = build_request(ctx, spec, yes, "Create").await?;
    let created = workload::register(ctx, &req).await?;
    if ctx.json() {
        return format::print_json(&created);
    }
    style::next_action("start", format!("targon vm start {}", created.uid));
    Ok(())
}

async fn start(ctx: &Context, uid: &str) -> Result<()> {
    commands::ensure_vm(ctx, uid, "start").await?;
    workload::start_workload(ctx, uid).await
}

async fn reboot(ctx: &Context, uid: &str) -> Result<()> {
    commands::ensure_vm(ctx, uid, "reboot").await?;
    let spinner = progress::spinner_if(!ctx.json(), format!("Rebooting {}…", format::short_uid(uid)));
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

async fn images(ctx: &Context) -> Result<()> {
    let images = ctx.client.workloads().vm_images().await?;
    if ctx.json() {
        return format::print_json(&images);
    }
    if images.is_empty() {
        style::dim("no vm images available");
        return Ok(());
    }
    let mut t = table::table(&["NAME", "DISPLAY", "DESCRIPTION"]);
    for image in &images {
        t.add_row(vec![
            Cell::new(&image.name),
            Cell::new(&image.display_name),
            table::dim_cell(image.description.clone()),
        ]);
    }
    table::print(&t);
    table::summary(workload::plural(images.len(), "image"));
    Ok(())
}

async fn build_request(
    ctx: &Context,
    spec: VmSpec,
    yes: bool,
    verb: &str,
) -> Result<CreateWorkloadRequest> {
    let name = match spec.name {
        Some(name) => name,
        None => {
            prompt::require_tty("--name")?;
            prompt::input("VM name")?
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
            commands::select_resource(ctx, "vm").await?
        }
    };
    let password = read_password(spec.password_stdin)?;
    let ports = if spec.port.is_empty() && prompt::is_tty() {
        commands::prompt_list("Ports to expose (comma-separated, blank for none)")?
    } else {
        spec.port
    };
    let ssh_keys = if !spec.ssh_key.is_empty() {
        spec.ssh_key
    } else if prompt::is_tty() {
        commands::select_ssh_keys(ctx).await?
    } else {
        vec![]
    };

    let mut req = CreateWorkloadRequest::new(WorkloadType::Vm, name, image, resource_name);
    req.project_id = ctx.project(spec.project);
    req.ssh_keys = ssh_keys;
    for raw in &ports {
        req.ports.push(workload::parse_port(raw)?);
    }
    req.vm_config = Some(VmConfig { password });

    if prompt::is_tty() && !yes {
        let pricing = commands::resource_pricing(ctx, &req.resource_name, "vm").await;
        let mut rows = vec![
            ("name", req.name.clone()),
            ("image", req.image.clone()),
            (
                "resource",
                crate::commands::rental::resource_value(&req.resource_name, pricing.as_ref()),
            ),
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
        if !req.ssh_keys.is_empty() {
            rows.push(("ssh keys", req.ssh_keys.join(", ")));
        }
        eprintln!();
        style::summary_box("vm", &rows);
        let confirm_msg = match &pricing {
            Some(p) => format!("{verb} for {}?", format::cost_per_hour(p.cost_per_hour)),
            None => format!("{verb} this vm?"),
        };
        if !prompt::confirm(&confirm_msg, false)? {
            return Err(CliError::Cancelled);
        }
    }
    Ok(req)
}

async fn select_image(ctx: &Context) -> Result<String> {
    let images = ctx.client.workloads().vm_images().await?;
    if images.is_empty() {
        return Err(CliError::Config("no vm images available".to_string()));
    }
    let labels: Vec<String> = images
        .iter()
        .map(|i| format!("{:<24} {}", i.display_name, i.description))
        .collect();
    let idx = prompt::select("Select an image", &labels)?;
    Ok(images[idx].name.clone())
}

fn read_password(from_stdin: bool) -> Result<String> {
    let password = if from_stdin {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf.trim().to_string()
    } else {
        prompt::require_tty("--password-stdin")?;
        prompt::password("VM password")?
    };
    if password.chars().count() < 4 {
        return Err(CliError::Config(
            "vm password must be at least 4 characters".to_string(),
        ));
    }
    Ok(password)
}
