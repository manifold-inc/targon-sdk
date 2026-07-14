use clap::Subcommand;
use colored::Colorize;
use comfy_table::Cell;

use crate::client::pagination::Page;
use crate::client::types::{CreateProjectRequest, UpdateProjectRequest};
use crate::commands::{workload, Context};
use crate::error::Result;
use crate::output::{format, palettes, style, table};

#[derive(Debug, Subcommand)]
pub enum ProjectCommands {
    /// List projects
    List,
    /// Show a project
    Get {
        uid: String,
    },
    /// Create a project
    Create {
        name: String,
    },
    /// Rename a project
    Update {
        uid: String,
        #[arg(long)]
        name: String,
    },
    /// Delete a project
    Delete {
        uid: String,
    },
    /// Set the active project for this profile
    Use {
        uid: String,
    },
}

pub async fn handle(ctx: &Context, cmd: &ProjectCommands) -> Result<()> {
    match cmd {
        ProjectCommands::List => list(ctx).await,
        ProjectCommands::Get { uid } => get(ctx, uid).await,
        ProjectCommands::Create { name } => create(ctx, name).await,
        ProjectCommands::Update { uid, name } => update(ctx, uid, name).await,
        ProjectCommands::Delete { uid } => delete(ctx, uid).await,
        ProjectCommands::Use { uid } => set_active(ctx, uid).await,
    }
}

async fn set_active(ctx: &Context, uid: &str) -> Result<()> {
    let project = ctx.client.projects().get(uid).await?;
    let profile = crate::config::set_project(Some(&ctx.profile), Some(project.uid.clone()))?;
    style::success(format!(
        "active project for profile '{profile}' set to {} ({})",
        project.name, project.uid
    ));
    Ok(())
}

async fn list(ctx: &Context) -> Result<()> {
    let projects = ctx.client.projects().list(&Page::default()).await?;
    if ctx.json() {
        return format::print_json(&projects);
    }
    if projects.items.is_empty() {
        style::dim("no projects");
        return Ok(());
    }
    let mut t = table::table(&["UID", "NAME", "CREATED"]);
    for project in &projects.items {
        t.add_row(vec![
            table::uid_cell(&project.uid),
            Cell::new(&project.name),
            table::dim_cell(format::relative_time(project.created_at)),
        ]);
    }
    table::print(&t);
    table::summary(workload::plural(projects.items.len(), "project"));
    Ok(())
}

async fn get(ctx: &Context, uid: &str) -> Result<()> {
    let project = ctx.client.projects().get(uid).await?;
    if ctx.json() {
        return format::print_json(&project);
    }
    style::field("UID", project.uid.color(palettes::ACCENT).to_string());
    style::field("Name", &project.name);
    style::field("Created", format::relative_time(project.created_at));
    style::field("Updated", format::relative_time(project.updated_at));
    Ok(())
}

async fn create(ctx: &Context, name: &str) -> Result<()> {
    let project = ctx
        .client
        .projects()
        .create(&CreateProjectRequest { name: name.to_string() })
        .await?;
    if ctx.json() {
        return format::print_json(&project);
    }
    style::success(format!("created project {} ({})", project.name, project.uid));
    Ok(())
}

async fn update(ctx: &Context, uid: &str, name: &str) -> Result<()> {
    let project = ctx
        .client
        .projects()
        .update(uid, &UpdateProjectRequest { name: name.to_string() })
        .await?;
    if ctx.json() {
        return format::print_json(&project);
    }
    style::success(format!("updated project {}", project.uid));
    Ok(())
}

async fn delete(ctx: &Context, uid: &str) -> Result<()> {
    ctx.client.projects().delete(uid).await?;
    style::success(format!("deleted project {uid}"));
    Ok(())
}
