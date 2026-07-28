use clap::{Subcommand, ValueEnum};
use colored::Colorize;
use comfy_table::Cell;

use crate::client::pagination::Page;
use crate::client::types::{
    ListMembersParams, Membership, MembershipStatus, OrgRole, UpdateMemberRequest,
};
use crate::commands::{workload, Context};
use crate::error::{CliError, Result};
use crate::output::{format, palettes, prompt, style, table};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum RoleArg {
    Owner,
    Admin,
    Member,
}

impl From<RoleArg> for OrgRole {
    fn from(value: RoleArg) -> Self {
        match value {
            RoleArg::Owner => Self::Owner,
            RoleArg::Admin => Self::Admin,
            RoleArg::Member => Self::Member,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum StatusArg {
    Active,
    Invited,
    Suspended,
}

impl From<StatusArg> for MembershipStatus {
    fn from(value: StatusArg) -> Self {
        match value {
            StatusArg::Active => Self::Active,
            StatusArg::Invited => Self::Invited,
            StatusArg::Suspended => Self::Suspended,
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum MemberCommands {
    /// List members of the selected organization
    List {
        #[arg(long, value_enum, ignore_case = true)]
        role: Option<RoleArg>,
        #[arg(long, value_enum, ignore_case = true)]
        status: Option<StatusArg>,
        #[arg(long, default_value_t = 50)]
        limit: u32,
        #[arg(long)]
        cursor: Option<String>,
    },
    /// Show an organization member
    Get { username: String },
    /// Change an organization member's role
    Update {
        username: String,
        #[arg(long, value_enum, ignore_case = true)]
        role: RoleArg,
    },
    /// Remove a member from the selected organization
    Remove {
        username: String,
        #[arg(long, short = 'y')]
        yes: bool,
    },
}

pub async fn handle(ctx: &Context, cmd: &MemberCommands) -> Result<()> {
    match cmd {
        MemberCommands::List {
            role,
            status,
            limit,
            cursor,
        } => list(ctx, *role, *status, *limit, cursor.clone()).await,
        MemberCommands::Get { username } => get(ctx, username).await,
        MemberCommands::Update { username, role } => update(ctx, username, *role).await,
        MemberCommands::Remove { username, yes } => remove(ctx, username, *yes).await,
    }
}

async fn list(
    ctx: &Context,
    role: Option<RoleArg>,
    status: Option<StatusArg>,
    limit: u32,
    cursor: Option<String>,
) -> Result<()> {
    let members = ctx
        .client
        .members(ctx.org()?)
        .list(&ListMembersParams {
            page: Page {
                limit: Some(limit),
                cursor,
            },
            role: role.map(Into::into),
            status: status.map(Into::into),
        })
        .await?;
    if ctx.json() {
        return format::print_json(&members);
    }
    if members.items.is_empty() {
        style::dim("no members");
        return Ok(());
    }
    let mut output = table::table(&["USERNAME", "NAME", "EMAIL", "ROLE", "STATUS"]);
    for member in &members.items {
        output.add_row(vec![
            Cell::new(&member.user.username),
            Cell::new(full_name(member)),
            Cell::new(&member.user.email),
            Cell::new(member.role.as_str()),
            table::state_cell(member.status.as_str()),
        ]);
    }
    table::print(&output);
    table::summary(workload::plural(members.items.len(), "member"));
    Ok(())
}

async fn get(ctx: &Context, username: &str) -> Result<()> {
    let member = ctx.client.members(ctx.org()?).get(username).await?;
    if ctx.json() {
        return format::print_json(&member);
    }
    style::field(
        "Username",
        member.user.username.color(palettes::ACCENT).to_string(),
    );
    style::field("Name", full_name(&member));
    style::field("Email", &member.user.email);
    style::field("Role", member.role.as_str());
    style::field("Status", format::state_badge(member.status.as_str()));
    if let Some(joined_at) = member.joined_at {
        style::field("Joined", format::relative_time(joined_at));
    }
    Ok(())
}

async fn update(ctx: &Context, username: &str, role: RoleArg) -> Result<()> {
    let member = ctx
        .client
        .members(ctx.org()?)
        .update(username, &UpdateMemberRequest { role: role.into() })
        .await?;
    if ctx.json() {
        return format::print_json(&member);
    }
    style::success(format!(
        "updated {} to {}",
        member.user.username,
        member.role.as_str()
    ));
    Ok(())
}

async fn remove(ctx: &Context, username: &str, yes: bool) -> Result<()> {
    if prompt::is_tty()
        && !yes
        && !prompt::confirm(&format!("Remove {username} from this organization?"), false)?
    {
        return Err(CliError::Cancelled);
    }
    ctx.client.members(ctx.org()?).delete(username).await?;
    style::success(format!("removed member {username}"));
    Ok(())
}

fn full_name(member: &Membership) -> String {
    let name = [
        member.user.first_name.as_deref(),
        member.user.last_name.as_deref(),
    ]
    .into_iter()
    .flatten()
    .filter(|part| !part.trim().is_empty())
    .collect::<Vec<_>>()
    .join(" ");
    if name.is_empty() {
        member.user.username.clone()
    } else {
        name
    }
}
