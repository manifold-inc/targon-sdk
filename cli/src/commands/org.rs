use clap::Subcommand;
use colored::Colorize;
use comfy_table::Cell;

use crate::client::pagination::Page;
use crate::client::types::{CreateOrgRequest, Org, UpdateOrgRequest};
use crate::commands::{workload, Context};
use crate::error::{CliError, Result};
use crate::output::{format, palettes, prompt, style, table};

#[derive(Debug, Subcommand)]
pub enum OrgCommands {
    /// List organizations you belong to
    List {
        #[arg(long, default_value_t = 50)]
        limit: u32,
        #[arg(long)]
        cursor: Option<String>,
    },
    /// Create an organization
    Create {
        #[arg(long)]
        name: String,
        #[arg(long)]
        slug: String,
    },
    /// Set the active organization for this profile
    Use { slug: String },
    /// Show the selected organization
    Get,
    /// Update the selected organization
    Update {
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        slug: Option<String>,
        #[arg(long = "billing-email")]
        billing_email: Option<String>,
    },
    /// Delete the selected organization
    Delete {
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Show credits for the selected organization
    Credits,
    /// Show the wallet for the selected organization
    Wallet,
}

pub async fn handle(ctx: &Context, cmd: &OrgCommands) -> Result<()> {
    match cmd {
        OrgCommands::List { limit, cursor } => list(ctx, *limit, cursor.clone()).await,
        OrgCommands::Create { name, slug } => create(ctx, name, slug).await,
        OrgCommands::Use { slug } => set_active(ctx, slug).await,
        OrgCommands::Get => get(ctx).await,
        OrgCommands::Update {
            name,
            slug,
            billing_email,
        } => update(ctx, name.clone(), slug.clone(), billing_email.clone()).await,
        OrgCommands::Delete { yes } => delete(ctx, *yes).await,
        OrgCommands::Credits => credits(ctx).await,
        OrgCommands::Wallet => wallet(ctx).await,
    }
}

async fn list(ctx: &Context, limit: u32, cursor: Option<String>) -> Result<()> {
    let orgs = ctx
        .client
        .orgs()
        .list(&Page {
            limit: Some(limit),
            cursor,
        })
        .await?;
    if ctx.json() {
        return format::print_json(&orgs);
    }
    if orgs.items.is_empty() {
        style::dim("no organizations");
        return Ok(());
    }
    let mut output = table::table(&["", "SLUG", "NAME", "TYPE", "ROLE", "CREDITS", "CREATED"]);
    for org in &orgs.items {
        let active = ctx.org.as_deref() == Some(org.slug.as_str());
        output.add_row(vec![
            table::active_org_cell(active),
            table::org_slug_cell(&org.slug, active),
            Cell::new(&org.name),
            table::org_type_cell(&org.org_type),
            table::org_role_cell(org.role.as_str()),
            table::org_credits_cell(org.credits),
            table::dim_cell(format::relative_time(org.created_at)),
        ]);
    }
    table::print(&output);
    let marker = "▸".color(palettes::ACCENT).to_string();
    let mut summary = format!(
        "{} {} {marker} = active",
        workload::plural(orgs.items.len(), "organization"),
        style::SEP
    );
    if let Some(target) = orgs
        .items
        .iter()
        .find(|org| ctx.org.as_deref() != Some(org.slug.as_str()))
    {
        summary.push_str(&format!(
            " {} targon org use {} to switch",
            style::SEP,
            target.slug
        ));
    }
    table::summary(summary);
    Ok(())
}

async fn create(ctx: &Context, name: &str, slug: &str) -> Result<()> {
    let org = ctx
        .client
        .orgs()
        .create(&CreateOrgRequest {
            name: name.to_string(),
            slug: slug.to_string(),
        })
        .await?;
    if ctx.json() {
        return format::print_json(&org);
    }
    style::success(format!("created organization {} ({})", org.name, org.slug));
    style::next_action("select", format!("targon org use {}", org.slug));
    Ok(())
}

async fn set_active(ctx: &Context, slug: &str) -> Result<()> {
    let org = ctx.client.orgs().get(slug).await?;
    let profile = crate::config::set_org(Some(&ctx.profile), Some(org.slug.clone()))?;
    if ctx.json() {
        return format::print_json(&serde_json::json!({
            "profile": profile,
            "org": org,
        }));
    }
    style::success(format!(
        "active organization for profile '{profile}' set to {} ({})",
        org.name, org.slug
    ));
    Ok(())
}

async fn get(ctx: &Context) -> Result<()> {
    let org = ctx.client.orgs().get(ctx.org()?).await?;
    print_org(ctx, &org)
}

async fn update(
    ctx: &Context,
    name: Option<String>,
    slug: Option<String>,
    billing_email: Option<String>,
) -> Result<()> {
    if name.is_none() && slug.is_none() && billing_email.is_none() {
        return Err(CliError::Config(
            "at least one of --name, --slug, or --billing-email is required".to_string(),
        ));
    }
    let current_slug = ctx.org()?.to_string();
    let org = ctx
        .client
        .orgs()
        .update(
            &current_slug,
            &UpdateOrgRequest {
                name,
                slug,
                billing_email,
            },
        )
        .await?;
    if org.slug != current_slug {
        crate::config::rename_org_slug(Some(&ctx.profile), &current_slug, &org.slug)?;
    }
    if ctx.json() {
        return format::print_json(&org);
    }
    style::success(format!("updated organization {}", org.slug));
    Ok(())
}

async fn delete(ctx: &Context, yes: bool) -> Result<()> {
    let slug = ctx.org()?.to_string();
    if prompt::is_tty() && !yes && !prompt::confirm(&format!("Delete organization {slug}?"), false)?
    {
        return Err(CliError::Cancelled);
    }
    ctx.client.orgs().delete(&slug).await?;
    crate::config::clear_org_context(Some(&ctx.profile), &slug)?;
    style::success(format!("deleted organization {slug}"));
    Ok(())
}

async fn credits(ctx: &Context) -> Result<()> {
    let credits = ctx.client.credits(ctx.org()?).get().await?;
    if ctx.json() {
        return format::print_json(&credits);
    }
    style::field(
        "Credits",
        format::credits_badge(credits.credits, &credits.currency),
    );
    Ok(())
}

async fn wallet(ctx: &Context) -> Result<()> {
    let wallet = ctx.client.wallet(ctx.org()?).get().await?;
    if ctx.json() {
        return format::print_json(&wallet);
    }
    style::field("Wallet", wallet.address.color(palettes::ACCENT).to_string());
    Ok(())
}

fn print_org(ctx: &Context, org: &Org) -> Result<()> {
    if ctx.json() {
        return format::print_json(org);
    }
    style::field("UID", org.uid.color(palettes::ACCENT).to_string());
    style::field("Slug", &org.slug);
    style::field("Name", &org.name);
    style::field("Type", &org.org_type);
    style::field("Role", org.role.as_str().to_ascii_lowercase());
    if !org.billing_email.is_empty() {
        style::field("Billing email", &org.billing_email);
    }
    style::field("Credits", format::credits_badge(org.credits, "USD"));
    style::field("Overage", org.overage.to_string());
    style::field("Created", format::relative_time(org.created_at));
    Ok(())
}
