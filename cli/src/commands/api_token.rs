use clap::Subcommand;
use colored::Colorize;
use comfy_table::Cell;

use crate::client::pagination::Page;
use crate::client::types::{CreateApiTokenRequest, UpdateApiTokenRequest};
use crate::commands::{workload, Context};
use crate::error::{CliError, Result};
use crate::output::{format, palettes, prompt, style, table};

#[derive(Debug, Subcommand)]
pub enum ApiTokenCommands {
    /// List personal API tokens
    List {
        #[arg(long, default_value_t = 50)]
        limit: u32,
        #[arg(long)]
        cursor: Option<String>,
    },
    /// Create a personal API token
    Create { name: String },
    /// Rename a personal API token
    Update {
        uid: String,
        #[arg(long)]
        name: String,
    },
    /// Delete a personal API token
    Delete {
        uid: String,
        #[arg(long, short = 'y')]
        yes: bool,
    },
}

pub async fn handle(ctx: &Context, cmd: &ApiTokenCommands) -> Result<()> {
    match cmd {
        ApiTokenCommands::List { limit, cursor } => list(ctx, *limit, cursor.clone()).await,
        ApiTokenCommands::Create { name } => create(ctx, name).await,
        ApiTokenCommands::Update { uid, name } => update(ctx, uid, name).await,
        ApiTokenCommands::Delete { uid, yes } => delete(ctx, uid, *yes).await,
    }
}

async fn list(ctx: &Context, limit: u32, cursor: Option<String>) -> Result<()> {
    let tokens = ctx
        .client
        .api_tokens()
        .list(&Page {
            limit: Some(limit),
            cursor,
        })
        .await?;
    if ctx.json() {
        return format::print_json(&tokens);
    }
    if tokens.items.is_empty() {
        style::dim("no personal API tokens");
        return Ok(());
    }
    let mut output = table::table(&["UID", "NAME", "CREATED", "UPDATED"]);
    for token in &tokens.items {
        output.add_row(vec![
            table::uid_cell(&token.uid),
            Cell::new(&token.name),
            table::dim_cell(format::relative_time(token.created_at)),
            table::dim_cell(format::relative_time(token.updated_at)),
        ]);
    }
    table::print(&output);
    table::summary(workload::plural(tokens.items.len(), "token"));
    Ok(())
}

async fn create(ctx: &Context, name: &str) -> Result<()> {
    let token = ctx
        .client
        .api_tokens()
        .create(&CreateApiTokenRequest {
            name: name.to_string(),
        })
        .await?;
    if ctx.json() {
        return format::print_json(&token);
    }
    style::success(format!("created personal API token {}", token.uid));
    if let Some(raw) = token.token {
        style::warn("copy this token now; it may not be shown again");
        println!("{}", raw.color(palettes::ACCENT));
    }
    Ok(())
}

async fn update(ctx: &Context, uid: &str, name: &str) -> Result<()> {
    let token = ctx
        .client
        .api_tokens()
        .update(
            uid,
            &UpdateApiTokenRequest {
                name: name.to_string(),
            },
        )
        .await?;
    if ctx.json() {
        return format::print_json(&token);
    }
    style::success(format!("updated personal API token {}", token.uid));
    Ok(())
}

async fn delete(ctx: &Context, uid: &str, yes: bool) -> Result<()> {
    if prompt::is_tty()
        && !yes
        && !prompt::confirm(&format!("Delete personal API token {uid}?"), false)?
    {
        return Err(CliError::Cancelled);
    }
    ctx.client.api_tokens().delete(uid).await?;
    style::success(format!("deleted personal API token {uid}"));
    Ok(())
}
