use clap::Subcommand;
use colored::Colorize;
use comfy_table::Cell;

use crate::client::pagination::Page;
use crate::client::types::CreateServiceTokenRequest;
use crate::commands::{workload, Context};
use crate::error::{CliError, Result};
use crate::output::{format, palettes, prompt, style, table};

#[derive(Debug, Subcommand)]
pub enum ServiceTokenCommands {
    /// List service tokens in the selected organization
    List {
        #[arg(long, default_value_t = 50)]
        limit: u32,
        #[arg(long)]
        cursor: Option<String>,
    },
    /// Create a service token in the selected organization
    Create { name: String },
    /// Delete a service token from the selected organization
    Delete {
        uid: String,
        #[arg(long, short = 'y')]
        yes: bool,
    },
}

pub async fn handle(ctx: &Context, cmd: &ServiceTokenCommands) -> Result<()> {
    match cmd {
        ServiceTokenCommands::List { limit, cursor } => list(ctx, *limit, cursor.clone()).await,
        ServiceTokenCommands::Create { name } => create(ctx, name).await,
        ServiceTokenCommands::Delete { uid, yes } => delete(ctx, uid, *yes).await,
    }
}

async fn list(ctx: &Context, limit: u32, cursor: Option<String>) -> Result<()> {
    let tokens = ctx
        .client
        .service_tokens(ctx.org()?)
        .list(&Page {
            limit: Some(limit),
            cursor,
        })
        .await?;
    if ctx.json() {
        return format::print_json(&tokens);
    }
    if tokens.items.is_empty() {
        style::dim("no service tokens");
        return Ok(());
    }
    let mut output = table::table(&["UID", "NAME", "CREATED BY", "CREATED"]);
    for token in &tokens.items {
        output.add_row(vec![
            table::uid_cell(&token.uid),
            Cell::new(&token.name),
            Cell::new(&token.created_by.username),
            table::dim_cell(format::relative_time(token.created_at)),
        ]);
    }
    table::print(&output);
    table::summary(workload::plural(tokens.items.len(), "token"));
    Ok(())
}

async fn create(ctx: &Context, name: &str) -> Result<()> {
    let token = ctx
        .client
        .service_tokens(ctx.org()?)
        .create(&CreateServiceTokenRequest {
            name: name.to_string(),
        })
        .await?;
    if ctx.json() {
        return format::print_json(&token);
    }
    style::success(format!("created service token {}", token.uid));
    if let Some(raw) = token.token {
        style::warn("copy this token now; it will not be shown again");
        println!("{}", raw.color(palettes::ACCENT));
    }
    Ok(())
}

async fn delete(ctx: &Context, uid: &str, yes: bool) -> Result<()> {
    if prompt::is_tty() && !yes && !prompt::confirm(&format!("Delete service token {uid}?"), false)?
    {
        return Err(CliError::Cancelled);
    }
    ctx.client.service_tokens(ctx.org()?).delete(uid).await?;
    style::success(format!("deleted service token {uid}"));
    Ok(())
}
