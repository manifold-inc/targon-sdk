use std::path::{Path, PathBuf};

use clap::Subcommand;
use colored::Colorize;
use comfy_table::Cell;

use crate::client::pagination::Page;
use crate::client::types::CreateSshKeyRequest;
use crate::commands::{workload, Context};
use crate::error::{CliError, Result};
use crate::output::{format, palettes, style, table};

#[derive(Debug, Subcommand)]
pub enum SshKeyCommands {
    /// Register a public key from a file
    Add {
        path: PathBuf,
        #[arg(long)]
        name: Option<String>,
    },
    /// List registered SSH keys
    List,
    /// Show an SSH key
    Get {
        uid: String,
    },
    /// Delete an SSH key
    Delete {
        uid: String,
    },
}

pub async fn handle(ctx: &Context, cmd: &SshKeyCommands) -> Result<()> {
    match cmd {
        SshKeyCommands::Add { path, name } => add(ctx, path, name.as_deref()).await,
        SshKeyCommands::List => list(ctx).await,
        SshKeyCommands::Get { uid } => get(ctx, uid).await,
        SshKeyCommands::Delete { uid } => delete(ctx, uid).await,
    }
}

async fn add(ctx: &Context, path: &Path, name: Option<&str>) -> Result<()> {
    let contents = std::fs::read_to_string(path)?;
    let public_key = contents.trim();
    if public_key.is_empty() {
        return Err(CliError::Config(format!(
            "{} does not contain a public key",
            path.display()
        )));
    }
    let name = name
        .map(str::to_string)
        .or_else(|| public_key.split_whitespace().nth(2).map(str::to_string))
        .or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "ssh-key".to_string());

    let key = ctx
        .client
        .ssh_keys()
        .create(&CreateSshKeyRequest {
            name,
            ssh_key: public_key.to_string(),
        })
        .await?;

    if ctx.json() {
        return format::print_json(&key);
    }
    style::success(format!("added ssh key {} ({})", key.name, key.uid));
    Ok(())
}

async fn list(ctx: &Context) -> Result<()> {
    let keys = ctx.client.ssh_keys().list(&Page::default()).await?;
    if ctx.json() {
        return format::print_json(&keys);
    }
    if keys.items.is_empty() {
        style::dim("no ssh keys");
        return Ok(());
    }
    let mut t = table::table(&["UID", "NAME", "FINGERPRINT", "CREATED"]);
    for key in &keys.items {
        t.add_row(vec![
            table::uid_cell(&key.uid),
            Cell::new(&key.name),
            table::dim_cell(fingerprint(&key.public_key)),
            table::dim_cell(format::relative_time(key.created_at)),
        ]);
    }
    table::print(&t);
    table::summary(workload::plural(keys.items.len(), "key"));
    Ok(())
}

async fn get(ctx: &Context, uid: &str) -> Result<()> {
    let key = ctx.client.ssh_keys().get(uid).await?;
    if ctx.json() {
        return format::print_json(&key);
    }
    style::field("UID", key.uid.color(palettes::ACCENT).to_string());
    style::field("Name", &key.name);
    style::field("Created", format::relative_time(key.created_at));
    println!("{}", key.public_key);
    Ok(())
}

async fn delete(ctx: &Context, uid: &str) -> Result<()> {
    ctx.client.ssh_keys().delete(uid).await?;
    style::success(format!("deleted ssh key {uid}"));
    Ok(())
}

fn fingerprint(public_key: &str) -> String {
    let parts: Vec<&str> = public_key.split_whitespace().collect();
    match parts.as_slice() {
        [kind, data, ..] => {
            let chars: Vec<char> = data.chars().collect();
            let start = chars.len().saturating_sub(12);
            let tail: String = chars[start..].iter().collect();
            format!("{kind} …{tail}")
        }
        _ => format::short_uid(public_key),
    }
}
