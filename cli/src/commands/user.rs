use clap::Subcommand;

use crate::commands::Context;
use crate::error::Result;
use crate::output::{format, style};

#[derive(Debug, Subcommand)]
pub enum UserCommands {
    /// Show wallet address and remaining credits
    Wallet,
}

pub async fn handle(ctx: &Context, cmd: &UserCommands) -> Result<()> {
    match cmd {
        UserCommands::Wallet => wallet(ctx).await,
    }
}

async fn wallet(ctx: &Context) -> Result<()> {
    let wallet = ctx.client.user().wallet().await?;
    let credits = ctx.client.user().credits().await?;

    if ctx.json() {
        return format::print_json(&serde_json::json!({
            "address": wallet.address,
            "credits": credits.credits,
            "currency": credits.currency,
        }));
    }

    style::field("Address", &wallet.address);
    style::field(
        "Credits",
        format!("{:.2} {}", credits.credits, credits.currency),
    );
    Ok(())
}
