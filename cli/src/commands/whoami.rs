use colored::Colorize;

use crate::commands::Context;
use crate::error::Result;
use crate::output::{format, palettes, style};

pub async fn handle(ctx: &Context) -> Result<()> {
    let wallet = ctx.client.user().wallet().await?;
    let credits = ctx.client.user().credits().await?;

    if ctx.json() {
        return format::print_json(&serde_json::json!({
            "address": wallet.address,
            "credits": credits.credits,
            "currency": credits.currency,
            "profile": ctx.profile,
            "base_url": ctx.base_url,
            "project": ctx.project,
        }));
    }

    style::field("Wallet", &wallet.address);
    style::field(
        "Credits",
        format::credits_badge(credits.credits, &credits.currency),
    );
    style::field(
        "Profile",
        format!(
            "{} {}",
            ctx.profile,
            format!("{} {}", style::ARROW, ctx.base_url).color(palettes::DIM)
        ),
    );
    if let Some(project) = &ctx.project {
        style::field("Project", project.color(palettes::ACCENT).to_string());
    }
    Ok(())
}
