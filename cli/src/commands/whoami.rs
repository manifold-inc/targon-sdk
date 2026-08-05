use colored::Colorize;

use crate::commands::Context;
use crate::error::Result;
use crate::output::{format, palettes, style};

pub async fn handle(ctx: &Context) -> Result<()> {
    let org = ctx.org()?;
    let wallet = ctx.client.wallet(org).get().await?;
    let credits = ctx.client.credits(org).get().await?;

    if ctx.json() {
        return format::print_json(&serde_json::json!({
            "address": wallet.address,
            "credits": credits.credits,
            "currency": credits.currency,
            "profile": ctx.profile,
            "base_url": ctx.base_url,
            "org": org,
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
    style::field("Organization", org.color(palettes::ACCENT).to_string());
    if let Some(project) = &ctx.project {
        style::field("Project", project.color(palettes::ACCENT).to_string());
    }
    Ok(())
}
