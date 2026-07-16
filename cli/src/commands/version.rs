use crate::commands::Context;
use crate::error::Result;
use crate::output::{format, style};

const CLI_VERSION: &str = env!("CARGO_PKG_VERSION");

pub async fn handle(ctx: &Context) -> Result<()> {
    let api = ctx.client.version().get().await?;
    if ctx.json() {
        return format::print_json(&serde_json::json!({
            "cli": CLI_VERSION,
            "api": api,
        }));
    }
    style::field("CLI", format!("targon {CLI_VERSION}"));
    style::field("API name", &api.name);
    style::field("API version", &api.version);
    style::field("Git hash", &api.git_hash);
    style::field("Build date", &api.build_date);
    Ok(())
}
