use crate::commands::Context;
use crate::error::Result;
use crate::output::{format, style};

pub async fn handle(ctx: &Context) -> Result<()> {
    let version = ctx.client.version().get().await?;
    if ctx.json() {
        return format::print_json(&version);
    }
    style::field("Name", &version.name);
    style::field("Version", &version.version);
    style::field("Git hash", &version.git_hash);
    style::field("Build date", &version.build_date);
    Ok(())
}
