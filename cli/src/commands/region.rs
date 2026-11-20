use comfy_table::Cell;

use crate::commands::{workload, Context};
use crate::error::Result;
use crate::output::{format, style, table};

pub async fn handle(ctx: &Context) -> Result<()> {
    let regions = ctx.client.inventory().regions().await?;
    if ctx.json() {
        return format::print_json(&regions);
    }
    if regions.is_empty() {
        style::dim("no active regions");
        return Ok(());
    }
    let mut t = table::table(&["UID", "NAME", "COUNTRY", "CONTINENT"]);
    for region in &regions {
        t.add_row(vec![
            table::uid_cell(&region.uid),
            Cell::new(&region.display_name),
            Cell::new(&region.country),
            table::dim_cell(region.continent.clone()),
        ]);
    }
    table::print(&t);
    table::summary(workload::plural(regions.len(), "region"));
    Ok(())
}
