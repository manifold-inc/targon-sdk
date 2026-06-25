use std::process::ExitCode;

use clap::Parser;

use targon_cli::cli::{Cli, Commands};
use targon_cli::client::Client;
use targon_cli::commands::{self, Context};
use targon_cli::config;
use targon_cli::error::{CliError, Result};
use targon_cli::output::{style, OutputFormat};

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            style::error(e.to_string());
            ExitCode::from(e.exit_code())
        }
    }
}

async fn run(cli: Cli) -> Result<()> {
    let format = if cli.json {
        OutputFormat::Json
    } else {
        OutputFormat::Human
    };

    if let Commands::Auth(cmd) = &cli.command {
        return commands::auth::handle(cmd, cli.profile.as_deref(), cli.base_url.as_deref()).await;
    }

    let requires_key = !matches!(cli.command, Commands::Version | Commands::Inventory(_));
    let resolved = config::resolve(cli.profile.as_deref(), cli.base_url.as_deref())?;
    let api_key = if requires_key {
        resolved.api_key.ok_or(CliError::NotAuthenticated)?
    } else {
        resolved.api_key.unwrap_or_default()
    };

    let client = Client::builder()
        .api_key(api_key)
        .base_url(resolved.base_url)
        .build()?;
    let ctx = Context::new(client, format);

    match &cli.command {
        Commands::Auth(_) => unreachable!("auth handled above"),
        Commands::Workload(cmd) => commands::workload::handle(&ctx, cmd).await,
        Commands::Volume(cmd) => commands::volume::handle(&ctx, cmd).await,
        Commands::SshKey(cmd) => commands::ssh_key::handle(&ctx, cmd).await,
        Commands::Project(cmd) => commands::project::handle(&ctx, cmd).await,
        Commands::Inventory(args) => commands::inventory::handle(&ctx, args).await,
        Commands::User(cmd) => commands::user::handle(&ctx, cmd).await,
        Commands::Version => commands::version::handle(&ctx).await,
    }
}
