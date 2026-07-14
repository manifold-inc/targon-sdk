use std::process::ExitCode;

use clap::error::ErrorKind;
use clap::{CommandFactory, Parser};
use clap_complete::generate;

use targon_cli::cli::{Cli, Commands};
use targon_cli::client::Client;
use targon_cli::commands::{self, Context};
use targon_cli::config;
use targon_cli::error::{CliError, Result};
use targon_cli::output::{style, OutputFormat};

#[tokio::main]
async fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => return handle_parse_error(e),
    };
    let json = cli.json;
    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            report_error(&e, json);
            ExitCode::from(e.exit_code())
        }
    }
}

/// Human mode: red `✗ fact` line, then dim `→` recovery hints (any extra
/// lines in the error message). JSON mode: a JSON object on stderr.
fn report_error(e: &CliError, json: bool) {
    if json {
        let payload = serde_json::json!({
            "error": e.to_string(),
            "exit_code": e.exit_code(),
        });
        eprintln!("{payload}");
        return;
    }
    let message = e.to_string();
    let mut lines = message.lines();
    if let Some(fact) = lines.next() {
        style::error(fact);
    }
    for hint in lines {
        style::hint(hint);
    }
}

fn handle_parse_error(e: clap::Error) -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if e.kind() == ErrorKind::InvalidSubcommand && args.first().map(String::as_str) == Some("vm") {
        let verb = args.get(1).cloned().unwrap_or_default();
        let uid = args
            .get(2)
            .map(|u| format!(" {u}"))
            .unwrap_or_else(|| " <uid>".to_string());
        style::error(format!("unrecognized subcommand '{verb}'"));
        style::hint(format!("VMs are managed as workloads: targon workload {verb}{uid}"));
        return ExitCode::from(2);
    }
    e.exit()
}

async fn run(cli: Cli) -> Result<()> {
    let format = if cli.json {
        OutputFormat::Json
    } else {
        OutputFormat::Human
    };

    match &cli.command {
        Commands::Auth(cmd) => {
            return commands::auth::handle(cmd, cli.profile.as_deref(), cli.base_url.as_deref())
                .await;
        }
        Commands::Completion { shell } => {
            generate(*shell, &mut Cli::command(), "targon", &mut std::io::stdout());
            return Ok(());
        }
        _ => {}
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
        .base_url(resolved.base_url.clone())
        .build()?;
    let ctx = Context::new(
        client,
        format,
        resolved.profile,
        resolved.base_url,
        resolved.project,
    );

    match &cli.command {
        Commands::Auth(_) | Commands::Completion { .. } => unreachable!("handled above"),
        Commands::Workload(cmd) => commands::workload::handle(&ctx, cmd).await,
        Commands::Rental(cmd) => commands::rental::handle(&ctx, cmd).await,
        Commands::Vm(cmd) => commands::vm::handle(&ctx, cmd).await,
        Commands::Volume(cmd) => commands::volume::handle(&ctx, cmd).await,
        Commands::SshKey(cmd) => commands::ssh_key::handle(&ctx, cmd).await,
        Commands::Project(cmd) => commands::project::handle(&ctx, cmd).await,
        Commands::Inventory(cmd) => commands::inventory::handle(&ctx, cmd).await,
        Commands::Whoami => commands::whoami::handle(&ctx).await,
        Commands::Version => commands::version::handle(&ctx).await,
    }
}
