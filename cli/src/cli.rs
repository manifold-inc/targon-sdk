use clap::{Parser, Subcommand};

use crate::commands::auth::AuthCommands;
use crate::commands::inventory::InventoryArgs;
use crate::commands::project::ProjectCommands;
use crate::commands::ssh_key::SshKeyCommands;
use crate::commands::user::UserCommands;
use crate::commands::volume::VolumeCommands;
use crate::commands::workload::WorkloadCommands;

#[derive(Debug, Parser)]
#[command(
    name = "targon",
    version,
    about = "Interact with Targon workloads",
    propagate_version = true
)]
pub struct Cli {
    /// Configuration profile to use
    #[arg(long, global = true)]
    pub profile: Option<String>,
    /// Override the API base URL
    #[arg(long = "base-url", global = true)]
    pub base_url: Option<String>,
    /// Emit machine-readable JSON
    #[arg(long, global = true)]
    pub json: bool,
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Manage authentication
    #[command(subcommand)]
    Auth(AuthCommands),
    /// Manage workloads
    #[command(subcommand, alias = "wl")]
    Workload(Box<WorkloadCommands>),
    /// Manage volumes
    #[command(subcommand, alias = "vol")]
    Volume(VolumeCommands),
    /// Manage SSH keys
    #[command(subcommand, alias = "key")]
    SshKey(SshKeyCommands),
    /// Manage projects
    #[command(subcommand, alias = "proj")]
    Project(ProjectCommands),
    /// Browse available inventory
    Inventory(InventoryArgs),
    /// Show wallet and credits
    #[command(subcommand)]
    User(UserCommands),
    /// Show the API version
    Version,
}
