use clap::{Parser, Subcommand};
use clap_complete::Shell;

use crate::commands::auth::AuthCommands;
use crate::commands::inventory::InventoryArgs;
use crate::commands::project::ProjectCommands;
use crate::commands::rental::RentalCommands;
use crate::commands::ssh_key::SshKeyCommands;
use crate::commands::vm::VmCommands;
use crate::commands::volume::VolumeCommands;
use crate::commands::workload::WorkloadCommands;

const VM_ABOUT: &str = "Deploy confidential VMs";
const VM_LONG_ABOUT: &str = "Deploy confidential VMs

VMs are workloads: manage them with `targon workload <get|delete|logs|...> <UID>`.
SSH keys are fixed at creation - pass --ssh-key to `vm deploy`; they cannot be
attached or detached after the VM boots.";

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
    /// Manage any workload by UID
    #[command(subcommand, alias = "wl")]
    Workload(Box<WorkloadCommands>),
    /// Deploy container rentals
    #[command(subcommand)]
    Rental(Box<RentalCommands>),
    /// Deploy confidential VMs
    #[command(subcommand, about = VM_ABOUT, long_about = VM_LONG_ABOUT)]
    Vm(Box<VmCommands>),
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
    /// Show wallet, credits, and active profile
    Whoami,
    /// Show the API version
    Version,
    /// Generate shell completions
    Completion {
        #[arg(value_enum)]
        shell: Shell,
    },
}
