use clap::{Parser, Subcommand};
use clap_complete::Shell;

use crate::commands::api_token::ApiTokenCommands;
use crate::commands::auth::AuthCommands;
use crate::commands::bm::BmCommands;
use crate::commands::inventory::InventoryArgs;
use crate::commands::member::MemberCommands;
use crate::commands::org::OrgCommands;
use crate::commands::project::ProjectCommands;
use crate::commands::service_token::ServiceTokenCommands;
use crate::commands::ssh_key::SshKeyCommands;
use crate::commands::vm::VmCommands;
use crate::commands::workload::WorkloadCommands;

const BM_ABOUT: &str = "Deploy bare metal servers";
const BM_LONG_ABOUT: &str = "Deploy bare metal servers

Bare metal servers are workloads: manage them with
`targon workload <get|state|delete|events> <UID>`.
Changing the OS image, SSH keys or hostname is a reinstall - use
`targon bm reinstall`, which wipes every disk on the machine.
`bm power-off` keeps the server claimed and billed; use
`targon workload delete` to release it and stop billing.";

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
    /// Organization slug for this command
    #[arg(long, global = true)]
    pub org: Option<String>,
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
    /// Manage organizations and active organization context
    #[command(subcommand)]
    Org(OrgCommands),
    /// Manage members of the selected organization
    #[command(subcommand)]
    Member(MemberCommands),
    /// Manage personal API tokens
    #[command(subcommand, name = "api-token")]
    ApiToken(ApiTokenCommands),
    /// Manage service tokens in the selected organization
    #[command(subcommand, name = "service-token")]
    ServiceToken(ServiceTokenCommands),
    /// Manage any workload by UID
    #[command(subcommand, alias = "wl")]
    Workload(Box<WorkloadCommands>),
    /// Deploy confidential VMs
    #[command(subcommand, about = VM_ABOUT, long_about = VM_LONG_ABOUT)]
    Vm(Box<VmCommands>),
    /// Deploy bare metal servers
    #[command(subcommand, alias = "baremetal", about = BM_ABOUT, long_about = BM_LONG_ABOUT)]
    Bm(Box<BmCommands>),
    /// Manage SSH keys
    #[command(subcommand, alias = "key")]
    SshKey(SshKeyCommands),
    /// Manage projects
    #[command(subcommand, alias = "proj")]
    Project(ProjectCommands),
    /// Browse available inventory
    Inventory(InventoryArgs),
    /// List active inventory regions
    #[command(alias = "regions")]
    Region,
    /// Show wallet, credits, and active profile
    Whoami,
    /// Show CLI and API versions
    Version,
    /// Generate shell completions
    Completion {
        #[arg(value_enum)]
        shell: Shell,
    },
}
