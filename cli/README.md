# Targon CLI

Command-line interface to interact with Targon workloads.

Written in Rust and distributed as a standalone binary (the CLI is no longer
shipped with the Python package).

## Build

```bash
cargo build --release
```

The binary is produced at `target/release/targon`.

## Usage

```bash
targon --help
```

The v3 API scopes workloads, projects, volumes, and SSH keys to an
organization. Authenticate, select an organization, and optionally select a
default project:

```bash
targon auth login
targon org list
targon org use acme
targon project list
targon project use proj_123
targon vm deploy --name dev --image ubuntu --resource gpu-small
```

The organization is resolved in this order:

1. `--org <slug>` for a one-off command
2. `TARGON_ORG`
3. The organization saved by `targon org use <slug>` for the active profile

Projects are remembered separately for each organization. A command that
requires an organization prints a setup hint if none is selected.

```bash
# One-off organization override
targon --org personal workload list

# Organization administration
targon org get
targon member list
targon service-token create ci

# Personal tokens are user-scoped and do not require an organization
targon api-token list
```

Run `targon <command> --help` for all options. Use `--json` for
machine-readable output.

## License

Apache 2.0 — see [LICENSE](../LICENSE) for details.
