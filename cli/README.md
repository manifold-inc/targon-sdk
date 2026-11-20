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

The v3 API scopes workloads, projects, and SSH keys to an
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

## Bare metal

Bare metal servers are region-scoped and always require at least one SSH key —
there is no password path, and the keys are injected via cloud-init at deploy.

```bash
targon region
targon inventory --type bm --region aus --available
targon bm images
targon bm deploy \
  --name training-node-1 \
  --image ubuntu-24.04-noble \
  --resource bm-h100-sxm5-8x \
  --region aus \
  --ssh-key sshkey_2ac93b6f
```

Run `targon bm deploy` with no flags on a terminal to pick the region, image,
hardware class and SSH keys interactively.

Servers are workloads, so `targon workload get|state|events|delete <UID>`
applies. The BM-specific verbs live under `targon bm`:

```bash
targon bm list
targon bm reinstall wrk_123 --image rocky-9.4   # wipes every disk
targon bm power-off wrk_123                     # stays claimed and billed
targon bm power-on wrk_123
targon bm reboot wrk_123                        # out-of-band power cycle
targon workload delete wrk_123                  # release and stop billing
```

Changing the OS image, SSH keys or hostname on a running server is a reinstall —
`targon bm reinstall` is the only path, and it destroys all data on the machine.
Add `--reservation <UID>` to `bm deploy` to place the server on one of the
organization's reservations instead of on-demand capacity.

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
