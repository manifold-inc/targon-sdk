# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Component releases use prefixed tags (`cli/vX.Y.Z`, `python/vX.Y.Z`, …).

## [Unreleased]

### Fixed
- Go request deadlines now come from the call context, and TypeScript applies
  operation-specific deadlines. Exec and file transfer are no longer cut off
  by the 30-second ordinary request timeout.

### Added
- Added typed VM and Bare Metal services to Go and Python, alongside the typed
  Sandbox services in Go, Python, and TypeScript.
- Added consistent `Create` convenience operations and advanced
  `Register`/`Deploy` lifecycle operations for typed workloads.
- Added `reservation_uid` and nested `vm_config.hostname` /
  `bm_config.hostname` to typed VM and Bare Metal registration, and
  `reservation_uid`, `created_by`, `vm_config`, and `bm_config` to common
  workload responses.
- Added VM `Logs`/`StreamLogs` (`get_logs`/`stream_logs`) and `Images`
  (`vm-images`), and Bare Metal `Images` (`bm-images`) on the typed services.
- Added a `region` filter and the current `vm`/`bm` types to inventory.
- Added first-class Sandbox APIs to the Go, Python, and TypeScript SDKs.
- Added fetched, client-bound `SandboxTemplate` resources with refresh, update,
  delete, and sandbox-creation methods.
- Added sandbox lifecycle, fork/publish, exec, binary file transfer, access
  tickets, terminal session management, raw-binary terminal WebSockets, and
  desktop metadata.
- Added typed sandbox errors, strict client validation, cursor pagination, and
  method-bearing `Sandbox` resources.
- Added the zero-runtime-dependency TypeScript SDK and optional terminal
  transport for Python.
- Added mock-backed Sandbox regression suites in all three SDKs.

### Changed
- Python `ssh_keys` must be a sequence (tuple or list) of non-empty SSH key UID
  strings. Bare strings, bytes, sets, generators, and blank or non-string
  entries are rejected with a `ValidationError` before any request, instead of
  being split into characters. Go rejects blank SSH key UIDs and trims
  surrounding whitespace. This applies to VM and Bare Metal create, Bare Metal
  reinstall, and Sandbox create/update.
- Common `wait_until_ready` / `WaitUntilReady` fails only on `error`,
  `suspended`, and `deleted`. `stopped` and `frozen` are not terminal because
  they can transition later; Sandbox target-specific waits are unchanged.
- Prepared coordinated `4.0.0-rc.1` releases for Go, Python, and TypeScript.
- In Go and Python, generic workload creation now migrates to typed
  `VMs.Create`, `BareMetal.Create`, or `Sandboxes.Create` operations. Common
  workload list, get, state, events, delete, wait-until-ready, verify, and SSH
  key attach/detach remain available in those SDKs. TypeScript currently
  exposes Sandbox only.
- Moved workload logs and VM images from the common Workloads service to the
  typed VM service, and workload reboot to the typed VM and Bare Metal
  services, because the API rejects them for other workload types.
- Bare Metal `Create`/`Deploy` wait up to 30 minutes by default.
- Bare Metal reinstall SSH keys are optional; omitted keys reuse the attached
  set.
- Workload responses no longer have top-level `region` or `hostname`; read
  `bm_config.region`, `bm_config.hostname`, or `vm_config.hostname`.
- Python `VM` and `BareMetal` expose workload fields through explicit
  read-only properties instead of attribute forwarding.
- Clarified that organization discovery/management, personal API tokens, and
  inventory are user-scoped and organization-optional, while typed workloads,
  projects, SSH keys, members, service tokens, wallet, and credits are
  organization-bound.
- Sandbox creation now requires a fetched `READY` `SandboxTemplate` resource;
  raw template UIDs are not accepted by the public create APIs.
- Sandbox mutations do not retry automatically, preventing duplicate forks,
  publishes, execs, or misleading replay failures. Safe reads retain retries.
- Grouped Python and TypeScript file and terminal operations under bound
  `sandbox.files` and `sandbox.terminals` capabilities, removing redundant
  pre-release aliases.
- Simplified resource binding, validation, path construction, and lifecycle
  polling across the public SDKs.

### Removed
- Removed the legacy RENTAL-backed Python Sandbox shim and its workload
  streaming-exec behavior.
- Removed Rental and Volume APIs from Go and Python. They have no current v4
  replacement.
- Removed generic workload mutation APIs (create, update, deploy, suspend,
  exec) in favor of the typed VM, Bare Metal, and Sandbox services.
- Removed the VM password (`vm_config.password`); the API accepts only
  `vm_config.hostname`.
- Removed the `rental`, `storage`, and `serverless` inventory types.

## [python/3.0.0] - 2026-07-27
### Added
- Added bound organization context through `Client(org=...)`, `TARGON_ORG`, active CLI profiles, and `Client.for_org()`.
- Added organization, member, wallet, credits, personal API token, and organization service token clients.
- Added mocked request and configuration regression tests for the Python SDK.

### Changed
- Migrated Python workloads, projects, volumes, SSH keys, and inventory from `/tha/v2` to `/tha/v3`.
- Made custom API hosts and transport configuration apply consistently to every resource client.
- Expanded workload lifecycle support with suspend, reboot, VM image, and VM log-type operations.

### Removed
- Removed the v2 `client.user` API-key facade, API-key rotation, and volume deployment deletion endpoint.
- Templates, experiments, and deployment-type header configuration remain intentionally unsupported.

## [cli/3.0.0] - 2026-07-27
### Added
- Added explicit organization context through `--org`, `TARGON_ORG`, and `targon org use`, with per-organization default projects.
- Added organization and member management commands.
- Added personal API token and organization service token commands.
- Added SSH key and volume update commands, workload digest verification, and cursor pagination options.

### Changed
- Migrated the Rust CLI from `/tha/v2` to the organization-scoped `/tha/v3` API.
- Moved wallet and credit reporting to the selected organization and included organization context in `whoami` and `auth status`.
- Renamed the workload list filter to `--status`; `--state` remains available as a compatibility alias.

### Removed
- Removed v2 personal API-key rotation and unscoped `app_id` assumptions.

## [cli/2.1.2] - 2026-07-15
Standalone Rust CLI release (`targon` binary). Install via Homebrew (`brew tap manifold-inc/tap && brew install targon`) or build from `cli/`.

## [1.0.0] - 2026-04-18
### Added
- Added workload state and events support via `targon get state <wrk-uid>` and `targon get events <wrk-uid>`.
- Added workload-backed logs support with follow and non-follow modes.
- Added workload deletion support for individual function workloads.
- Added config/container deployment support for v2 workload-backed serverless resources, including the ability to inspect deployed workloads with logs, state, and events.

### Changed
- Migrated app management APIs to the v2 `/tha/v2/apps` endpoints.
- Migrated function registration to the v2 workload model so functions are created, updated, listed, fetched, and deleted as workloads.
- Migrated publish/deploy flows to the v2 workload deploy endpoints and updated deployment responses to include workload state, revision, URLs, and cost data.
- Migrated serverless container management to the v2 workload endpoints for create, list, delete, logs, and state operations.
- Updated config-based container deployment flows and CLI output to reflect workload-backed URLs, status, and hourly cost.
- Expanded the `targon capacity` output with richer inventory information ahead of its planned rename to `targon inventory`.

## [0.5.0] - 2026-04-16
- Added the support for RTX Pro 6000 Blackwell gpus.

## [0.4.0] - 2026-10-26
- Added the support for B200s gpus.

## [0.3.1] - 2026-01-20
- Added the support for H100s gpus.
