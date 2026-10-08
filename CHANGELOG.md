# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Component releases use prefixed tags (`cli/vX.Y.Z`, `python/vX.Y.Z`, …).

## [Unreleased]

### Added
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
- Prepared coordinated `4.0.0-rc.1` releases for Go, Python, and TypeScript.
- Sandbox creation now requires a fetched `READY` `SandboxTemplate` resource;
  raw template UIDs are not accepted by the public create APIs.
- Sandbox mutations do not retry automatically, preventing duplicate forks,
  publishes, execs, or misleading replay failures. Safe reads retain retries.

### Removed
- Removed the legacy RENTAL-backed Python Sandbox shim and its workload
  streaming-exec behavior.

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
