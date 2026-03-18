# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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