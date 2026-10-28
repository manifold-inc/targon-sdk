# Typed Workload Resources Design

## Goal

Replace stale generic workload and volume APIs in the Go and Python SDKs with
typed VM, Bare Metal, and Sandbox resources that match the public v3 API.

The redesign must:

- Remove deprecated Rental and Volume functionality.
- Add missing Bare Metal support.
- Make create/deploy behavior consistent across workload types.
- Prevent callers from constructing invalid combinations of type-specific
  fields.
- Preserve a small common workload API for reads and deletion.
- Clarify user-scoped and organization-scoped client services.

This is a breaking release-candidate change. No compatibility aliases or
deprecated shims will be added.

## Public object model

### Shared workload data

`Workload` remains the common response model containing fields shared by all
workload types. Typed resources wrap or embed this model:

```text
VM
└── Workload

BareMetal
└── Workload

Sandbox
└── Workload
```

Each typed resource is bound to the service that fetched or created it. This
allows resource methods without storing credentials directly on the resource.

### Common workload service

The common `Workloads` service is limited to operations the API accepts for
every workload type:

- Get
- List
- Delete
- Get state
- List events
- Wait until ready (polls state until `running`; fails on `error`, `suspended`, or `deleted`; `stopped`/`frozen` can still transition and keep waiting)
- Verify (`POST /workloads/verify` with `{uid, digest}`)
- Attach and detach SSH keys (`PUT`/`DELETE /workloads/{uid}/ssh-keys/{key}`)

It does not create, update, deploy, suspend, or exec workloads and does not
expose type-specific actions.

Common response models mirror the backend `Workload` and
`WorkloadOperationResponse` shapes, including `reservation_uid`, `created_by`,
`vm_config {hostname}`, and `bm_config {region, hostname}`. There are no
top-level `region` or `hostname` fields.

### Typed services

The client exposes:

- `VMs`
- `BareMetal`
- `Sandboxes`

Python uses equivalent snake_case client properties.

Every typed service validates the workload discriminator before hydrating a
typed resource. A VM UID cannot be returned as a Sandbox or Bare Metal
resource.

## Create and deploy lifecycle

All typed services expose the same lifecycle concepts:

### Register

`Register` sends `POST /workloads` with a type-specific request and returns a
hydrated typed resource in its registered state.

### Deploy

`Deploy` sends `POST /workloads/{uid}/deploy`, then returns the hydrated typed
resource. It accepts wait options so callers can return immediately or wait for
the target running state.

### Create

`Create` is the ergonomic convenience:

```text
Register → Deploy → optional wait
```

It returns the typed resource.

Resource objects expose equivalent methods where meaningful:

```text
vm.Deploy(...)
bareMetal.Deploy(...)
sandbox.Freeze(...)
```

## Type-specific requests and methods

### VM

VM registration accepts only VM-supported fields, including:

- Name
- VM image
- Resource
- Optional `vm_config` (`VMConfigInput {hostname?}`, sent as
  `vm_config.hostname`)
- Optional project
- Optional reservation
- SSH keys
- TCP/UDP ports (port 22 is reserved for SSH and rejected)

There is no VM password. Blank optional strings are omitted from the payload.

VM methods are limited to routes the API supports for VM workloads: deploy,
update (name and ports), reboot, logs, streamed logs, and VM images
(`GET /workloads/vm-images`). Power on/off and reinstall are Bare Metal-only
in the API and are not exposed on VMs.

### Bare Metal

Bare Metal registration accepts only Bare Metal-supported fields, including:

- Name
- Bare Metal image
- Resource
- Region
- At least one SSH key
- Optional `bm_config` (`BMConfigInput {hostname?}`, sent as
  `bm_config.hostname`)
- Optional project
- Optional reservation

Container commands, environment variables, registry credentials, and exposed
container ports are not accepted.

Bare Metal methods are limited to routes the API supports for Bare Metal
workloads: deploy, reboot (power cycle), power on, power off, reinstall (image
required; SSH keys optional and reused when omitted; optional hostname), and
Bare Metal images (`GET /workloads/bm-images`). Bare Metal `Create` and
`Deploy` wait up to 30 minutes when the caller does not set a timeout.

### Inventory

Inventory accepts the current `vm` and `bm` types, a GPU filter, and a region
filter (lowercase slug, at most 32 characters). The deprecated `rental`,
`storage`, and `serverless` types fail validation before a request is sent.

### Sandbox

Sandbox keeps its existing fetched-template resource model and specialized
capabilities:

- Exec
- Files
- Terminal sessions
- Access tickets
- Desktop metadata
- Freeze/thaw
- Fork
- Publish

Sandbox `Create` retains its existing register/deploy/wait behavior but shares
the same lifecycle naming and wait conventions as VM and Bare Metal.

## Removed APIs

Remove from both Go and Python where present:

- Rental workload type and defaults
- Generic workload creation
- Generic update
- Generic deploy and type-specific mutation methods
- Generic streamed workload exec
- Suspend
- VM password (`vm_config.password`); the API accepts only
  `vm_config.hostname`
- Volume clients and services
- Volume models and workload volume fields
- Volume attach/detach methods
- Volume examples and exports
- Request fields accepted only by deprecated container workload types
- Deprecated `rental`, `storage`, and `serverless` inventory types

Moved rather than removed, because the API rejects them for other workload
types:

- Workload logs and streamed logs → VM service (the API returns 501 for
  non-VM workloads)
- VM images → VM service
- Reboot → VM and Bare Metal services

The common response model retains only fields returned by current workload
routes.

## Organization scope

No organization object hierarchy is introduced in this change.

Documentation and API comments must clearly distinguish:

- User-scoped: organization discovery/management and personal API tokens
- Organization-bound: typed workloads, projects, SSH keys, members, service
  tokens, wallet, and credits

`ForOrg` remains the supported way to create an organization-bound client
view.

## Error and retry behavior

- Preserve each SDK's existing base API, authentication, validation, timeout,
  and network errors.
- Return validation errors before network calls for invalid type-specific
  fields.
- Preserve machine-readable backend reasons.
- Do not automatically retry mutations whose first attempt may have succeeded.
- Safe reads and polling may retain bounded retries.
- Typed hydration failures must identify the expected and actual workload
  types.

## Migration guidance

Document these replacements:

```text
Workloads.Create + Workloads.Deploy
  → VMs.Create or BareMetal.Create

Workloads.Create with type SANDBOX
  → Sandboxes.Create

Workloads.Get/List/State/Events/Delete/WaitUntilReady/Verify
Workloads.AttachSSHKey/DetachSSHKey
  → unchanged common Workloads operations

VMConfig{Password}
  → removed; use SSH keys and optional VMConfigInput{Hostname}

Workloads.GetLogs/StreamLogs and Workloads.VMImages
  → VMs.Logs/StreamLogs and VMs.Images

Workloads.Reboot
  → VMs.Reboot or BareMetal.Reboot

Top-level Workload.Region/Hostname
  → BMConfig.Region/Hostname or VMConfig.Hostname

Volumes and workload volume attachment
  → removed; no current API equivalent
```

Examples must demonstrate fetched resources and consistent `Create` behavior.

## Testing

### Contract tests

- VM register, deploy, create, no-wait, and type mismatch
- VM payload uses `vm_config.hostname` and never contains a password
- VM logs, streamed logs, and `vm-images`
- Bare Metal register, deploy, create, required region/SSH keys, and type
  mismatch
- Bare Metal payload uses `bm_config.hostname` and `reservation_uid`
- Bare Metal `bm-images` and the 30-minute default wait
- Sandbox create behavior remains unchanged
- Common workload reads handle VM, Bare Metal, and Sandbox responses in the
  backend shape (`reservation_uid`, `created_by`, `vm_config`, `bm_config`)
- Common wait-until-ready, verify, and SSH key attach/detach routes
- Inventory sends `type` (`vm`/`bm`), `gpu`, and `region`, and rejects
  deprecated types
- Deprecated workload types fail before network calls
- Removed volume and legacy mutation symbols are not exported

### Regression tests

- Request JSON matches the public API contract
- Mutation retry rules remain safe
- Context, request, exec, file, and polling timeouts retain current behavior
- Organization binding is preserved through `ForOrg`

### Verification

- Go tests, vet, formatting, examples, and package build
- Python tests, Black, isort, flake8, examples, wheel, and sdist
- Public documentation checks and production build
- Live smoke tests for VM, Bare Metal, and Sandbox when credentials and
  capacity are available

## Out of scope

- TypeScript VM or Bare Metal support
- A new bound Organization resource hierarchy
- Volume replacement APIs
- Compatibility shims for Rental or Volume
- Changes to backend routes
