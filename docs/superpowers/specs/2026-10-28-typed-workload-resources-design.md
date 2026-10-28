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

The common `Workloads` service is limited to operations valid for every
workload type:

- Get
- List
- Delete
- Get state
- List events

It does not create or deploy workloads and does not expose type-specific
actions.

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
- Password
- Optional project
- Optional reservation
- SSH keys
- TCP/UDP ports

VM methods are limited to routes supported for VM workloads, such as deploy,
reboot, power operations, and reinstall.

### Bare Metal

Bare Metal registration accepts only Bare Metal-supported fields, including:

- Name
- Bare Metal image
- Resource
- Region
- At least one SSH key
- Optional hostname
- Optional project

Container commands, environment variables, registry credentials, and exposed
container ports are not accepted.

Bare Metal methods are limited to routes supported for Bare Metal workloads.

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
- Generic deploy and type-specific mutation methods
- Generic streamed workload exec
- Suspend
- Volume clients and services
- Volume models and workload volume fields
- Volume attach/detach methods
- Volume examples and exports
- Request fields accepted only by deprecated container workload types

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

Workloads.Get/List/State/Events/Delete
  → unchanged common Workloads operations

Volumes and workload volume attachment
  → removed; no current API equivalent
```

Examples must demonstrate fetched resources and consistent `Create` behavior.

## Testing

### Contract tests

- VM register, deploy, create, no-wait, and type mismatch
- Bare Metal register, deploy, create, required region/SSH keys, and type
  mismatch
- Sandbox create behavior remains unchanged
- Common workload reads handle VM, Bare Metal, and Sandbox responses
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
