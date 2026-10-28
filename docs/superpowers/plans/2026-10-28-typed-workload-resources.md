# Typed Workload Resources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stale generic workload and volume APIs in Go and Python with
typed VM, Bare Metal, and Sandbox resources that consistently support
register, deploy, and create.

**Architecture:** Keep `Workloads` as a small common read/delete service.
Expose `VMs`, `BareMetal`, and `Sandboxes` as typed services returning
method-bearing resources that contain the shared workload response. `Create`
is the high-level register/deploy/wait operation; `Register` and `Deploy`
remain available for advanced callers.

**Tech Stack:** Go 1.24, Python 3.9+, `net/http`, `requests`, pytest, Go
`httptest`, GitHub Actions.

## Global Constraints

- Public API is v4 release-candidate quality; breaking cleanup is allowed.
- Do not add compatibility aliases for Rental or Volume APIs.
- Do not change TypeScript workload support.
- Do not add runtime dependencies.
- Preserve existing timeout, typed error, and mutation retry behavior.
- Keep `cli/**` unchanged.
- Do not expose implementation-specific backend names or topology.

---

## File structure

### Go

- `libs/go/workload.go` — common workload response/read operations only
- `libs/go/workload_lifecycle.go` — private typed register/deploy/wait helpers
- `libs/go/vm.go` — VM request, service, resource, and VM-specific methods
- `libs/go/bare_metal.go` — Bare Metal request, service, resource, and methods
- `libs/go/sandbox.go` — existing typed Sandbox service using shared lifecycle
- `libs/go/client.go` — register typed services; remove Volumes
- Delete `libs/go/volume.go`

### Python

- `libs/python/src/targon/client/workload.py` — common workload models/read API
- `libs/python/src/targon/client/_workload_lifecycle.py` — shared typed helpers
- `libs/python/src/targon/client/vm.py` — VM models/resource/client
- `libs/python/src/targon/client/bare_metal.py` — Bare Metal models/resource/client
- `libs/python/src/targon/client/client.py` — expose typed clients; remove volume
- Delete `libs/python/src/targon/client/volume.py`

---

### Task 1: Lock backend parity with failing Go tests

**Files:**
- Create: `libs/go/typed_workload_test.go`
- Modify: `libs/go/workload.go`
- Test: `libs/go/typed_workload_test.go`

**Interfaces:**
- Produces the expected valid types: `VM`, `BM`, `SANDBOX`
- Produces the deprecated types: `RENTAL`, `SERVERLESS`, `INFERENCE`, `FUNCTION`
- Establishes that common `Workloads` no longer creates or deploys resources

- [ ] **Step 1: Add failing type-contract tests**

```go
func TestCurrentWorkloadTypes(t *testing.T) {
	for _, workloadType := range []string{"VM", "BM", "SANDBOX"} {
		if _, ok := validWorkloadTypes[workloadType]; !ok {
			t.Errorf("%s should be valid", workloadType)
		}
	}
	for _, workloadType := range []string{
		"RENTAL", "SERVERLESS", "INFERENCE", "FUNCTION",
	} {
		if _, ok := deprecatedWorkloadTypes[workloadType]; !ok {
			t.Errorf("%s should be deprecated", workloadType)
		}
	}
}
```

- [ ] **Step 2: Add API-surface compile assertions**

The test package must compile only against these common methods:

```go
type commonWorkloads interface {
	Get(context.Context, string) (*Workload, error)
	List(context.Context, ListWorkloadsParams) (List[WorkloadListItem], error)
	Delete(context.Context, string) error
	GetState(context.Context, string) (*WorkloadStateResponse, error)
	GetEvents(context.Context, string, Page) (List[WorkloadEvent], error)
}
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```bash
go -C libs/go test ./...
```

Expected: type assertions fail because Rental is currently valid and BM is
missing.

- [ ] **Step 4: Correct the type sets**

```go
validWorkloadTypes = map[string]struct{}{
	"VM": {}, "BM": {}, "SANDBOX": {},
}
deprecatedWorkloadTypes = map[string]struct{}{
	"RENTAL": {}, "SERVERLESS": {}, "INFERENCE": {}, "FUNCTION": {},
}
```

- [ ] **Step 5: Run the focused tests**

Run:

```bash
go -C libs/go test ./... -run 'TestCurrentWorkloadTypes'
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/go/workload.go libs/go/typed_workload_test.go
git commit -m "test: lock current workload types"
```

---

### Task 2: Remove deprecated Go Volume and generic mutation APIs

**Files:**
- Delete: `libs/go/volume.go`
- Modify: `libs/go/client.go`
- Modify: `libs/go/workload.go`
- Modify: `libs/go/client_test.go`
- Modify: `libs/go/README.md`

**Interfaces:**
- `Client.Volumes` is removed
- `Workload` and `WorkloadListItem` no longer contain volume fields
- `WorkloadService` contains only common read/delete/state/event methods

- [ ] **Step 1: Add removal assertions**

Use an external-package compile test that documents the supported surface:

```go
func TestCommonWorkloadServiceSurface(t *testing.T) {
	var _ commonWorkloads = (*targon.WorkloadService)(nil)
}
```

Remove tests that exercise Volume, attach/detach Volume, generic exec, suspend,
generic deploy, and generic create.

- [ ] **Step 2: Run Go tests and record compile failures**

Run:

```bash
go -C libs/go test ./...
```

Expected: failures reference the APIs scheduled for removal.

- [ ] **Step 3: Remove stale symbols**

Remove:

```text
Client.Volumes
VolumeService and all volume models
VolumeMount and WorkloadVolumeMount
Workload.Volumes and WorkloadListItem.Volumes
WorkloadService.Create
WorkloadService.Deploy
WorkloadService.Suspend
WorkloadService.Exec and ExecStream
WorkloadService.AttachVolume and DetachVolume
WorkloadService.VMImages
```

Keep common `Get`, `List`, `Update` only if update is moved to typed services
within the same task; otherwise remove common `Update`.

- [ ] **Step 4: Remove unsupported request fields**

Delete generic create request types and container-only fields that no longer
belong to a public current workload create API.

- [ ] **Step 5: Run tests, vet, and formatting**

```bash
gofmt -w libs/go
go -C libs/go test ./...
go -C libs/go vet ./...
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A libs/go
git commit -m "refactor: remove deprecated Go workload APIs"
```

---

### Task 3: Add shared Go typed lifecycle

**Files:**
- Create: `libs/go/workload_lifecycle.go`
- Modify: `libs/go/workload.go`
- Modify: `libs/go/sandbox.go`
- Test: `libs/go/typed_workload_test.go`

**Interfaces:**

```go
type typedWorkloadService interface {
	path(parts ...string) (string, error)
	get(context.Context, string) (*Workload, error)
}

func registerTyped(
	ctx context.Context,
	client *Client,
	payload any,
	expectedType string,
) (*Workload, error)

func deployTyped(
	ctx context.Context,
	client *Client,
	uid string,
	expectedType string,
	wait WaitOptions,
) (*Workload, error)
```

- [ ] **Step 1: Add failing lifecycle tests**

Cover:

```text
Register posts once and GETs the full resource.
Deploy posts /deploy and returns immediately when NoWait is true.
Deploy polls state and returns running when waiting.
Hydration rejects a mismatched type.
Mutations use the no-retry transport.
```

- [ ] **Step 2: Run focused tests**

```bash
go -C libs/go test ./... -run 'TestTyped(Register|Deploy|TypeMismatch)'
```

Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Implement private lifecycle helpers**

Use existing path, timeout, polling, no-retry, and typed validation helpers.
Do not add a generic public typed-workload interface.

- [ ] **Step 4: Move Sandbox create/deploy internals to the helpers**

`SandboxesService.Register`, `Deploy`, and `Create` must use the common helper
while preserving template validation and Sandbox-specific wait errors.

- [ ] **Step 5: Run all Go tests**

```bash
go -C libs/go test ./...
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/go/workload_lifecycle.go libs/go/workload.go \
  libs/go/sandbox.go libs/go/typed_workload_test.go
git commit -m "refactor: share typed workload lifecycle"
```

---

### Task 4: Add Go VM resource and service

**Files:**
- Create: `libs/go/vm.go`
- Create: `libs/go/vm_test.go`
- Modify: `libs/go/client.go`
- Modify: `libs/go/workload.go`

**Interfaces:**

```go
type VMCreateParams struct {
	Name           string
	Image          string
	ResourceName   string
	Password       string
	ProjectID      string
	ReservationUID string
	SSHKeys        []string
	Ports          []PortConfig
	Wait           WaitOptions
}

type VMUpdateParams struct {
	Name  *string
	Ports *[]PortConfig
}

type VM struct {
	Workload
	service *VMService
}

type VMService struct { client *Client }

func (s *VMService) Register(context.Context, VMCreateParams) (*VM, error)
func (s *VMService) Deploy(context.Context, string, WaitOptions) (*VM, error)
func (s *VMService) Create(context.Context, VMCreateParams) (*VM, error)
func (s *VMService) Get(context.Context, string) (*VM, error)
func (s *VMService) Update(context.Context, string, VMUpdateParams) (*VM, error)
func (s *VMService) Reboot(context.Context, string) (*VM, error)
func (s *VMService) Logs(context.Context, string, LogOptions) (string, error)
```

- [ ] **Step 1: Add failing VM tests**

Assert exact create JSON:

```json
{
  "type": "VM",
  "name": "vm-example",
  "image": "ubuntu-24.04",
  "resource_name": "cpu-small",
  "vm_config": {"password": "secret"}
}
```

Cover password validation, update field restrictions, type mismatch,
register-only, create/deploy/wait, and resource forwarding methods.

- [ ] **Step 2: Run focused tests**

```bash
go -C libs/go test ./... -run TestVM
```

Expected: FAIL because VM types do not exist.

- [ ] **Step 3: Implement VM request and resource**

Build payloads explicitly; do not reuse a generic map of container fields.

- [ ] **Step 4: Register `Client.VMs`**

```go
VMs *VMService
```

Initialize it in `Client.initServices`.

- [ ] **Step 5: Run tests and vet**

```bash
go -C libs/go test ./...
go -C libs/go vet ./...
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/go/vm.go libs/go/vm_test.go libs/go/client.go libs/go/workload.go
git commit -m "feat: add typed Go VM resources"
```

---

### Task 5: Add Go Bare Metal resource and service

**Files:**
- Create: `libs/go/bare_metal.go`
- Create: `libs/go/bare_metal_test.go`
- Modify: `libs/go/client.go`
- Modify: `libs/go/workload.go`

**Interfaces:**

```go
type BareMetalCreateParams struct {
	Name         string
	Image        string
	ResourceName string
	Region       string
	Hostname     string
	ProjectID    string
	SSHKeys      []string
	Wait         WaitOptions
}

type BareMetalReinstallParams struct {
	Image   string
	SSHKeys []string
	Hostname string
}

type BareMetal struct {
	Workload
	service *BareMetalService
}

func (s *BareMetalService) Register(context.Context, BareMetalCreateParams) (*BareMetal, error)
func (s *BareMetalService) Deploy(context.Context, string, WaitOptions) (*BareMetal, error)
func (s *BareMetalService) Create(context.Context, BareMetalCreateParams) (*BareMetal, error)
func (s *BareMetalService) Get(context.Context, string) (*BareMetal, error)
func (s *BareMetalService) Reboot(context.Context, string) (*BareMetal, error)
func (s *BareMetalService) PowerOn(context.Context, string) (*BareMetal, error)
func (s *BareMetalService) PowerOff(context.Context, string) (*BareMetal, error)
func (s *BareMetalService) Reinstall(context.Context, string, BareMetalReinstallParams) (*BareMetal, error)
```

- [ ] **Step 1: Add failing Bare Metal tests**

Assert exact registration JSON and reject missing region, missing SSH keys,
ports, environment variables, commands, registry credentials, and VM config.

- [ ] **Step 2: Run focused tests**

```bash
go -C libs/go test ./... -run TestBareMetal
```

Expected: FAIL because Bare Metal types do not exist.

- [ ] **Step 3: Implement Bare Metal resource/service**

Hydrate and verify `type == "BM"`. Reinstall uses
`POST /workloads/{uid}/reinstall`; power methods use `power-on` and
`power-off`.

- [ ] **Step 4: Register `Client.BareMetal`**

Initialize the service in `Client.initServices`.

- [ ] **Step 5: Run all Go verification**

```bash
gofmt -w libs/go
go -C libs/go test ./...
go -C libs/go vet ./...
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/go/bare_metal.go libs/go/bare_metal_test.go \
  libs/go/client.go libs/go/workload.go
git commit -m "feat: add typed Go Bare Metal resources"
```

---

### Task 6: Remove deprecated Python Volume and generic mutation APIs

**Files:**
- Delete: `libs/python/src/targon/client/volume.py`
- Modify: `libs/python/src/targon/client/client.py`
- Modify: `libs/python/src/targon/client/workload.py`
- Modify: `libs/python/src/targon/client/__init__.py`
- Modify: `libs/python/src/targon/__init__.py`
- Test: `libs/python/tests/test_workloads.py`

**Interfaces:**
- `Client.volume` is removed
- `WorkloadClient` keeps get/list/delete/state/events
- Generic create/deploy/suspend/exec/volume methods are removed

- [ ] **Step 1: Add failing export and surface tests**

```python
def test_deprecated_resources_are_not_exported():
    client = Client(api_key="test", org="acme")
    assert not hasattr(client, "volume")
    assert not hasattr(client.workload, "create")
    assert not hasattr(client.workload, "deploy")
    assert not hasattr(client.workload, "suspend")
    assert not hasattr(client.workload, "exec")
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
PYTHONPATH=libs/python/src \
  libs/python/.venv/bin/python -m pytest \
  libs/python/tests/test_workloads.py -v
```

Expected: FAIL because deprecated attributes exist.

- [ ] **Step 3: Remove deprecated files, properties, exports, and models**

Delete Volume classes and remove volume fields from workload responses.
Retain only current shared response fields.

- [ ] **Step 4: Run Python tests**

```bash
PYTHONPATH=libs/python/src \
  libs/python/.venv/bin/python -m pytest libs/python/tests
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A libs/python
git commit -m "refactor: remove deprecated Python workload APIs"
```

---

### Task 7: Add Python typed lifecycle and VM resources

**Files:**
- Create: `libs/python/src/targon/client/_workload_lifecycle.py`
- Create: `libs/python/src/targon/client/vm.py`
- Modify: `libs/python/src/targon/client/client.py`
- Modify: `libs/python/src/targon/client/workload.py`
- Test: `libs/python/tests/test_vm.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class VMCreateParams:
    name: str
    image: str
    resource_name: str
    password: str
    project_id: Optional[str] = None
    reservation_uid: Optional[str] = None
    ssh_keys: Sequence[str] = ()
    ports: Sequence[PortConfig] = ()
    wait_until_running: bool = True

@dataclass(frozen=True)
class VM:
    _client: Client
    workload: WorkloadResponse

class VMClient:
    def register(self, params: VMCreateParams) -> VM: ...
    def deploy(self, uid: str, *, wait=True, timeout=300, poll_interval=5) -> VM: ...
    def create(self, params: VMCreateParams, *, timeout=300, poll_interval=5) -> VM: ...
    def get(self, uid: str) -> VM: ...
    def update(self, uid: str, request: VMUpdateParams) -> VM: ...
    def reboot(self, uid: str) -> VM: ...
```

- [ ] **Step 1: Add failing VM payload/lifecycle tests**

Cover register, create, no-wait, mismatched type, password validation, updates,
reboot, and bound resource forwarding.

- [ ] **Step 2: Run focused tests**

```bash
PYTHONPATH=libs/python/src \
  libs/python/.venv/bin/python -m pytest libs/python/tests/test_vm.py -v
```

Expected: FAIL because `Client.vms` is not defined.

- [ ] **Step 3: Implement shared lifecycle helper**

The helper must POST registration with retries disabled, hydrate via GET,
deploy once, and use the existing workload state polling/error semantics.

- [ ] **Step 4: Implement and export VM client/resource**

Add lazy `Client.vms` initialization and top-level exports.

- [ ] **Step 5: Run Python verification**

```bash
PYTHONPATH=libs/python/src \
  libs/python/.venv/bin/python -m pytest libs/python/tests
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/python/src/targon/client/_workload_lifecycle.py \
  libs/python/src/targon/client/vm.py libs/python/src/targon/client/client.py \
  libs/python/src/targon/client/workload.py libs/python/tests/test_vm.py
git commit -m "feat: add typed Python VM resources"
```

---

### Task 8: Add Python Bare Metal resources

**Files:**
- Create: `libs/python/src/targon/client/bare_metal.py`
- Modify: `libs/python/src/targon/client/client.py`
- Modify: `libs/python/src/targon/client/__init__.py`
- Modify: `libs/python/src/targon/__init__.py`
- Test: `libs/python/tests/test_bare_metal.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class BareMetalCreateParams:
    name: str
    image: str
    resource_name: str
    region: str
    ssh_keys: Sequence[str]
    hostname: Optional[str] = None
    project_id: Optional[str] = None
    wait_until_running: bool = True

class BareMetalClient:
    def register(self, params: BareMetalCreateParams) -> BareMetal: ...
    def deploy(self, uid: str, *, wait=True, timeout=300, poll_interval=5) -> BareMetal: ...
    def create(self, params: BareMetalCreateParams, *, timeout=300, poll_interval=5) -> BareMetal: ...
    def get(self, uid: str) -> BareMetal: ...
    def reboot(self, uid: str) -> BareMetal: ...
    def power_on(self, uid: str) -> BareMetal: ...
    def power_off(self, uid: str) -> BareMetal: ...
    def reinstall(self, uid: str, request: BareMetalReinstallParams) -> BareMetal: ...
```

- [ ] **Step 1: Add failing Bare Metal tests**

Test exact JSON, required fields, type mismatch, register/deploy/create,
power/reboot/reinstall, and resource methods.

- [ ] **Step 2: Run focused tests**

```bash
PYTHONPATH=libs/python/src \
  libs/python/.venv/bin/python -m pytest \
  libs/python/tests/test_bare_metal.py -v
```

Expected: FAIL because Bare Metal support does not exist.

- [ ] **Step 3: Implement Bare Metal client/resource and exports**

Use the shared lifecycle helper and reject unsupported fields before requests.

- [ ] **Step 4: Run all Python tests and lint**

```bash
PYTHONPATH=libs/python/src \
  libs/python/.venv/bin/python -m pytest libs/python/tests
libs/python/.venv/bin/black --check libs/python/src libs/python/tests
libs/python/.venv/bin/isort --check-only libs/python/src libs/python/tests
libs/python/.venv/bin/flake8 \
  --config libs/python/.flake8 libs/python/src libs/python/tests
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/python/src/targon/client/bare_metal.py \
  libs/python/src/targon/client/client.py \
  libs/python/src/targon/client/__init__.py \
  libs/python/src/targon/__init__.py \
  libs/python/tests/test_bare_metal.py
git commit -m "feat: add typed Python Bare Metal resources"
```

---

### Task 9: Align Sandbox lifecycle and rewrite examples

**Files:**
- Modify: `libs/go/sandbox.go`
- Modify: `libs/go/sandbox_resource.go`
- Modify: `libs/python/src/targon/client/sandbox/client.py`
- Modify: `libs/python/src/targon/client/sandbox/resources.py`
- Modify: `examples/go/client/getting-started.go`
- Modify: `examples/go/sandbox/lifecycle.go`
- Modify: `examples/python/client/getting-started.py`
- Modify: `examples/python/sandbox/lifecycle.py`

**Interfaces:**
- Sandbox exposes `Register`, `Deploy`, and `Create`
- Existing `Create` behavior remains register/deploy/wait
- VM and Bare Metal examples use typed services

- [ ] **Step 1: Add Sandbox lifecycle parity tests**

Test that Register does not deploy, Deploy performs one deploy call, and Create
performs register then deploy.

- [ ] **Step 2: Refactor Sandbox to shared lifecycle**

Preserve files, terminals, exec, freeze/thaw, fork, and publish behavior.

- [ ] **Step 3: Rewrite Go and Python examples**

Provide one VM, one Bare Metal, and one Sandbox lifecycle example per language.
Examples must clean up created resources.

- [ ] **Step 4: Compile and syntax-check examples**

```bash
go -C examples/go test ./...
cargo test --manifest-path cli/Cargo.toml
PYTHONPATH=libs/python/src \
  libs/python/.venv/bin/python -m py_compile \
  examples/python/client/getting-started.py \
  examples/python/sandbox/lifecycle.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/go/sandbox*.go libs/python/src/targon/client/sandbox \
  examples/go examples/python
git commit -m "refactor: align typed workload lifecycle"
```

---

### Task 10: Clarify organization scope and publish migration docs

**Files:**
- Modify: `libs/go/README.md`
- Modify: `libs/python/README.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/sdk/getting-started.md` in `targon-docs`
- Modify: `docs/sdk/sandboxes.md` in `targon-docs`
- Create: `docs/sdk/workloads.md` in `targon-docs`

**Interfaces:**
- Documentation names user-scoped and organization-bound services explicitly
- Migration tables contain no removed API as a recommended path

- [ ] **Step 1: Document organization scope**

Add examples:

```go
orgs, err := client.Orgs.List(ctx, targon.Page{})
acme, err := client.ForOrg("acme")
members, err := acme.Members.List(ctx, ...)
credits, err := acme.Credits.Get(ctx)
```

```python
orgs = client.orgs.list()
acme = client.for_org("acme")
members = acme.members.list()
credits = acme.credits.get()
```

- [ ] **Step 2: Document typed workload migration**

Show `VMs.Create`, `BareMetal.Create`, and `Sandboxes.Create`, plus advanced
Register/Deploy.

- [ ] **Step 3: Document removed APIs**

State that Rental and Volume APIs are unavailable in v4 and have no current
replacement.

- [ ] **Step 4: Run docs verification**

```bash
cd ../targon-docs
bun run check
bun run build
```

Expected: PASS.

- [ ] **Step 5: Commit SDK and docs changes separately**

```bash
git add README.md CHANGELOG.md libs/go/README.md libs/python/README.md
git commit -m "docs: explain typed workload migration"
```

Commit the docs repository changes on its matching branch.

---

### Task 11: Final verification and release checks

**Files:**
- Verify all files changed by Tasks 1–10

- [ ] **Step 1: Run repository verification**

```bash
PATH="$PWD/libs/python/.venv/bin:$PATH" make test
PATH="$PWD/libs/python/.venv/bin:$PATH" make lint
CARGO_TARGET_DIR="$PWD/cli/target" \
  PATH="$PWD/libs/python/.venv/bin:$PATH" make build
```

Expected: PASS.

- [ ] **Step 2: Check package artifacts**

```bash
python -m twine check libs/python/dist/*
npm --prefix libs/typescript audit --audit-level=high
cd libs/typescript && npm pack --dry-run
go -C libs/go test ./...
go -C libs/go vet ./...
```

Expected: all checks pass and npm reports zero high vulnerabilities.

- [ ] **Step 3: Scan removed APIs**

```bash
rg 'RENTAL|VolumeService|VolumeClient|AttachVolume|DetachVolume|suspend' \
  libs/go libs/python examples
```

Expected: no public implementation or example references. Migration
documentation may mention removed names.

- [ ] **Step 4: Confirm branch hygiene**

```bash
git diff --check
git status --short
git log --format=%B origin/main..HEAD | rg '^Co-authored-by: Cursor'
```

Expected: clean tree, no whitespace errors, and no Cursor co-author trailers.

- [ ] **Step 5: Update PR checks**

Push the SDK and docs branches, wait for all checks, and confirm both PRs are
mergeable and green.
