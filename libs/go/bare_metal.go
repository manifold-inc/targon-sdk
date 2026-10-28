package targon

import (
	"context"
	"net/http"
	"time"
)

// DefaultBareMetalWaitTimeout is the Create and Deploy wait timeout used when
// WaitOptions.Timeout is unset. Bare Metal provisioning is much slower than VM
// or Sandbox provisioning.
const DefaultBareMetalWaitTimeout = 30 * time.Minute

// BareMetalCreateParams is the public Bare Metal registration contract.
type BareMetalCreateParams struct {
	Name           string
	Image          string
	ResourceName   string
	Region         string
	BMConfig       *BMConfigInput
	ProjectID      string
	ReservationUID string
	SSHKeys        []string
	Wait           WaitOptions
}

// BareMetalReinstallParams contains the fields accepted by Bare Metal
// reinstall. When SSHKeys is empty the currently attached keys are reused.
type BareMetalReinstallParams struct {
	Image    string
	SSHKeys  []string
	Hostname string
}

// BareMetal is a method-bearing Bare Metal workload.
type BareMetal struct {
	Workload
	service *BareMetalService
}

// BareMetalService manages Bare Metal workloads.
type BareMetalService struct {
	client *Client
}

func (s *BareMetalService) registrationPayload(params BareMetalCreateParams) (map[string]any, error) {
	name, err := requireNonEmpty(params.Name, "name")
	if err != nil {
		return nil, err
	}
	image, err := requireNonEmpty(params.Image, "image")
	if err != nil {
		return nil, err
	}
	resourceName, err := requireNonEmpty(params.ResourceName, "resource_name")
	if err != nil {
		return nil, err
	}
	region, err := requireNonEmpty(params.Region, "region")
	if err != nil {
		return nil, err
	}
	sshKeys, err := normalizeSSHKeys(params.SSHKeys)
	if err != nil {
		return nil, err
	}
	if len(sshKeys) == 0 {
		return nil, validation("at least one SSH key is required", "ssh_keys", params.SSHKeys)
	}
	payload := map[string]any{
		"type":          "BM",
		"name":          name,
		"image":         image,
		"resource_name": resourceName,
		"region":        region,
		"ssh_keys":      sshKeys,
	}
	if params.BMConfig != nil {
		bmConfig, err := hostnameConfig(params.BMConfig.Hostname, "bm_config.hostname")
		if err != nil {
			return nil, err
		}
		if bmConfig != nil {
			payload["bm_config"] = bmConfig
		}
	}
	setOptionalString(payload, "project_id", params.ProjectID)
	setOptionalString(payload, "reservation_uid", params.ReservationUID)
	return payload, nil
}

func (s *BareMetalService) Register(ctx context.Context, params BareMetalCreateParams) (*BareMetal, error) {
	payload, err := s.registrationPayload(params)
	if err != nil {
		return nil, err
	}
	workload, err := registerTyped(ctx, s.client, payload, "BM")
	if err != nil {
		return nil, err
	}
	return hydrateBareMetal(s, workload), nil
}

// Deploy starts provisioning. Without WaitOptions.Timeout it waits up to
// DefaultBareMetalWaitTimeout for the workload to run.
func (s *BareMetalService) Deploy(ctx context.Context, workloadUID string, wait WaitOptions) (*BareMetal, error) {
	workload, err := deployTyped(ctx, s.client, workloadUID, "BM", bareMetalWait(wait))
	if err != nil {
		return nil, err
	}
	return hydrateBareMetal(s, workload), nil
}

func bareMetalWait(wait WaitOptions) WaitOptions {
	if wait.Timeout <= 0 {
		wait.Timeout = DefaultBareMetalWaitTimeout
	}
	return wait
}

func (s *BareMetalService) Create(ctx context.Context, params BareMetalCreateParams) (*BareMetal, error) {
	payload, err := s.registrationPayload(params)
	if err != nil {
		return nil, err
	}
	registered, err := registerWorkload(ctx, s.client, s.client.Workloads, payload)
	if err != nil {
		return nil, err
	}
	return s.Deploy(ctx, registered.UID, params.Wait)
}

func (s *BareMetalService) Get(ctx context.Context, workloadUID string) (*BareMetal, error) {
	workload, err := getTyped(ctx, s.client.Workloads, workloadUID, "BM")
	if err != nil {
		return nil, err
	}
	return hydrateBareMetal(s, workload), nil
}

func (s *BareMetalService) operation(ctx context.Context, workloadUID, action string, payload any) (*BareMetal, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.client.Workloads.path(workloadUID, action)
	if err != nil {
		return nil, err
	}
	var ignored WorkloadOperationResponse
	if err := s.client.doNoRetry(ctx, http.MethodPost, path, nil, payload, &ignored); err != nil {
		return nil, err
	}
	return s.Get(ctx, workloadUID)
}

func (s *BareMetalService) Reboot(ctx context.Context, workloadUID string) (*BareMetal, error) {
	return s.operation(ctx, workloadUID, "reboot", nil)
}

func (s *BareMetalService) PowerOn(ctx context.Context, workloadUID string) (*BareMetal, error) {
	return s.operation(ctx, workloadUID, "power-on", nil)
}

func (s *BareMetalService) PowerOff(ctx context.Context, workloadUID string) (*BareMetal, error) {
	return s.operation(ctx, workloadUID, "power-off", nil)
}

func (s *BareMetalService) Reinstall(
	ctx context.Context,
	workloadUID string,
	params BareMetalReinstallParams,
) (*BareMetal, error) {
	image, err := requireNonEmpty(params.Image, "image")
	if err != nil {
		return nil, err
	}
	payload := map[string]any{"image": image}
	sshKeys, err := normalizeSSHKeys(params.SSHKeys)
	if err != nil {
		return nil, err
	}
	if len(sshKeys) > 0 {
		payload["ssh_keys"] = sshKeys
	}
	hostname, err := hostnameConfig(params.Hostname, "hostname")
	if err != nil {
		return nil, err
	}
	for key, value := range hostname {
		payload[key] = value
	}
	return s.operation(ctx, workloadUID, "reinstall", payload)
}

// Images lists the images Bare Metal workloads can install.
func (s *BareMetalService) Images(ctx context.Context) ([]BareMetalImage, error) {
	return s.client.Workloads.images(ctx, "bm-images")
}

func hydrateBareMetal(service *BareMetalService, workload *Workload) *BareMetal {
	if workload == nil {
		return nil
	}
	return &BareMetal{Workload: *workload, service: service}
}

func (b *BareMetal) boundService() (*BareMetalService, error) {
	if b == nil || b.service == nil {
		return nil, validation("Bare Metal is not bound to a client service", "bare_metal", nil)
	}
	if _, err := requireNonEmpty(b.UID, "workload_uid"); err != nil {
		return nil, err
	}
	return b.service, nil
}

func (b *BareMetal) replace(updated *BareMetal) *BareMetal {
	if updated != nil {
		*b = *updated
	}
	return b
}

func (b *BareMetal) mutate(
	call func(*BareMetalService, string) (*BareMetal, error),
) (*BareMetal, error) {
	service, err := b.boundService()
	if err != nil {
		return nil, err
	}
	updated, err := call(service, b.UID)
	if err != nil {
		return nil, err
	}
	return b.replace(updated), nil
}

// Refresh reloads the full workload representation into this object.
func (b *BareMetal) Refresh(ctx context.Context) (*BareMetal, error) {
	return b.mutate(func(service *BareMetalService, uid string) (*BareMetal, error) {
		return service.Get(ctx, uid)
	})
}

func (b *BareMetal) Deploy(ctx context.Context, wait WaitOptions) (*BareMetal, error) {
	return b.mutate(func(service *BareMetalService, uid string) (*BareMetal, error) {
		return service.Deploy(ctx, uid, wait)
	})
}

func (b *BareMetal) Reboot(ctx context.Context) (*BareMetal, error) {
	return b.mutate(func(service *BareMetalService, uid string) (*BareMetal, error) {
		return service.Reboot(ctx, uid)
	})
}

func (b *BareMetal) PowerOn(ctx context.Context) (*BareMetal, error) {
	return b.mutate(func(service *BareMetalService, uid string) (*BareMetal, error) {
		return service.PowerOn(ctx, uid)
	})
}

func (b *BareMetal) PowerOff(ctx context.Context) (*BareMetal, error) {
	return b.mutate(func(service *BareMetalService, uid string) (*BareMetal, error) {
		return service.PowerOff(ctx, uid)
	})
}

func (b *BareMetal) Reinstall(
	ctx context.Context,
	params BareMetalReinstallParams,
) (*BareMetal, error) {
	return b.mutate(func(service *BareMetalService, uid string) (*BareMetal, error) {
		return service.Reinstall(ctx, uid, params)
	})
}
