package targon

import (
	"context"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// VMCreateParams is the public VM registration contract.
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

// VMUpdateParams contains the fields supported by the VM update route.
type VMUpdateParams struct {
	Name  *string
	Ports *[]PortConfig
}

// VM is a method-bearing VM workload.
type VM struct {
	Workload
	service *VMService
}

// VMService manages VM workloads.
type VMService struct {
	client *Client
}

func (s *VMService) registrationPayload(params VMCreateParams) (map[string]any, error) {
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
	vmConfig, err := (VMConfig{Password: params.Password}).toPayload()
	if err != nil {
		return nil, err
	}
	payload := map[string]any{
		"type":          "VM",
		"name":          name,
		"image":         image,
		"resource_name": resourceName,
		"vm_config":     vmConfig,
	}
	if params.ProjectID != "" {
		payload["project_id"] = params.ProjectID
	}
	if params.ReservationUID != "" {
		payload["reservation_uid"] = params.ReservationUID
	}
	if len(params.SSHKeys) > 0 {
		payload["ssh_keys"] = params.SSHKeys
	}
	if len(params.Ports) > 0 {
		payload["ports"] = params.Ports
	}
	return payload, nil
}

func (s *VMService) Register(ctx context.Context, params VMCreateParams) (*VM, error) {
	payload, err := s.registrationPayload(params)
	if err != nil {
		return nil, err
	}
	workload, err := registerTyped(ctx, s.client, payload, "VM")
	if err != nil {
		return nil, err
	}
	return hydrateVM(s, workload), nil
}

func (s *VMService) Deploy(ctx context.Context, workloadUID string, wait WaitOptions) (*VM, error) {
	workload, err := deployTyped(ctx, s.client, workloadUID, "VM", wait)
	if err != nil {
		return nil, err
	}
	return hydrateVM(s, workload), nil
}

func (s *VMService) Create(ctx context.Context, params VMCreateParams) (*VM, error) {
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

func (s *VMService) Get(ctx context.Context, workloadUID string) (*VM, error) {
	workload, err := getTyped(ctx, s.client.Workloads, workloadUID, "VM")
	if err != nil {
		return nil, err
	}
	return hydrateVM(s, workload), nil
}

func (s *VMService) Update(ctx context.Context, workloadUID string, params VMUpdateParams) (*VM, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	payload := map[string]any{}
	if params.Name != nil {
		name, err := requireNonEmpty(*params.Name, "name")
		if err != nil {
			return nil, err
		}
		payload["name"] = name
	}
	if params.Ports != nil {
		payload["ports"] = *params.Ports
	}
	if len(payload) == 0 {
		return nil, validation("at least one VM field must be supplied", "update", nil)
	}
	path, err := s.client.Workloads.path(workloadUID)
	if err != nil {
		return nil, err
	}
	var out Workload
	if err := s.client.doNoRetry(ctx, http.MethodPatch, path, nil, payload, &out); err != nil {
		return nil, err
	}
	if err := requireTypedWorkload(&out, "VM"); err != nil {
		return nil, err
	}
	return hydrateVM(s, &out), nil
}

func (s *VMService) Reboot(ctx context.Context, workloadUID string) (*VM, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.client.Workloads.path(workloadUID, "reboot")
	if err != nil {
		return nil, err
	}
	var ignored WorkloadOperationResponse
	if err := s.client.doNoRetry(ctx, http.MethodPost, path, nil, nil, &ignored); err != nil {
		return nil, err
	}
	return s.Get(ctx, workloadUID)
}

func vmLogQuery(options LogOptions) (url.Values, error) {
	query := url.Values{}
	if options.Since != "" {
		query.Set("since", options.Since)
	}
	if options.Tail > 0 {
		query.Set("tail", strconv.Itoa(options.Tail))
	}
	if options.Previous {
		query.Set("previous", "true")
	}
	if options.LogType != "" {
		logType, err := requireNonEmpty(options.LogType, "log_type")
		if err != nil {
			return nil, err
		}
		logType = strings.ToLower(logType)
		if _, ok := validLogTypes[logType]; !ok {
			return nil, validation("log_type must be one of qemu, serial", "log_type", options.LogType)
		}
		query.Set("type", logType)
	}
	return query, nil
}

func (s *VMService) Logs(ctx context.Context, workloadUID string, options LogOptions) (string, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return "", err
	}
	path, err := s.client.Workloads.path(workloadUID, "logs")
	if err != nil {
		return "", err
	}
	query, err := vmLogQuery(options)
	if err != nil {
		return "", err
	}
	return s.client.getText(ctx, path, query)
}

func hydrateVM(service *VMService, workload *Workload) *VM {
	if workload == nil {
		return nil
	}
	return &VM{Workload: *workload, service: service}
}

func (v *VM) boundService() (*VMService, error) {
	if v == nil || v.service == nil {
		return nil, validation("VM is not bound to a client service", "vm", nil)
	}
	if _, err := requireNonEmpty(v.UID, "workload_uid"); err != nil {
		return nil, err
	}
	return v.service, nil
}

func (v *VM) replace(updated *VM) *VM {
	if updated != nil {
		*v = *updated
	}
	return v
}

func (v *VM) mutate(call func(*VMService, string) (*VM, error)) (*VM, error) {
	service, err := v.boundService()
	if err != nil {
		return nil, err
	}
	updated, err := call(service, v.UID)
	if err != nil {
		return nil, err
	}
	return v.replace(updated), nil
}

// Refresh reloads the full workload representation into this object.
func (v *VM) Refresh(ctx context.Context) (*VM, error) {
	return v.mutate(func(service *VMService, uid string) (*VM, error) {
		return service.Get(ctx, uid)
	})
}

func (v *VM) Deploy(ctx context.Context, wait WaitOptions) (*VM, error) {
	return v.mutate(func(service *VMService, uid string) (*VM, error) {
		return service.Deploy(ctx, uid, wait)
	})
}

func (v *VM) Update(ctx context.Context, params VMUpdateParams) (*VM, error) {
	return v.mutate(func(service *VMService, uid string) (*VM, error) {
		return service.Update(ctx, uid, params)
	})
}

func (v *VM) Reboot(ctx context.Context) (*VM, error) {
	return v.mutate(func(service *VMService, uid string) (*VM, error) {
		return service.Reboot(ctx, uid)
	})
}

func (v *VM) Logs(ctx context.Context, options LogOptions) (string, error) {
	service, err := v.boundService()
	if err != nil {
		return "", err
	}
	return service.Logs(ctx, v.UID, options)
}
