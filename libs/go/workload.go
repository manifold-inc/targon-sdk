package targon

import (
	"context"
	"regexp"
	"strings"
)

var (
	validWorkloadTypes      = map[string]struct{}{"VM": {}, "BM": {}, "SANDBOX": {}}
	deprecatedWorkloadTypes = map[string]struct{}{"RENTAL": {}, "SERVERLESS": {}, "INFERENCE": {}, "FUNCTION": {}}
	validLogTypes           = map[string]struct{}{"serial": {}, "qemu": {}}
	hostnamePattern         = regexp.MustCompile(`^[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?)*$`)
)

// PortConfig exposes a workload port.
type PortConfig struct {
	Port     int    `json:"port"`
	Protocol string `json:"protocol,omitempty"`
	Routing  string `json:"routing,omitempty"`
}

// VMConfigInput is the optional vm_config accepted when registering a VM.
// Hostname defaults to the workload name on the server.
type VMConfigInput struct {
	Hostname string `json:"hostname,omitempty"`
}

// BMConfigInput is the optional bm_config accepted when registering Bare Metal.
// Hostname defaults to the workload name on the server.
type BMConfigInput struct {
	Hostname string `json:"hostname,omitempty"`
}

// hostnameConfig returns the {"hostname": ...} payload for vm_config or
// bm_config, or nil when no hostname is set.
func hostnameConfig(hostname, field string) (map[string]string, error) {
	hostname = strings.ToLower(strings.TrimSpace(hostname))
	if hostname == "" {
		return nil, nil
	}
	if len(hostname) > 63 || !hostnamePattern.MatchString(hostname) {
		return nil, validation(field+" must be a valid RFC 1123 hostname (lowercase letters, digits, hyphens, dots; max 63 chars)", field, hostname)
	}
	return map[string]string{"hostname": hostname}, nil
}

// setOptionalString adds a trimmed value to payload, omitting empty strings.
func setOptionalString(payload map[string]any, key, value string) {
	if value = strings.TrimSpace(value); value != "" {
		payload[key] = value
	}
}

// VMConfig is the VM configuration returned on VM workloads.
type VMConfig struct {
	Hostname string `json:"hostname,omitempty"`
}

// BMConfig is the Bare Metal configuration returned on BM workloads.
type BMConfig struct {
	Region   string `json:"region"`
	Hostname string `json:"hostname,omitempty"`
}

// WorkloadCreator identifies the token that created a workload.
type WorkloadCreator struct {
	TokenType string `json:"token_type"`
	TokenName string `json:"token_name,omitempty"`
	Email     string `json:"email,omitempty"`
	Username  string `json:"username,omitempty"`
}

// WorkloadSSHKey is an SSH key attached to a workload.
type WorkloadSSHKey struct {
	UID       string `json:"uid"`
	Name      string `json:"name"`
	PublicKey string `json:"public_key_raw"`
}

// WorkloadURL is a published workload URL.
type WorkloadURL struct {
	Port int    `json:"port"`
	URL  string `json:"url"`
}

// WorkloadState is nested workload state.
type WorkloadState struct {
	Status        string        `json:"status"`
	Message       string        `json:"message"`
	ReadyReplicas int           `json:"ready_replicas"`
	TotalReplicas int           `json:"total_replicas"`
	URLs          []WorkloadURL `json:"urls"`
	PublicIP      *string       `json:"public_ip"`
	SSHPort       *int          `json:"ssh_port"`
}

// WorkloadResource is the resolved hardware profile.
type WorkloadResource struct {
	Name          string  `json:"name"`
	DisplayName   string  `json:"display_name"`
	GPUVendor     string  `json:"gpu_vendor,omitempty"`
	GPUModel      string  `json:"gpu_model,omitempty"`
	GPUType       *string `json:"gpu_type,omitempty"`
	GPUCount      *int    `json:"gpu_count,omitempty"`
	CPUMillicores int     `json:"cpu_millicores"`
	CPUVendor     string  `json:"cpu_vendor,omitempty"`
	CPUModel      string  `json:"cpu_model,omitempty"`
	CPUSockets    int     `json:"cpu_sockets,omitempty"`
	MemoryMiB     int64   `json:"memory_mib"`
	DiskSizeMiB   *int64  `json:"disk_size_mib,omitempty"`
	DiskLabel     string  `json:"disk_label,omitempty"`
	NetworkMode   *string `json:"network_mode,omitempty"`
	CCEnabled     *bool   `json:"cc_enabled,omitempty"`
	VCPU          int     `json:"vcpu"`
	Memory        int64   `json:"memory"`
}

// Workload is a full workload record.
type Workload struct {
	UID               string            `json:"uid"`
	Name              string            `json:"name"`
	Image             string            `json:"image"`
	Type              string            `json:"type"`
	ResourceName      string            `json:"resource_name"`
	ProjectID         *string           `json:"project_id,omitempty"`
	ReservationUID    *string           `json:"reservation_uid,omitempty"`
	CreatedBy         *WorkloadCreator  `json:"created_by,omitempty"`
	Ports             []PortConfig      `json:"ports,omitempty"`
	SSHKeys           []WorkloadSSHKey  `json:"ssh_keys,omitempty"`
	VMConfig          *VMConfig         `json:"vm_config,omitempty"`
	BMConfig          *BMConfig         `json:"bm_config,omitempty"`
	SandboxConfig     *SandboxConfig    `json:"sandbox_config,omitempty"`
	State             *WorkloadState    `json:"state,omitempty"`
	Resource          *WorkloadResource `json:"resource,omitempty"`
	CostPerHour       *float64          `json:"cost_per_hour,omitempty"`
	FrozenCostPerHour *float64          `json:"frozen_cost_per_hour,omitempty"`
	Revision          string            `json:"revision,omitempty"`
	CreatedAt         string            `json:"created_at"`
	UpdatedAt         string            `json:"updated_at"`
}

// WorkloadListItem is a summarized list row.
type WorkloadListItem struct {
	UID               string            `json:"uid"`
	Name              string            `json:"name"`
	Image             string            `json:"image,omitempty"`
	Type              string            `json:"type"`
	ReservationUID    *string           `json:"reservation_uid,omitempty"`
	CreatedBy         *WorkloadCreator  `json:"created_by,omitempty"`
	State             *WorkloadState    `json:"state,omitempty"`
	Resource          *WorkloadResource `json:"resource,omitempty"`
	CostPerHour       *float64          `json:"cost_per_hour,omitempty"`
	FrozenCostPerHour *float64          `json:"frozen_cost_per_hour,omitempty"`
	CreatedAt         string            `json:"created_at"`
	UpdatedAt         string            `json:"updated_at"`
}

// WorkloadOperationResponse is returned by deploy and power operations.
type WorkloadOperationResponse WorkloadListItem

// WorkloadStateResponse is the dedicated state endpoint payload.
type WorkloadStateResponse struct {
	UID           string        `json:"uid"`
	WorkloadType  string        `json:"workload_type"`
	Status        string        `json:"status"`
	Message       string        `json:"message"`
	ReadyReplicas int           `json:"ready_replicas"`
	TotalReplicas int           `json:"total_replicas"`
	URLs          []WorkloadURL `json:"urls"`
	PublicIP      *string       `json:"public_ip"`
	SSHPort       *int          `json:"ssh_port"`
	UpdatedAt     string        `json:"updated_at"`
}

// WorkloadEvent is a workload lifecycle event.
type WorkloadEvent struct {
	WorkloadUID     string  `json:"workload_uid"`
	WorkloadType    string  `json:"workload_type"`
	EventType       string  `json:"event_type"`
	NewStatus       *string `json:"new_status"`
	Message         *string `json:"message"`
	DisplayMessage  *string `json:"display_message"`
	Reason          *string `json:"reason"`
	PodName         *string `json:"pod_name"`
	ContainerName   *string `json:"container_name"`
	ContainerImage  *string `json:"container_image"`
	ExitCode        *int    `json:"exit_code"`
	ReplicaCount    *int    `json:"replica_count"`
	OldReplicaCount *int    `json:"old_replica_count"`
	ResourceName    *string `json:"resource_name"`
	CreatedAt       string  `json:"created_at"`
}

// SSHKeyAttachResponse is returned when attaching an SSH key.
type SSHKeyAttachResponse struct {
	WorkloadUID string `json:"workload_uid"`
	SSHKeyUID   string `json:"ssh_key_uid"`
}

// ListWorkloadsParams filters a workload list.
type ListWorkloadsParams struct {
	Page      Page
	Type      string
	Status    string
	ProjectID string
	Name      string
}

// LogOptions controls VM log fetch and stream requests. Logs never follows;
// StreamLogs always follows.
type LogOptions struct {
	Since    string
	Tail     int
	Previous bool
	LogType  string
}

// WorkloadImage is a bootable image returned by the vm-images and bm-images
// routes.
type WorkloadImage struct {
	Name        string `json:"name"`
	DisplayName string `json:"display_name"`
	OSFamily    string `json:"os_family"`
	Description string `json:"description,omitempty"`
}

// VMImage is a bootable VM image.
type VMImage = WorkloadImage

// BareMetalImage is an installable Bare Metal image.
type BareMetalImage = WorkloadImage

// WorkloadService manages workloads.
type WorkloadService struct {
	client *Client
}

func (s *WorkloadService) path(parts ...string) (string, error) {
	return s.client.orgResourcePath("workloads", parts...)
}

func (s *WorkloadService) Get(ctx context.Context, workloadUID string) (*Workload, error) {
	return s.get(ctx, workloadUID)
}

func (s *WorkloadService) get(ctx context.Context, workloadUID string) (*Workload, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.path(workloadUID)
	if err != nil {
		return nil, err
	}
	var out Workload
	err = s.client.do(ctx, "GET", path, nil, nil, &out)
	return &out, err
}

func (s *WorkloadService) List(ctx context.Context, params ListWorkloadsParams) (List[WorkloadListItem], error) {
	path, err := s.path()
	if err != nil {
		return List[WorkloadListItem]{}, err
	}
	q := params.Page.query()
	if params.Type != "" {
		q.Set("type", params.Type)
	}
	if params.Status != "" {
		q.Set("status", params.Status)
	}
	if params.ProjectID != "" {
		q.Set("project_id", params.ProjectID)
	}
	if params.Name != "" {
		q.Set("name", params.Name)
	}
	var out List[WorkloadListItem]
	err = s.client.do(ctx, "GET", path, q, nil, &out)
	return out, err
}

func (s *WorkloadService) Delete(ctx context.Context, workloadUID string) error {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return err
	}
	path, err := s.path(workloadUID)
	if err != nil {
		return err
	}
	return s.client.do(ctx, "DELETE", path, nil, nil, nil)
}

func (s *WorkloadService) GetState(ctx context.Context, workloadUID string) (*WorkloadStateResponse, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.path(workloadUID, "state")
	if err != nil {
		return nil, err
	}
	var out WorkloadStateResponse
	err = s.client.do(ctx, "GET", path, nil, nil, &out)
	return &out, err
}

func (s *WorkloadService) GetEvents(ctx context.Context, workloadUID string, page Page) (List[WorkloadEvent], error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return List[WorkloadEvent]{}, err
	}
	path, err := s.path(workloadUID, "events")
	if err != nil {
		return List[WorkloadEvent]{}, err
	}
	var out List[WorkloadEvent]
	err = s.client.do(ctx, "GET", path, page.query(), nil, &out)
	return out, err
}

// WaitUntilReady polls workload state until it is running, reaches a terminal
// state, or the wait times out. It works for every workload type.
func (s *WorkloadService) WaitUntilReady(ctx context.Context, workloadUID string, options WaitOptions) (*WorkloadStateResponse, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	timeout, interval := waitDurations(options)
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	lastStatus := ""
	return pollUntil(ctx, interval, func() (*WorkloadStateResponse, bool, error) {
		state, err := s.GetState(ctx, workloadUID)
		if err != nil {
			return nil, false, err
		}
		lastStatus = state.Status
		status := strings.ToLower(state.Status)
		if status == "running" {
			return state, true, nil
		}
		if _, ok := terminalWorkloadStates[status]; ok {
			return nil, false, typedStateError(workloadUID, state.WorkloadType, state.Status, state.Message)
		}
		return state, false, nil
	}, func(*WorkloadStateResponse) error {
		return typedTimeoutError(workloadUID, "", "running", lastStatus, timeout.Seconds())
	})
}

// Verify reports whether digest matches the SHA-256 of the workload image.
func (s *WorkloadService) Verify(ctx context.Context, workloadUID, digest string) (bool, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "uid")
	if err != nil {
		return false, err
	}
	digest, err = requireNonEmpty(digest, "digest")
	if err != nil {
		return false, err
	}
	path, err := s.path("verify")
	if err != nil {
		return false, err
	}
	var out struct {
		Verified bool `json:"verified"`
	}
	err = s.client.do(ctx, "POST", path, nil, map[string]string{"uid": workloadUID, "digest": digest}, &out)
	return out.Verified, err
}

// AttachSSHKey attaches an organization SSH key to a workload.
func (s *WorkloadService) AttachSSHKey(ctx context.Context, workloadUID, sshKeyUID string) (*SSHKeyAttachResponse, error) {
	path, err := s.sshKeyPath(workloadUID, sshKeyUID)
	if err != nil {
		return nil, err
	}
	var out SSHKeyAttachResponse
	if err := s.client.doNoRetry(ctx, "PUT", path, nil, nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// DetachSSHKey removes an SSH key from a workload.
func (s *WorkloadService) DetachSSHKey(ctx context.Context, workloadUID, sshKeyUID string) error {
	path, err := s.sshKeyPath(workloadUID, sshKeyUID)
	if err != nil {
		return err
	}
	return s.client.do(ctx, "DELETE", path, nil, nil, nil)
}

func (s *WorkloadService) sshKeyPath(workloadUID, sshKeyUID string) (string, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return "", err
	}
	sshKeyUID, err = requireNonEmpty(sshKeyUID, "ssh_key_uid")
	if err != nil {
		return "", err
	}
	return s.path(workloadUID, "ssh-keys", sshKeyUID)
}

func (s *WorkloadService) images(ctx context.Context, route string) ([]WorkloadImage, error) {
	path, err := s.path(route)
	if err != nil {
		return nil, err
	}
	var out []WorkloadImage
	if err := s.client.do(ctx, "GET", path, nil, nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}
