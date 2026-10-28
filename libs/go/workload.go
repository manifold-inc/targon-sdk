package targon

import (
	"context"
)

var (
	validWorkloadTypes      = map[string]struct{}{"VM": {}, "BM": {}, "SANDBOX": {}}
	deprecatedWorkloadTypes = map[string]struct{}{"RENTAL": {}, "SERVERLESS": {}, "INFERENCE": {}, "FUNCTION": {}}
	validLogTypes           = map[string]struct{}{"serial": {}, "qemu": {}}
)

// PortConfig exposes a workload port.
type PortConfig struct {
	Port     int    `json:"port"`
	Protocol string `json:"protocol,omitempty"`
	Routing  string `json:"routing,omitempty"`
}

// VMConfig is required for VM workloads.
type VMConfig struct {
	Password string `json:"password"`
}

func (c VMConfig) toPayload() (map[string]string, error) {
	password, err := requireNonEmpty(c.Password, "vm_config.password")
	if err != nil {
		return nil, err
	}
	if len(password) < 4 {
		return nil, &ValidationError{Message: "vm_config.password must be at least 4 characters", Field: "vm_config.password"}
	}
	return map[string]string{"password": password}, nil
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
	Region            string            `json:"region"`
	Hostname          string            `json:"hostname"`
	ProjectID         *string           `json:"project_id"`
	Ports             []PortConfig      `json:"ports"`
	SSHKeys           []WorkloadSSHKey  `json:"ssh_keys"`
	SandboxConfig     *SandboxConfig    `json:"sandbox_config,omitempty"`
	State             *WorkloadState    `json:"state"`
	Resource          *WorkloadResource `json:"resource"`
	CostPerHour       *float64          `json:"cost_per_hour"`
	FrozenCostPerHour *float64          `json:"frozen_cost_per_hour,omitempty"`
	Revision          string            `json:"revision"`
	CreatedAt         string            `json:"created_at"`
	UpdatedAt         string            `json:"updated_at"`
}

// WorkloadListItem is a summarized list row.
type WorkloadListItem struct {
	UID               string            `json:"uid"`
	Name              string            `json:"name"`
	Image             string            `json:"image"`
	Type              string            `json:"type"`
	State             *WorkloadState    `json:"state"`
	Resource          *WorkloadResource `json:"resource"`
	CostPerHour       *float64          `json:"cost_per_hour"`
	FrozenCostPerHour *float64          `json:"frozen_cost_per_hour,omitempty"`
	Revision          string            `json:"revision"`
	CreatedAt         string            `json:"created_at"`
	UpdatedAt         string            `json:"updated_at"`
}

// WorkloadOperationResponse is returned by deploy/suspend/reboot.
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

// LogOptions controls log fetch/stream.
type LogOptions struct {
	Since    string
	Tail     int
	Previous bool
	LogType  string
	Follow   bool
}

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
