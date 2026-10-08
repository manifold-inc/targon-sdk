package targon

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const (
	argMaxBytes = 1 << 16
)

var (
	terminalWorkloadStates  = map[string]struct{}{"error": {}, "suspended": {}, "deleted": {}}
	validWorkloadTypes      = map[string]struct{}{"RENTAL": {}, "VM": {}}
	deprecatedWorkloadTypes = map[string]struct{}{"SERVERLESS": {}, "INFERENCE": {}, "FUNCTION": {}}
	validLogTypes           = map[string]struct{}{"serial": {}, "qemu": {}}
)

// EnvVar is a container environment variable.
type EnvVar struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// PortConfig exposes a workload port.
type PortConfig struct {
	Port     int    `json:"port"`
	Protocol string `json:"protocol,omitempty"`
	Routing  string `json:"routing,omitempty"`
}

// RegistryAuth is private registry credentials.
type RegistryAuth struct {
	Server   string `json:"server"`
	Username string `json:"username"`
	Password string `json:"password"`
}

func (r RegistryAuth) toPayload() (map[string]string, error) {
	server, err := requireNonEmpty(r.Server, "server")
	if err != nil {
		return nil, err
	}
	username, err := requireNonEmpty(r.Username, "username")
	if err != nil {
		return nil, err
	}
	password, err := requireNonEmpty(r.Password, "password")
	if err != nil {
		return nil, err
	}
	return map[string]string{"server": server, "username": username, "password": password}, nil
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

// VolumeMount is a volume to attach at create time.
type VolumeMount struct {
	UID       string `json:"uid"`
	MountPath string `json:"mount_path"`
	ReadOnly  bool   `json:"read_only"`
}

func (v VolumeMount) toPayload() (map[string]any, error) {
	uid, err := requireNonEmpty(v.UID, "volume.uid")
	if err != nil {
		return nil, err
	}
	path, err := requireAbsolutePath(v.MountPath, "volume.mount_path")
	if err != nil {
		return nil, err
	}
	return map[string]any{"uid": uid, "mount_path": path, "read_only": v.ReadOnly}, nil
}

func requireAbsolutePath(value, field string) (string, error) {
	path, err := requireNonEmpty(value, field)
	if err != nil {
		return "", err
	}
	if !strings.HasPrefix(path, "/") {
		return "", &ValidationError{Message: field + " must be an absolute path starting with '/'", Field: field, Value: value}
	}
	return path, nil
}

// WorkloadVolumeMount is a volume attached to a workload.
type WorkloadVolumeMount struct {
	UID          string  `json:"uid"`
	Name         string  `json:"name"`
	MountPath    string  `json:"mount_path"`
	ReadOnly     bool    `json:"read_only"`
	LastBackupAt *string `json:"last_backup_at"`
}

// WorkloadSSHKey is an SSH key attached to a workload.
type WorkloadSSHKey struct {
	UID       string `json:"uid"`
	Name      string `json:"name"`
	PublicKey string `json:"public_key_raw"`
}

// CreateWorkloadRequest is the body for creating a workload.
type CreateWorkloadRequest struct {
	Name         string
	Image        string
	ResourceName string
	Type         string
	ProjectID    string
	Ports        []PortConfig
	Envs         []EnvVar
	Commands     []string
	Args         []string
	RegistryAuth *RegistryAuth
	Volumes      []VolumeMount
	SSHKeys      []string
	VMConfig     *VMConfig
}

func (r CreateWorkloadRequest) toPayload() (map[string]any, error) {
	workloadType := strings.ToUpper(strings.TrimSpace(r.Type))
	if workloadType == "" {
		workloadType = "RENTAL"
	}
	if _, ok := deprecatedWorkloadTypes[workloadType]; ok {
		return nil, &ValidationError{Message: "workload type " + workloadType + " has been deprecated", Field: "type", Value: r.Type}
	}
	if _, ok := validWorkloadTypes[workloadType]; !ok {
		return nil, &ValidationError{Message: "type must be one of RENTAL, VM", Field: "type", Value: r.Type}
	}
	if workloadType == "VM" {
		if r.VMConfig == nil {
			return nil, &ValidationError{Message: "vm_config is required for VM workloads", Field: "vm_config"}
		}
		if len(r.Commands) > 0 || len(r.Args) > 0 || len(r.Envs) > 0 || len(r.Volumes) > 0 || r.RegistryAuth != nil {
			return nil, &ValidationError{Message: "commands, args, envs, volumes, and registry_auth are not supported for VM workloads"}
		}
	} else if r.VMConfig != nil {
		return nil, &ValidationError{Message: "vm_config is only valid for VM workloads", Field: "vm_config"}
	}

	name, err := requireNonEmpty(r.Name, "name")
	if err != nil {
		return nil, err
	}
	image, err := requireNonEmpty(r.Image, "image")
	if err != nil {
		return nil, err
	}
	resourceName, err := requireNonEmpty(r.ResourceName, "resource_name")
	if err != nil {
		return nil, err
	}
	payload := map[string]any{
		"name":          name,
		"image":         image,
		"resource_name": resourceName,
		"type":          workloadType,
	}
	if r.ProjectID != "" {
		payload["project_id"] = r.ProjectID
	}
	if len(r.Ports) > 0 {
		ports := make([]map[string]any, 0, len(r.Ports))
		for _, p := range r.Ports {
			item := map[string]any{"port": p.Port}
			if p.Protocol != "" {
				item["protocol"] = p.Protocol
			}
			if p.Routing != "" {
				item["routing"] = p.Routing
			}
			ports = append(ports, item)
		}
		payload["ports"] = ports
	}
	if len(r.Envs) > 0 {
		payload["envs"] = r.Envs
	}
	if len(r.Commands) > 0 {
		payload["commands"] = r.Commands
	}
	if len(r.Args) > 0 {
		payload["args"] = r.Args
	}
	if r.RegistryAuth != nil {
		auth, err := r.RegistryAuth.toPayload()
		if err != nil {
			return nil, err
		}
		payload["registry_auth"] = auth
	}
	if len(r.Volumes) > 0 {
		vols := make([]map[string]any, 0, len(r.Volumes))
		for _, v := range r.Volumes {
			item, err := v.toPayload()
			if err != nil {
				return nil, err
			}
			vols = append(vols, item)
		}
		payload["volumes"] = vols
	}
	if len(r.SSHKeys) > 0 {
		payload["ssh_keys"] = r.SSHKeys
	}
	if r.VMConfig != nil {
		vm, err := r.VMConfig.toPayload()
		if err != nil {
			return nil, err
		}
		payload["vm_config"] = vm
	}
	return payload, nil
}

// UpdateWorkloadRequest is a partial workload update.
type UpdateWorkloadRequest struct {
	Name         *string
	Image        *string
	ProjectID    *string
	Ports        []PortConfig
	Envs         []EnvVar
	Commands     []string
	Args         []string
	Volumes      []VolumeMount
	SSHKeys      []string
	RegistryAuth *RegistryAuth
}

func (r UpdateWorkloadRequest) toPayload() (map[string]any, error) {
	payload := map[string]any{}
	if r.Name != nil {
		payload["name"] = *r.Name
	}
	if r.Image != nil {
		payload["image"] = *r.Image
	}
	if r.ProjectID != nil {
		payload["project_id"] = *r.ProjectID
	}
	if r.Ports != nil {
		payload["ports"] = r.Ports
	}
	if r.Envs != nil {
		payload["envs"] = r.Envs
	}
	if r.Commands != nil {
		payload["commands"] = r.Commands
	}
	if r.Args != nil {
		payload["args"] = r.Args
	}
	if r.Volumes != nil {
		vols := make([]map[string]any, 0, len(r.Volumes))
		for _, v := range r.Volumes {
			item, err := v.toPayload()
			if err != nil {
				return nil, err
			}
			vols = append(vols, item)
		}
		payload["volumes"] = vols
	}
	if r.SSHKeys != nil {
		payload["ssh_keys"] = r.SSHKeys
	}
	if r.RegistryAuth != nil {
		auth, err := r.RegistryAuth.toPayload()
		if err != nil {
			return nil, err
		}
		payload["registry_auth"] = auth
	}
	return payload, nil
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
	UID               string                `json:"uid"`
	Name              string                `json:"name"`
	Image             string                `json:"image"`
	Type              string                `json:"type"`
	ResourceName      string                `json:"resource_name"`
	ProjectID         *string               `json:"project_id"`
	Ports             []PortConfig          `json:"ports"`
	Envs              []EnvVar              `json:"envs"`
	Commands          []string              `json:"commands"`
	Args              []string              `json:"args"`
	Volumes           []WorkloadVolumeMount `json:"volumes"`
	SSHKeys           []WorkloadSSHKey      `json:"ssh_keys"`
	RegistryAuth      *RegistryAuth         `json:"registry_auth"`
	SandboxConfig     *SandboxConfig        `json:"sandbox_config,omitempty"`
	State             *WorkloadState        `json:"state"`
	Resource          *WorkloadResource     `json:"resource"`
	CostPerHour       *float64              `json:"cost_per_hour"`
	FrozenCostPerHour *float64              `json:"frozen_cost_per_hour,omitempty"`
	Revision          string                `json:"revision"`
	CreatedAt         string                `json:"created_at"`
	UpdatedAt         string                `json:"updated_at"`
}

// WorkloadListItem is a summarized list row.
type WorkloadListItem struct {
	UID               string                `json:"uid"`
	Name              string                `json:"name"`
	Image             string                `json:"image"`
	Type              string                `json:"type"`
	State             *WorkloadState        `json:"state"`
	Resource          *WorkloadResource     `json:"resource"`
	CostPerHour       *float64              `json:"cost_per_hour"`
	FrozenCostPerHour *float64              `json:"frozen_cost_per_hour,omitempty"`
	Revision          string                `json:"revision"`
	Volumes           []WorkloadVolumeMount `json:"volumes"`
	CreatedAt         string                `json:"created_at"`
	UpdatedAt         string                `json:"updated_at"`
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

// VolumeMountResponse is returned when attaching a volume.
type VolumeMountResponse struct {
	WorkloadUID string `json:"workload_uid"`
	UID         string `json:"uid"`
	MountPath   string `json:"mount_path"`
	ReadOnly    bool   `json:"read_only"`
}

// SSHKeyAttachResponse is returned when attaching an SSH key.
type SSHKeyAttachResponse struct {
	WorkloadUID string `json:"workload_uid"`
	SSHKeyUID   string `json:"ssh_key_uid"`
}

// ExecResponse is a captured exec result.
type ExecResponse struct {
	ExitCode int
	Result   string
}

// VMImage is a bootable VM image.
type VMImage struct {
	Name        string `json:"name"`
	DisplayName string `json:"display_name"`
	Description string `json:"description"`
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

func (s *WorkloadService) Create(ctx context.Context, req CreateWorkloadRequest) (*Workload, error) {
	payload, err := req.toPayload()
	if err != nil {
		return nil, err
	}
	path, err := s.path()
	if err != nil {
		return nil, err
	}
	var out Workload
	err = s.client.do(ctx, "POST", path, nil, payload, &out)
	return &out, err
}

func (s *WorkloadService) Get(ctx context.Context, workloadUID string) (*Workload, error) {
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

func (s *WorkloadService) Update(ctx context.Context, workloadUID string, req UpdateWorkloadRequest) (*Workload, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	payload, err := req.toPayload()
	if err != nil {
		return nil, err
	}
	path, err := s.path(workloadUID)
	if err != nil {
		return nil, err
	}
	var out Workload
	err = s.client.do(ctx, "PATCH", path, nil, payload, &out)
	return &out, err
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

func (s *WorkloadService) postAction(ctx context.Context, workloadUID, action string) (*WorkloadOperationResponse, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.path(workloadUID, action)
	if err != nil {
		return nil, err
	}
	var out WorkloadOperationResponse
	err = s.client.do(ctx, "POST", path, nil, nil, &out)
	return &out, err
}

func (s *WorkloadService) Deploy(ctx context.Context, workloadUID string) (*WorkloadOperationResponse, error) {
	return s.postAction(ctx, workloadUID, "deploy")
}

func (s *WorkloadService) Suspend(ctx context.Context, workloadUID string) (*WorkloadOperationResponse, error) {
	return s.postAction(ctx, workloadUID, "suspend")
}

func (s *WorkloadService) Reboot(ctx context.Context, workloadUID string) (*WorkloadOperationResponse, error) {
	return s.postAction(ctx, workloadUID, "reboot")
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

// WaitUntilReady polls until status is running, a terminal state, or timeout.
func (s *WorkloadService) WaitUntilReady(ctx context.Context, workloadUID string, timeout, pollInterval time.Duration) (*WorkloadStateResponse, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	if timeout <= 0 {
		timeout = 300 * time.Second
	}
	if pollInterval <= 0 {
		pollInterval = 5 * time.Second
	}
	var lastStatus string
	return pollUntil(ctx, timeout, pollInterval, func() (*WorkloadStateResponse, bool, error) {
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
			msg := fmt.Sprintf("Workload %s entered terminal state '%s' before becoming ready", workloadUID, state.Status)
			if state.Message != "" {
				msg += ": " + state.Message
			}
			return nil, false, &Error{Message: msg}
		}
		return state, false, nil
	}, func(*WorkloadStateResponse) error {
		return &TimeoutError{
			Message: fmt.Sprintf("Workload %s was not ready within %.0fs (last status: '%s')", workloadUID, timeout.Seconds(), lastStatus),
			Timeout: timeout.Seconds(),
		}
	})
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

func logQuery(opts LogOptions) (url.Values, error) {
	q := url.Values{}
	if opts.Since != "" {
		q.Set("since", opts.Since)
	}
	if opts.Tail > 0 {
		q.Set("tail", strconv.Itoa(opts.Tail))
	}
	if opts.Previous {
		q.Set("previous", "true")
	}
	if opts.LogType != "" {
		normalized, err := requireNonEmpty(opts.LogType, "log_type")
		if err != nil {
			return nil, err
		}
		normalized = strings.ToLower(normalized)
		if _, ok := validLogTypes[normalized]; !ok {
			return nil, &ValidationError{Message: "log_type must be one of qemu, serial", Field: "log_type", Value: opts.LogType}
		}
		q.Set("type", normalized)
	}
	if opts.Follow {
		q.Set("follow", "true")
	}
	return q, nil
}

func (s *WorkloadService) GetLogs(ctx context.Context, workloadUID string, opts LogOptions) (string, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return "", err
	}
	path, err := s.path(workloadUID, "logs")
	if err != nil {
		return "", err
	}
	opts.Follow = false
	q, err := logQuery(opts)
	if err != nil {
		return "", err
	}
	return s.client.getText(ctx, path, q)
}

func (s *WorkloadService) StreamLogs(ctx context.Context, workloadUID string, opts LogOptions) (io.ReadCloser, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.path(workloadUID, "logs")
	if err != nil {
		return nil, err
	}
	opts.Follow = true
	q, err := logQuery(opts)
	if err != nil {
		return nil, err
	}
	return s.client.stream(ctx, "GET", path, q)
}

func (s *WorkloadService) execRaw(ctx context.Context, workloadUID string, command []string) (io.ReadCloser, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	if len(command) == 0 {
		return nil, &ValidationError{Message: "command must not be empty", Field: "command"}
	}
	total := 0
	q := url.Values{}
	for _, arg := range command {
		total += len(arg)
		q.Add("command", arg)
	}
	if total > argMaxBytes {
		return nil, &ValidationError{
			Message: fmt.Sprintf("command arguments exceed %d bytes (ARG_MAX); got %d", argMaxBytes, total),
			Field:   "command",
		}
	}
	path, err := s.path(workloadUID, "exec")
	if err != nil {
		return nil, err
	}
	return s.client.stream(ctx, "POST", path, q)
}

func (s *WorkloadService) ExecStream(ctx context.Context, workloadUID, command string) (io.ReadCloser, error) {
	command, err := requireNonEmpty(command, "command")
	if err != nil {
		return nil, err
	}
	return s.execRaw(ctx, workloadUID, []string{"sh", "-c", command})
}

func (s *WorkloadService) Exec(ctx context.Context, workloadUID, command string) (*ExecResponse, error) {
	command, err := requireNonEmpty(command, "command")
	if err != nil {
		return nil, err
	}
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return nil, err
	}
	sentinel := "__TARGON_EXIT_" + hex.EncodeToString(b[:]) + "__"
	wrapped := command + "\nprintf '\\n%s:%s' '" + sentinel + "' \"$?\""
	body, err := s.execRaw(ctx, workloadUID, []string{"sh", "-c", wrapped})
	if err != nil {
		return nil, err
	}
	defer body.Close()
	chunks, err := readLines(body)
	if err != nil {
		return nil, err
	}
	output := strings.Join(chunks, "\n")
	exitCode := 0
	marker := sentinel + ":"
	idx := strings.LastIndex(output, marker)
	result := output
	if idx != -1 {
		result = strings.TrimRight(output[:idx], "\n")
		tail := strings.TrimSpace(output[idx+len(marker):])
		if n, err := strconv.Atoi(tail); err == nil {
			exitCode = n
		}
	}
	return &ExecResponse{ExitCode: exitCode, Result: result}, nil
}

func readLines(r io.Reader) ([]string, error) {
	var lines []string
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimRight(scanner.Text(), "\r")
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines, scanner.Err()
}

func (s *WorkloadService) Verify(ctx context.Context, uid, digest string) (bool, error) {
	uid, err := requireNonEmpty(uid, "uid")
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
	err = s.client.do(ctx, "POST", path, nil, map[string]string{"uid": uid, "digest": digest}, &out)
	return out.Verified, err
}

func (s *WorkloadService) VMImages(ctx context.Context) ([]VMImage, error) {
	path, err := s.path("vm-images")
	if err != nil {
		return nil, err
	}
	var out []VMImage
	err = s.client.do(ctx, "GET", path, nil, nil, &out)
	return out, err
}

func (s *WorkloadService) AttachVolume(ctx context.Context, workloadUID, volumeUID, mountPath string, readOnly bool) (*VolumeMountResponse, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	volumeUID, err = requireNonEmpty(volumeUID, "volume_uid")
	if err != nil {
		return nil, err
	}
	mountPath, err = requireAbsolutePath(mountPath, "mount_path")
	if err != nil {
		return nil, err
	}
	path, err := s.path(workloadUID, "volumes", volumeUID)
	if err != nil {
		return nil, err
	}
	var out VolumeMountResponse
	err = s.client.do(ctx, "PUT", path, nil, map[string]any{"mount_path": mountPath, "read_only": readOnly}, &out)
	return &out, err
}

func (s *WorkloadService) DetachVolume(ctx context.Context, workloadUID, volumeUID string) error {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return err
	}
	volumeUID, err = requireNonEmpty(volumeUID, "volume_uid")
	if err != nil {
		return err
	}
	path, err := s.path(workloadUID, "volumes", volumeUID)
	if err != nil {
		return err
	}
	return s.client.do(ctx, "DELETE", path, nil, nil, nil)
}

func (s *WorkloadService) AttachSSHKey(ctx context.Context, workloadUID, sshKeyUID string) (*SSHKeyAttachResponse, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	sshKeyUID, err = requireNonEmpty(sshKeyUID, "ssh_key_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.path(workloadUID, "ssh-keys", sshKeyUID)
	if err != nil {
		return nil, err
	}
	var out SSHKeyAttachResponse
	err = s.client.do(ctx, "PUT", path, nil, nil, &out)
	return &out, err
}

func (s *WorkloadService) DetachSSHKey(ctx context.Context, workloadUID, sshKeyUID string) error {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return err
	}
	sshKeyUID, err = requireNonEmpty(sshKeyUID, "ssh_key_uid")
	if err != nil {
		return err
	}
	path, err := s.path(workloadUID, "ssh-keys", sshKeyUID)
	if err != nil {
		return err
	}
	return s.client.do(ctx, "DELETE", path, nil, nil, nil)
}
