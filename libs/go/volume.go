package targon

import "context"

// VolumeState is the runtime state of a volume.
type VolumeState struct {
	Status    string `json:"status"`
	Message   string `json:"message"`
	UpdatedAt string `json:"updated_at"`
}

// Volume is a persistent disk.
type Volume struct {
	UID          string       `json:"uid"`
	Name         string       `json:"name"`
	Size         int64        `json:"size"`
	ResourceName string       `json:"resource_name"`
	State        *VolumeState `json:"state"`
	CostPerHour  *float64     `json:"cost_per_hour"`
	MountPath    *string      `json:"mount_path"`
	WorkloadUID  *string      `json:"workload_uid"`
	PVCName      *string      `json:"pvc_name"`
	LastBackupAt *string      `json:"last_backup_at"`
	DeletedBy    *string      `json:"deleted_by"`
	CreatedAt    string       `json:"created_at"`
	UpdatedAt    string       `json:"updated_at"`
}

// VolumeStateResponse is the dedicated volume state endpoint payload.
type VolumeStateResponse struct {
	UID       string `json:"uid"`
	Status    string `json:"status"`
	Message   string `json:"message"`
	UpdatedAt string `json:"updated_at"`
}

// VolumeEvent is a volume lifecycle event.
type VolumeEvent struct {
	VolumeUID          string  `json:"volume_uid"`
	EventType          string  `json:"event_type"`
	BillingProcessedAt *string `json:"billing_processed_at"`
	BillingStatus      *string `json:"billing_status"`
	CostPerSecond      *int64  `json:"cost_per_second"`
	K8sResourceVersion *string `json:"k8s_resource_version"`
	Namespace          *string `json:"namespace"`
	OldStatus          *string `json:"old_status"`
	NewStatus          *string `json:"new_status"`
	Reason             *string `json:"reason"`
	ResourceName       *string `json:"resource_name"`
	PVCName            *string `json:"pvc_name"`
	RequestedSize      *string `json:"requested_size"`
	StorageClass       *string `json:"storage_class"`
	CreatedAt          string  `json:"created_at"`
}

// VolumeOperationResponse is returned from create.
type VolumeOperationResponse struct {
	UID   string       `json:"uid"`
	State *VolumeState `json:"state"`
}

// VolumeService manages volumes.
type VolumeService struct {
	client *Client
}

func (s *VolumeService) path(parts ...string) (string, error) {
	return s.client.orgResourcePath("volumes", parts...)
}

func (s *VolumeService) Create(ctx context.Context, name string, sizeInMB int, resourceName string) (*VolumeOperationResponse, error) {
	name, err := requireNonEmpty(name, "name")
	if err != nil {
		return nil, err
	}
	resourceName, err = requireNonEmpty(resourceName, "resource_name")
	if err != nil {
		return nil, err
	}
	path, err := s.path()
	if err != nil {
		return nil, err
	}
	var out VolumeOperationResponse
	err = s.client.do(ctx, "POST", path, nil, map[string]any{
		"name":          name,
		"size_in_mb":    sizeInMB,
		"resource_name": resourceName,
	}, &out)
	return &out, err
}

func (s *VolumeService) List(ctx context.Context, page Page, workloadUID string) (List[Volume], error) {
	path, err := s.path()
	if err != nil {
		return List[Volume]{}, err
	}
	q := page.query()
	if workloadUID != "" {
		q.Set("workload_uid", workloadUID)
	}
	var out List[Volume]
	err = s.client.do(ctx, "GET", path, q, nil, &out)
	return out, err
}

func (s *VolumeService) Get(ctx context.Context, volumeUID string) (*Volume, error) {
	volumeUID, err := requireNonEmpty(volumeUID, "volume_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.path(volumeUID)
	if err != nil {
		return nil, err
	}
	var out Volume
	err = s.client.do(ctx, "GET", path, nil, nil, &out)
	return &out, err
}

func (s *VolumeService) GetState(ctx context.Context, volumeUID string) (*VolumeStateResponse, error) {
	volumeUID, err := requireNonEmpty(volumeUID, "volume_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.path(volumeUID, "state")
	if err != nil {
		return nil, err
	}
	var out VolumeStateResponse
	err = s.client.do(ctx, "GET", path, nil, nil, &out)
	return &out, err
}

func (s *VolumeService) GetEvents(ctx context.Context, volumeUID string, page Page) (List[VolumeEvent], error) {
	volumeUID, err := requireNonEmpty(volumeUID, "volume_uid")
	if err != nil {
		return List[VolumeEvent]{}, err
	}
	path, err := s.path(volumeUID, "events")
	if err != nil {
		return List[VolumeEvent]{}, err
	}
	var out List[VolumeEvent]
	err = s.client.do(ctx, "GET", path, page.query(), nil, &out)
	return out, err
}

func (s *VolumeService) Update(ctx context.Context, volumeUID, name string) (*Volume, error) {
	volumeUID, err := requireNonEmpty(volumeUID, "volume_uid")
	if err != nil {
		return nil, err
	}
	name, err = requireNonEmpty(name, "name")
	if err != nil {
		return nil, err
	}
	path, err := s.path(volumeUID)
	if err != nil {
		return nil, err
	}
	var out Volume
	err = s.client.do(ctx, "PATCH", path, nil, map[string]string{"name": name}, &out)
	return &out, err
}

func (s *VolumeService) Delete(ctx context.Context, volumeUID string) error {
	volumeUID, err := requireNonEmpty(volumeUID, "volume_uid")
	if err != nil {
		return err
	}
	path, err := s.path(volumeUID)
	if err != nil {
		return err
	}
	return s.client.do(ctx, "DELETE", path, nil, nil, nil)
}
