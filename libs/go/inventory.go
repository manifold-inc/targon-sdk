package targon

import (
	"context"
	"net/url"
	"regexp"
	"strings"
)

var validInventoryTypes = map[string]struct{}{
	"vm": {},
	"bm": {},
}

var deprecatedInventoryTypes = map[string]struct{}{
	"rental":     {},
	"storage":    {},
	"serverless": {},
}

var regionPattern = regexp.MustCompile(`^[a-z0-9]+(?:-[a-z0-9]+)*$`)

// InventorySpec is the hardware profile of an inventory entry.
type InventorySpec struct {
	GPUVendor     *string `json:"gpu_vendor,omitempty"`
	GPUModel      *string `json:"gpu_model,omitempty"`
	GPUType       *string `json:"gpu_type,omitempty"`
	GPUCount      int     `json:"gpu_count,omitempty"`
	CPUMillicores int     `json:"cpu_millicores,omitempty"`
	CPUVendor     *string `json:"cpu_vendor,omitempty"`
	CPUModel      *string `json:"cpu_model,omitempty"`
	CPUSockets    *int    `json:"cpu_sockets,omitempty"`
	MemoryMiB     int64   `json:"memory_mib,omitempty"`
	DiskSizeMiB   int64   `json:"disk_size_mib,omitempty"`
	DiskLabel     *string `json:"disk_label,omitempty"`
	NetworkMode   *string `json:"network_mode,omitempty"`
	CCEnabled     *bool   `json:"cc_enabled,omitempty"`
	VCPU          int     `json:"vcpu,omitempty"`
	Memory        int64   `json:"memory,omitempty"`
	Storage       int64   `json:"storage,omitempty"`
}

// InventoryRegion is the region an inventory entry is offered in.
type InventoryRegion struct {
	UID         string `json:"uid"`
	DisplayName string `json:"display_name"`
	Country     string `json:"country"`
	Continent   string `json:"continent"`
}

// Inventory is a resource tier available to deploy.
type Inventory struct {
	Name        string           `json:"name"`
	DisplayName string           `json:"display_name"`
	Description string           `json:"description"`
	Type        string           `json:"type"`
	Region      *InventoryRegion `json:"region,omitempty"`
	GPU         bool             `json:"gpu"`
	Spec        InventorySpec    `json:"spec"`
	CostPerHour float64          `json:"cost_per_hour"`
	Available   int              `json:"available"`
}

// InventoryListParams filters an inventory listing. Zero values are omitted.
type InventoryListParams struct {
	// Type is "vm" or "bm".
	Type string
	GPU  *bool
	// Region is a region UID such as the value of Inventory.Region.UID.
	Region string
}

// InventoryService queries available resource capacity.
type InventoryService struct {
	client *Client
}

func (p InventoryListParams) query() (url.Values, error) {
	q := url.Values{}
	if p.Type != "" {
		normalized := strings.ToLower(strings.TrimSpace(p.Type))
		if normalized == "" {
			return nil, validation("type must be a non-empty string", "type", p.Type)
		}
		if _, ok := deprecatedInventoryTypes[normalized]; ok {
			return nil, validation("inventory type "+normalized+" has been deprecated", "type", p.Type)
		}
		if _, ok := validInventoryTypes[normalized]; !ok {
			return nil, validation("type must be one of vm, bm", "type", p.Type)
		}
		q.Set("type", normalized)
	}
	if p.GPU != nil {
		if *p.GPU {
			q.Set("gpu", "true")
		} else {
			q.Set("gpu", "false")
		}
	}
	if p.Region != "" {
		region := strings.ToLower(strings.TrimSpace(p.Region))
		if len(region) > 32 || !regionPattern.MatchString(region) {
			return nil, validation("region must be a lowercase slug no longer than 32 characters", "region", p.Region)
		}
		q.Set("region", region)
	}
	return q, nil
}

// List returns inventory entries, optionally filtered by type, GPU support,
// and region.
func (s *InventoryService) List(ctx context.Context, params InventoryListParams) ([]Inventory, error) {
	q, err := params.query()
	if err != nil {
		return nil, err
	}
	var out []Inventory
	if err := s.client.do(ctx, "GET", APIVersion+"/inventory", q, nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Capacity is an alias for [InventoryService.List].
func (s *InventoryService) Capacity(ctx context.Context, params InventoryListParams) ([]Inventory, error) {
	return s.List(ctx, params)
}
