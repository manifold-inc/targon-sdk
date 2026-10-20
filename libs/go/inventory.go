package targon

import (
	"context"
	"net/url"
	"strings"
)

var validInventoryTypes = map[string]struct{}{
	"rental":  {},
	"storage": {},
	"vm":      {},
}

var deprecatedInventoryTypes = map[string]struct{}{
	"serverless": {},
}

// InventorySpec is the hardware profile of an inventory entry.
type InventorySpec struct {
	GPUType  *string `json:"gpu_type"`
	GPUCount int     `json:"gpu_count"`
	VCPU     int     `json:"vcpu"`
	Memory   int     `json:"memory"`
	Storage  int     `json:"storage"`
}

// Inventory is a resource tier available to deploy.
type Inventory struct {
	Name        string        `json:"name"`
	DisplayName string        `json:"display_name"`
	Description string        `json:"description"`
	Type        string        `json:"type"`
	GPU         bool          `json:"gpu"`
	Spec        InventorySpec `json:"spec"`
	CostPerHour float64       `json:"cost_per_hour"`
	Available   int           `json:"available"`
}

// InventoryService queries available resource capacity.
type InventoryService struct {
	client *Client
}

// List returns inventory entries, optionally filtered by type and GPU support.
func (s *InventoryService) List(ctx context.Context, inventoryType string, gpu *bool) ([]Inventory, error) {
	q := url.Values{}
	if inventoryType != "" {
		normalized := strings.ToLower(strings.TrimSpace(inventoryType))
		if normalized == "" {
			return nil, &ValidationError{Message: "inventory_type must be a non-empty string", Field: "inventory_type"}
		}
		if _, ok := deprecatedInventoryTypes[normalized]; ok {
			return nil, &ValidationError{Message: "inventory type " + normalized + " has been deprecated", Field: "inventory_type", Value: inventoryType}
		}
		if _, ok := validInventoryTypes[normalized]; !ok {
			return nil, &ValidationError{Message: "inventory_type must be one of rental, storage, vm", Field: "inventory_type", Value: inventoryType}
		}
		q.Set("type", normalized)
	}
	if gpu != nil {
		if *gpu {
			q.Set("gpu", "true")
		} else {
			q.Set("gpu", "false")
		}
	}
	var out []Inventory
	if err := s.client.do(ctx, "GET", APIVersion+"/inventory", q, nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Capacity is an alias for [InventoryService.List].
func (s *InventoryService) Capacity(ctx context.Context, inventoryType string, gpu *bool) ([]Inventory, error) {
	return s.List(ctx, inventoryType, gpu)
}
