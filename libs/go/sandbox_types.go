package targon

import (
	"encoding/json"
	"fmt"
	"time"
)

const (
	MaxSandboxListLimit         = 1000
	MaxSandboxExecCommandBytes  = 64 << 10
	MaxSandboxExecTimeoutSec    = 600
	MaxSandboxFileBytes         = 256 << 20
	MaxSandboxAccessTicketTTL   = 300
	MaxSandboxTerminalDimension = 1000
)

type SandboxTemplateKind string

const (
	SandboxTemplateKindFresh SandboxTemplateKind = "FRESH"
	SandboxTemplateKindUser  SandboxTemplateKind = "USER"
)

func (k *SandboxTemplateKind) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	switch SandboxTemplateKind(value) {
	case SandboxTemplateKindFresh, SandboxTemplateKindUser:
		*k = SandboxTemplateKind(value)
		return nil
	default:
		return fmt.Errorf("unknown sandbox template kind %q", value)
	}
}

type SandboxTemplateStatus string

const (
	SandboxTemplateStatusPending SandboxTemplateStatus = "PENDING"
	SandboxTemplateStatusReady   SandboxTemplateStatus = "READY"
	SandboxTemplateStatusFailed  SandboxTemplateStatus = "FAILED"
)

func (s *SandboxTemplateStatus) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	switch SandboxTemplateStatus(value) {
	case SandboxTemplateStatusPending, SandboxTemplateStatusReady, SandboxTemplateStatusFailed:
		*s = SandboxTemplateStatus(value)
		return nil
	default:
		return fmt.Errorf("unknown sandbox template status %q", value)
	}
}

type SandboxTemplate struct {
	UID               string                `json:"uid"`
	Name              string                `json:"name"`
	DisplayName       *string               `json:"display_name,omitempty"`
	Description       *string               `json:"description,omitempty"`
	Kind              SandboxTemplateKind   `json:"kind"`
	Status            SandboxTemplateStatus `json:"status"`
	StatusMessage     *string               `json:"status_message,omitempty"`
	ResourceName      string                `json:"resource_name"`
	Resource          *WorkloadResource     `json:"resource,omitempty"`
	CostPerHour       *float64              `json:"cost_per_hour,omitempty"`
	FrozenCostPerHour *float64              `json:"frozen_cost_per_hour,omitempty"`
	DesktopPort       *int                  `json:"desktop_port,omitempty"`
	SourceWorkloadUID *string               `json:"source_workload_uid,omitempty"`
	CreatedAt         time.Time             `json:"created_at"`
	UpdatedAt         time.Time             `json:"updated_at"`
	service           *SandboxTemplatesService
}

// SandboxConfigInput contains caller-controlled sandbox timeout settings.
// A zero TTL disables the TTL. Idle timeout, when supplied, must be positive.
type SandboxConfigInput struct {
	TTLSec         *int `json:"ttl_sec,omitempty"`
	IdleTimeoutSec *int `json:"idle_timeout_sec,omitempty"`
}

// SandboxConfig is the read model returned on sandbox workloads.
type SandboxConfig struct {
	TemplateUID       string     `json:"template_uid"`
	ParentWorkloadUID *string    `json:"parent_workload_uid,omitempty"`
	TTLSec            *int       `json:"ttl_sec,omitempty"`
	IdleTimeoutSec    *int       `json:"idle_timeout_sec,omitempty"`
	ExpiresAt         *time.Time `json:"expires_at,omitempty"`
	LastActivityAt    *time.Time `json:"last_activity_at,omitempty"`
}

type SandboxCreateParams struct {
	Name          string
	Template      *SandboxTemplate
	ProjectID     string
	SSHKeys       []string
	Ports         []PortConfig
	SandboxConfig *SandboxConfigInput
	Wait          WaitOptions

	// Client and Org are only used by the package-level CreateSandbox helper.
	Client *Client
	Org    string
}

type SandboxUpdateParams struct {
	Name          *string
	ProjectID     *string
	SSHKeys       *[]string
	Ports         *[]PortConfig
	SandboxConfig *SandboxConfigInput
}

type ListSandboxesParams struct {
	Page      Page
	Status    string
	ProjectID string
	Name      string
}

// SandboxSummary is the intentionally sparse workload-list representation.
// Call Sandboxes.Get with UID to obtain a method-bearing full Sandbox.
type SandboxSummary struct {
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

type ListSandboxTemplatesParams struct {
	Page   Page
	Kind   SandboxTemplateKind
	Status SandboxTemplateStatus
}

type UpdateSandboxTemplateParams struct {
	DisplayName *string `json:"display_name,omitempty"`
	Description *string `json:"description,omitempty"`
}

type ForkSandboxParams struct {
	Name          *string             `json:"name,omitempty"`
	ProjectID     *string             `json:"project_id,omitempty"`
	SandboxConfig *SandboxConfigInput `json:"sandbox_config,omitempty"`
	Wait          WaitOptions         `json:"-"`
}

type PublishSandboxParams struct {
	Name        string      `json:"name"`
	DisplayName string      `json:"display_name,omitempty"`
	Description string      `json:"description,omitempty"`
	Wait        WaitOptions `json:"-"`
}

// WaitOptions controls lifecycle polling. Lifecycle methods wait by default;
// set NoWait to return after the mutating request succeeds.
type WaitOptions struct {
	NoWait       bool
	Timeout      time.Duration
	PollInterval time.Duration
}

type SandboxExecResult struct {
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
	Code     int    `json:"code"`
	TimedOut bool   `json:"timed_out"`
}

type SandboxFileInfo struct {
	Path       string `json:"path"`
	ContentB64 string `json:"content_b64"`
}

type SandboxAccessTicket struct {
	Ticket    string    `json:"ticket"`
	ExpiresAt time.Time `json:"expires_at"`
}

type TerminalSession struct {
	ID        string    `json:"id"`
	PID       int       `json:"pid"`
	StartedAt time.Time `json:"started_at"`
	Exited    bool      `json:"exited"`
	ExitCode  *int      `json:"exit_code,omitempty"`
}

type SandboxDesktopInfo struct {
	Available bool   `json:"available"`
	Port      int    `json:"port,omitempty"`
	Listening *bool  `json:"listening,omitempty"`
	WSURL     string `json:"ws_url,omitempty"`
}

type ConnectTerminalOptions struct {
	// UseBearer sends the client's bearer token instead of minting a one-time
	// ticket. Ticket authentication is the default and is safer for URLs.
	UseBearer bool
	TicketTTL int
}
