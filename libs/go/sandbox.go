package targon

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var sandboxNamePattern = regexp.MustCompile(`^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$`)

// SandboxesService manages first-class SANDBOX workloads.
type SandboxesService struct {
	client    *Client
	Templates *SandboxTemplatesService
	Files     *SandboxFilesService
	Terminals *SandboxTerminalsService
}

// Sandbox is a method-bearing sandbox resource. Workload is embedded so
// callers can access the complete workload response, including UID and state.
type Sandbox struct {
	Workload
	service *SandboxesService
}

func (s *SandboxesService) path(parts ...string) (string, error) {
	org, err := s.client.RequireOrg()
	if err != nil {
		return "", err
	}
	p, err := orgPath(org, "workloads")
	if err != nil {
		return "", err
	}
	return joinPath(p, parts...), nil
}

// CreateSandbox is a convenience wrapper around Client.Sandboxes.Create.
func CreateSandbox(ctx context.Context, params SandboxCreateParams) (*Sandbox, error) {
	client := params.Client
	var err error
	if client == nil && params.Template != nil && params.Template.service != nil {
		client = params.Template.service.client
	}
	if client == nil {
		client, err = NewFromEnv()
		if err != nil {
			return nil, err
		}
	}
	if params.Org != "" && client.Org() != params.Org {
		client, err = client.ForOrg(params.Org)
		if err != nil {
			return nil, err
		}
	}
	return client.Sandboxes.Create(ctx, params)
}

func (s *SandboxesService) Create(ctx context.Context, params SandboxCreateParams) (*Sandbox, error) {
	if err := validateSandboxName(params.Name, "name"); err != nil {
		return nil, err
	}
	templateService, err := params.Template.boundService()
	if err != nil {
		return nil, err
	}
	if templateService != s.Templates {
		return nil, validation("sandbox template is bound to a different client service", "template", params.Template.UID)
	}
	templateUID, err := requireNonEmpty(params.Template.UID, "template.uid")
	if err != nil {
		return nil, err
	}
	if !strings.HasPrefix(templateUID, "sbt-") {
		return nil, validation("template.uid must be a sandbox template uid (sbt-...)", "template.uid", templateUID)
	}
	if params.Template.Status != SandboxTemplateStatusReady {
		return nil, validation("sandbox template status must be READY", "template.status", params.Template.Status)
	}
	if err := validateSandboxPorts(params.Ports); err != nil {
		return nil, err
	}
	if err := validateSandboxConfigForCreate(params.SandboxConfig); err != nil {
		return nil, err
	}
	payload := map[string]any{
		"type":  "SANDBOX",
		"name":  params.Name,
		"image": templateUID,
	}
	if params.ProjectID != "" {
		payload["project_id"] = params.ProjectID
	}
	if params.SSHKeys != nil {
		payload["ssh_keys"] = params.SSHKeys
	}
	if params.Ports != nil {
		payload["ports"] = normalizedSandboxPorts(params.Ports)
	}
	if params.SandboxConfig != nil {
		payload["sandbox_config"] = params.SandboxConfig
	}
	path, err := s.path()
	if err != nil {
		return nil, err
	}
	var created Workload
	if err := s.client.doNoRetry(ctx, http.MethodPost, path, nil, payload, &created); err != nil {
		return nil, err
	}
	deployPath, _ := s.path(created.UID, "deploy")
	var deployed WorkloadOperationResponse
	if err := s.client.doNoRetry(ctx, http.MethodPost, deployPath, nil, nil, &deployed); err != nil {
		return nil, err
	}
	if params.Wait.NoWait {
		return s.Get(ctx, created.UID)
	}
	return s.WaitForStatus(ctx, created.UID, "running", params.Wait)
}

func (s *SandboxesService) Get(ctx context.Context, workloadUID string) (*Sandbox, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.path(workloadUID)
	if err != nil {
		return nil, err
	}
	var out Workload
	if err := s.client.do(ctx, http.MethodGet, path, nil, nil, &out); err != nil {
		return nil, err
	}
	if err := requireSandboxWorkload(&out); err != nil {
		return nil, err
	}
	return hydrateSandbox(s, &out), nil
}

func (s *SandboxesService) List(ctx context.Context, params ListSandboxesParams) (List[SandboxSummary], error) {
	if err := validatePage(params.Page); err != nil {
		return List[SandboxSummary]{}, err
	}
	path, err := s.path()
	if err != nil {
		return List[SandboxSummary]{}, err
	}
	q := params.Page.query()
	if params.Page.Limit == 0 {
		q.Set("limit", strconv.Itoa(MaxSandboxListLimit))
	}
	q.Set("type", "SANDBOX")
	if params.Status != "" {
		q.Set("status", params.Status)
	}
	if params.ProjectID != "" {
		q.Set("project_id", params.ProjectID)
	}
	if params.Name != "" {
		q.Set("name", params.Name)
	}
	var out List[SandboxSummary]
	if err := s.client.do(ctx, http.MethodGet, path, q, nil, &out); err != nil {
		return List[SandboxSummary]{}, err
	}
	for i := range out.Items {
		if err := requireSandboxType(out.Items[i].UID, out.Items[i].Type); err != nil {
			return List[SandboxSummary]{}, err
		}
	}
	return out, nil
}

func (s *SandboxesService) GetState(ctx context.Context, workloadUID string) (*WorkloadStateResponse, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.path(workloadUID, "state")
	if err != nil {
		return nil, err
	}
	var out WorkloadStateResponse
	if err := s.client.do(ctx, http.MethodGet, path, nil, nil, &out); err != nil {
		return nil, err
	}
	if err := requireSandboxType(out.UID, out.WorkloadType); err != nil {
		return nil, err
	}
	return &out, nil
}

func (s *SandboxesService) Update(ctx context.Context, workloadUID string, params SandboxUpdateParams) (*Sandbox, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	payload := map[string]any{}
	if params.Name != nil {
		if err := validateSandboxName(*params.Name, "name"); err != nil {
			return nil, err
		}
		payload["name"] = *params.Name
	}
	if params.ProjectID != nil {
		payload["project_id"] = *params.ProjectID
	}
	if params.SSHKeys != nil {
		payload["ssh_keys"] = *params.SSHKeys
	}
	if params.Ports != nil {
		if err := validateSandboxPorts(*params.Ports); err != nil {
			return nil, err
		}
		payload["ports"] = normalizedSandboxPorts(*params.Ports)
	}
	if params.SandboxConfig != nil {
		if err := validateSandboxConfigForUpdate(params.SandboxConfig); err != nil {
			return nil, err
		}
		payload["sandbox_config"] = params.SandboxConfig
	}
	if len(payload) == 0 {
		return nil, validation("at least one sandbox field must be supplied", "update", nil)
	}
	path, err := s.path(workloadUID)
	if err != nil {
		return nil, err
	}
	var out Workload
	if err := s.client.doNoRetry(ctx, http.MethodPatch, path, nil, payload, &out); err != nil {
		return nil, err
	}
	if err := requireSandboxWorkload(&out); err != nil {
		return nil, err
	}
	return hydrateSandbox(s, &out), nil
}

func (s *SandboxesService) Delete(ctx context.Context, workloadUID string) error {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return err
	}
	path, err := s.path(workloadUID)
	if err != nil {
		return err
	}
	return s.client.doNoRetry(ctx, http.MethodDelete, path, nil, nil, nil)
}

func (s *SandboxesService) AttachSSHKey(ctx context.Context, workloadUID, keyUID string) error {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return err
	}
	keyUID, err = requireNonEmpty(keyUID, "ssh_key_uid")
	if err != nil {
		return err
	}
	path, err := s.path(workloadUID, "ssh-keys", keyUID)
	if err != nil {
		return err
	}
	var ignored SSHKeyAttachResponse
	return s.client.doNoRetry(ctx, http.MethodPut, path, nil, nil, &ignored)
}

func (s *SandboxesService) DetachSSHKey(ctx context.Context, workloadUID, keyUID string) error {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return err
	}
	keyUID, err = requireNonEmpty(keyUID, "ssh_key_uid")
	if err != nil {
		return err
	}
	path, err := s.path(workloadUID, "ssh-keys", keyUID)
	if err != nil {
		return err
	}
	return s.client.doNoRetry(ctx, http.MethodDelete, path, nil, nil, nil)
}

func (s *SandboxesService) Freeze(ctx context.Context, workloadUID string, options ...WaitOptions) (*Sandbox, error) {
	return s.lifecycle(ctx, workloadUID, "freeze", "frozen", firstWait(options))
}

func (s *SandboxesService) Thaw(ctx context.Context, workloadUID string, options ...WaitOptions) (*Sandbox, error) {
	return s.lifecycle(ctx, workloadUID, "thaw", "running", firstWait(options))
}

func (s *SandboxesService) lifecycle(ctx context.Context, workloadUID, action, target string, wait WaitOptions) (*Sandbox, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.path(workloadUID, action)
	if err != nil {
		return nil, err
	}
	var ignored WorkloadOperationResponse
	if err := s.client.doNoRetry(ctx, http.MethodPost, path, nil, nil, &ignored); err != nil {
		return nil, err
	}
	if wait.NoWait {
		return s.Get(ctx, workloadUID)
	}
	return s.WaitForStatus(ctx, workloadUID, target, wait)
}

func (s *SandboxesService) Fork(ctx context.Context, workloadUID string, params ForkSandboxParams) (*Sandbox, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	if params.Name != nil && *params.Name != "" {
		if err := validateSandboxName(*params.Name, "name"); err != nil {
			return nil, err
		}
	}
	if err := validateSandboxConfigForCreate(params.SandboxConfig); err != nil {
		return nil, err
	}
	path, err := s.path(workloadUID, "fork")
	if err != nil {
		return nil, err
	}
	var created WorkloadOperationResponse
	if err := s.client.doNoRetry(ctx, http.MethodPost, path, nil, params, &created); err != nil {
		return nil, err
	}
	if params.Wait.NoWait {
		return s.Get(ctx, created.UID)
	}
	return s.WaitForStatus(ctx, created.UID, "running", params.Wait)
}

func (s *SandboxesService) Publish(ctx context.Context, workloadUID string, params PublishSandboxParams) (*SandboxTemplate, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	if err := validateTemplateName(params.Name); err != nil {
		return nil, err
	}
	if len(strings.TrimSpace(params.DisplayName)) > 128 {
		return nil, validation("display_name must be at most 128 characters", "display_name", params.DisplayName)
	}
	path, err := s.path(workloadUID, "publish")
	if err != nil {
		return nil, err
	}
	var out SandboxTemplate
	if err := s.client.doNoRetry(ctx, http.MethodPost, path, nil, params, &out); err != nil {
		return nil, err
	}
	if params.Wait.NoWait {
		return hydrateSandboxTemplate(s.Templates, &out), nil
	}
	return s.Templates.waitUntilReady(ctx, out.UID, workloadUID, params.Wait)
}

func (s *SandboxesService) Exec(ctx context.Context, workloadUID, command string, timeoutSec int) (*SandboxExecResult, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(command) == "" {
		return nil, validation("cmd must be a non-empty string", "cmd", command)
	}
	if len(command) > MaxSandboxExecCommandBytes {
		return nil, validation(fmt.Sprintf("cmd must not exceed %d bytes", MaxSandboxExecCommandBytes), "cmd", nil)
	}
	if timeoutSec == 0 {
		timeoutSec = 60
	}
	if timeoutSec < 1 || timeoutSec > MaxSandboxExecTimeoutSec {
		return nil, validation("timeout_sec must be between 1 and 600", "timeout_sec", timeoutSec)
	}
	path, err := s.path(workloadUID, "exec")
	if err != nil {
		return nil, err
	}
	var out SandboxExecResult
	err = s.client.doNoRetry(ctx, http.MethodPost, path, nil, map[string]any{"cmd": command, "timeout_sec": timeoutSec}, &out)
	return &out, err
}

func (s *SandboxesService) MintAccessTicket(ctx context.Context, workloadUID string, ttlSec int) (*SandboxAccessTicket, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	if ttlSec == 0 {
		ttlSec = 60
	}
	if ttlSec < 1 || ttlSec > MaxSandboxAccessTicketTTL {
		return nil, validation("ttl_sec must be between 1 and 300", "ttl_sec", ttlSec)
	}
	path, err := s.path(workloadUID, "access-tickets")
	if err != nil {
		return nil, err
	}
	var out SandboxAccessTicket
	err = s.client.doNoRetry(ctx, http.MethodPost, path, nil, map[string]int{"ttl_sec": ttlSec}, &out)
	return &out, err
}

func (s *SandboxesService) GetDesktop(ctx context.Context, workloadUID string) (*SandboxDesktopInfo, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.path(workloadUID, "desktop")
	if err != nil {
		return nil, err
	}
	var out SandboxDesktopInfo
	err = s.client.do(ctx, http.MethodGet, path, nil, nil, &out)
	return &out, err
}

func (s *SandboxesService) WaitForStatus(ctx context.Context, workloadUID, target string, options WaitOptions) (*Sandbox, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	target, err = requireNonEmpty(target, "status")
	if err != nil {
		return nil, err
	}
	timeout, interval := waitDurations(options)
	deadline := time.Now().Add(timeout)
	for {
		state, err := s.GetState(ctx, workloadUID)
		if err != nil {
			return nil, err
		}
		status := strings.ToLower(state.Status)
		if status == strings.ToLower(target) {
			return s.Get(ctx, workloadUID)
		}
		if status == "error" || status == "deleted" || status == "suspended" {
			return nil, &SandboxStateError{APIError: APIError{
				StatusCode:  http.StatusConflict,
				Message:     fmt.Sprintf("sandbox %s entered terminal state %q: %s", workloadUID, state.Status, state.Message),
				Reason:      "WORKLOAD_SANDBOX_INVALID_STATE",
				WorkloadUID: workloadUID,
			}}
		}
		if time.Now().After(deadline) {
			return nil, &TimeoutError{
				Message: fmt.Sprintf("sandbox %s did not reach %q within %.0fs (last status: %q)", workloadUID, target, timeout.Seconds(), state.Status),
				Timeout: timeout.Seconds(),
			}
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
}

func requireSandboxWorkload(workload *Workload) error {
	if workload == nil {
		return &SandboxStateError{APIError: APIError{
			StatusCode: http.StatusBadGateway,
			Message:    "sandbox endpoint returned an empty workload",
			Reason:     "WORKLOAD_SANDBOX_TYPE_MISMATCH",
		}}
	}
	return requireSandboxType(workload.UID, workload.Type)
}

func requireSandboxType(workloadUID, workloadType string) error {
	if strings.EqualFold(strings.TrimSpace(workloadType), "SANDBOX") {
		return nil
	}
	return &SandboxStateError{APIError: APIError{
		StatusCode:  http.StatusBadGateway,
		Message:     fmt.Sprintf("expected SANDBOX workload, got %q", workloadType),
		Reason:      "WORKLOAD_SANDBOX_TYPE_MISMATCH",
		WorkloadUID: workloadUID,
	}}
}

func validateSandboxName(name, field string) error {
	if !sandboxNamePattern.MatchString(strings.TrimSpace(name)) {
		return validation("name must be 1-32 lowercase alphanumeric characters or hyphens, without leading or trailing hyphens", field, name)
	}
	return nil
}

func validateTemplateName(name string) error {
	name = strings.TrimSpace(name)
	if ok, _ := regexp.MatchString(`^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$`, name); !ok {
		return validation("name must be 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit", "name", name)
	}
	return nil
}

func validateSandboxConfigForCreate(config *SandboxConfigInput) error {
	if config == nil {
		return nil
	}
	if config.TTLSec != nil && *config.TTLSec < 0 {
		return validation("sandbox_config.ttl_sec must be zero or positive", "sandbox_config.ttl_sec", *config.TTLSec)
	}
	if config.IdleTimeoutSec != nil && *config.IdleTimeoutSec < 0 {
		return validation("sandbox_config.idle_timeout_sec must be zero or positive", "sandbox_config.idle_timeout_sec", *config.IdleTimeoutSec)
	}
	if config.TTLSec != nil && config.IdleTimeoutSec != nil &&
		*config.TTLSec > 0 && *config.IdleTimeoutSec > 0 && *config.IdleTimeoutSec >= *config.TTLSec {
		return validation("sandbox_config.idle_timeout_sec must be less than ttl_sec", "sandbox_config.idle_timeout_sec", *config.IdleTimeoutSec)
	}
	return nil
}

func validateSandboxConfigForUpdate(config *SandboxConfigInput) error {
	if err := validateSandboxConfigForCreate(config); err != nil || config == nil {
		return err
	}
	if config.IdleTimeoutSec != nil && *config.IdleTimeoutSec == 0 {
		return validation("sandbox_config.idle_timeout_sec must be positive", "sandbox_config.idle_timeout_sec", *config.IdleTimeoutSec)
	}
	return nil
}

func validateSandboxPorts(ports []PortConfig) error {
	seen := map[string]struct{}{}
	for _, port := range ports {
		if port.Port < 1 || port.Port > 65535 {
			return validation("sandbox port must be between 1 and 65535", "ports", port.Port)
		}
		if port.Port == 22 {
			return validation("sandbox port 22 is reserved for SSH and cannot be forwarded", "ports", port.Port)
		}
		protocol := strings.ToUpper(strings.TrimSpace(port.Protocol))
		if protocol == "" {
			protocol = "TCP"
		}
		if protocol != "TCP" && protocol != "UDP" {
			return validation("sandbox ports only support TCP or UDP", "ports", port.Protocol)
		}
		key := strconv.Itoa(port.Port) + "/" + protocol
		if _, ok := seen[key]; ok {
			return validation("duplicate sandbox port and protocol combination", "ports", key)
		}
		seen[key] = struct{}{}
	}
	return nil
}

func normalizedSandboxPorts(ports []PortConfig) []PortConfig {
	out := append([]PortConfig(nil), ports...)
	for i := range out {
		out[i].Protocol = strings.ToUpper(strings.TrimSpace(out[i].Protocol))
		if out[i].Protocol == "" {
			out[i].Protocol = "TCP"
		}
		out[i].Routing = ""
	}
	return out
}

func validatePage(page Page) error {
	if page.Limit < 0 || page.Limit > MaxSandboxListLimit {
		return validation("limit must be between 1 and 1000 when supplied", "limit", page.Limit)
	}
	return nil
}

func validation(message, field string, value any) error {
	return &ValidationError{Message: message, Field: field, Value: value}
}

func firstWait(options []WaitOptions) WaitOptions {
	if len(options) > 0 {
		return options[0]
	}
	return WaitOptions{}
}

func waitDurations(options WaitOptions) (time.Duration, time.Duration) {
	timeout, interval := options.Timeout, options.PollInterval
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}
	if interval <= 0 {
		interval = 5 * time.Second
	}
	return timeout, interval
}

func addTicket(rawURL, ticket string) (string, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return "", err
	}
	q := u.Query()
	q.Set("ticket", ticket)
	u.RawQuery = q.Encode()
	return u.String(), nil
}
