package targon

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

type SandboxTemplatesService struct {
	client *Client
}

func hydrateSandboxTemplate(service *SandboxTemplatesService, template *SandboxTemplate) *SandboxTemplate {
	if template == nil {
		return nil
	}
	template.service = service
	return template
}

func (s *SandboxTemplatesService) path(parts ...string) (string, error) {
	org, err := s.client.RequireOrg()
	if err != nil {
		return "", err
	}
	p, err := orgPath(org, "sandbox-templates")
	if err != nil {
		return "", err
	}
	return joinPath(p, parts...), nil
}

func (s *SandboxTemplatesService) List(ctx context.Context, params ListSandboxTemplatesParams) (List[SandboxTemplate], error) {
	if err := validatePage(params.Page); err != nil {
		return List[SandboxTemplate]{}, err
	}
	path, err := s.path()
	if err != nil {
		return List[SandboxTemplate]{}, err
	}
	q := params.Page.query()
	if params.Page.Limit == 0 {
		q.Set("limit", strconv.Itoa(MaxSandboxListLimit))
	}
	if params.Kind != "" {
		switch params.Kind {
		case SandboxTemplateKindFresh, SandboxTemplateKindUser:
		default:
			return List[SandboxTemplate]{}, validation("kind must be FRESH or USER", "kind", params.Kind)
		}
		q.Set("kind", string(params.Kind))
	}
	if params.Status != "" {
		switch params.Status {
		case SandboxTemplateStatusPending, SandboxTemplateStatusReady, SandboxTemplateStatusFailed:
		default:
			return List[SandboxTemplate]{}, validation("status must be PENDING, READY or FAILED", "status", params.Status)
		}
		q.Set("status", string(params.Status))
	}
	var out List[SandboxTemplate]
	if err := s.client.do(ctx, http.MethodGet, path, q, nil, &out); err != nil {
		return List[SandboxTemplate]{}, err
	}
	for i := range out.Items {
		hydrateSandboxTemplate(s, &out.Items[i])
	}
	return out, nil
}

func (s *SandboxTemplatesService) Get(ctx context.Context, templateUID string) (*SandboxTemplate, error) {
	templateUID, err := requireNonEmpty(templateUID, "template_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.path(templateUID)
	if err != nil {
		return nil, err
	}
	var out SandboxTemplate
	if err := s.client.do(ctx, http.MethodGet, path, nil, nil, &out); err != nil {
		return nil, err
	}
	return hydrateSandboxTemplate(s, &out), nil
}

func (s *SandboxTemplatesService) Update(ctx context.Context, templateUID string, params UpdateSandboxTemplateParams) (*SandboxTemplate, error) {
	templateUID, err := requireNonEmpty(templateUID, "template_uid")
	if err != nil {
		return nil, err
	}
	if params.DisplayName == nil && params.Description == nil {
		return nil, validation("display_name or description is required", "update", nil)
	}
	if params.DisplayName != nil && len(strings.TrimSpace(*params.DisplayName)) > 128 {
		return nil, validation("display_name must be at most 128 characters", "display_name", *params.DisplayName)
	}
	path, err := s.path(templateUID)
	if err != nil {
		return nil, err
	}
	var out SandboxTemplate
	if err := s.client.doNoRetry(ctx, http.MethodPatch, path, nil, params, &out); err != nil {
		return nil, err
	}
	return hydrateSandboxTemplate(s, &out), nil
}

func (s *SandboxTemplatesService) Delete(ctx context.Context, templateUID string) error {
	templateUID, err := requireNonEmpty(templateUID, "template_uid")
	if err != nil {
		return err
	}
	path, err := s.path(templateUID)
	if err != nil {
		return err
	}
	return s.client.doNoRetry(ctx, http.MethodDelete, path, nil, nil, nil)
}

func (s *SandboxTemplatesService) waitUntilReady(ctx context.Context, templateUID, workloadUID string, options WaitOptions) (*SandboxTemplate, error) {
	timeout, interval := waitDurations(options)
	deadline := time.Now().Add(timeout)
	for {
		template, err := s.Get(ctx, templateUID)
		if err != nil {
			return nil, err
		}
		switch template.Status {
		case SandboxTemplateStatusReady:
			return template, nil
		case SandboxTemplateStatusFailed:
			message := fmt.Sprintf("sandbox template %s failed to publish", templateUID)
			if template.StatusMessage != nil && *template.StatusMessage != "" {
				message += ": " + *template.StatusMessage
			}
			return nil, &SandboxTemplateError{APIError: APIError{
				StatusCode:  http.StatusConflict,
				Message:     message,
				Reason:      "SANDBOX_TEMPLATE_PUBLISH_FAILED",
				WorkloadUID: workloadUID,
			}}
		}
		if time.Now().After(deadline) {
			return nil, &TimeoutError{
				Message: fmt.Sprintf("sandbox template %s was not ready within %.0fs (last status: %q)", templateUID, timeout.Seconds(), template.Status),
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
