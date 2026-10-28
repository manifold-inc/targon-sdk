package targon

import "context"

func (t *SandboxTemplate) boundService() (*SandboxTemplatesService, error) {
	if t == nil || t.service == nil {
		return nil, validation("sandbox template is not bound to a client service", "template", nil)
	}
	if _, err := requireNonEmpty(t.UID, "template.uid"); err != nil {
		return nil, err
	}
	return t.service, nil
}

func (t *SandboxTemplate) replace(updated *SandboxTemplate) *SandboxTemplate {
	if updated != nil {
		*t = *updated
	}
	return t
}

func sandboxTemplateCall[T any](t *SandboxTemplate, call func(*SandboxTemplatesService, string) (T, error)) (T, error) {
	service, err := t.boundService()
	if err != nil {
		var zero T
		return zero, err
	}
	return call(service, t.UID)
}

func (t *SandboxTemplate) mutate(call func(*SandboxTemplatesService, string) (*SandboxTemplate, error)) (*SandboxTemplate, error) {
	updated, err := sandboxTemplateCall(t, call)
	if err != nil {
		return nil, err
	}
	return t.replace(updated), nil
}

// Refresh reloads this template into the same resource object.
func (t *SandboxTemplate) Refresh(ctx context.Context) (*SandboxTemplate, error) {
	return t.mutate(func(service *SandboxTemplatesService, uid string) (*SandboxTemplate, error) {
		return service.Get(ctx, uid)
	})
}

// Update changes the template and reloads this resource object.
func (t *SandboxTemplate) Update(ctx context.Context, params UpdateSandboxTemplateParams) (*SandboxTemplate, error) {
	return t.mutate(func(service *SandboxTemplatesService, uid string) (*SandboxTemplate, error) {
		return service.Update(ctx, uid, params)
	})
}

// Delete removes this template.
func (t *SandboxTemplate) Delete(ctx context.Context) error {
	_, err := sandboxTemplateCall(t, func(service *SandboxTemplatesService, uid string) (struct{}, error) {
		return struct{}{}, service.Delete(ctx, uid)
	})
	return err
}

// CreateSandbox creates a sandbox from this template.
func (t *SandboxTemplate) CreateSandbox(ctx context.Context, params SandboxCreateParams) (*Sandbox, error) {
	return sandboxTemplateCall(t, func(service *SandboxTemplatesService, _ string) (*Sandbox, error) {
		params.Template = t
		return service.client.Sandboxes.Create(ctx, params)
	})
}
