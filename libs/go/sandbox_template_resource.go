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

// Refresh reloads this template into the same resource object.
func (t *SandboxTemplate) Refresh(ctx context.Context) (*SandboxTemplate, error) {
	service, err := t.boundService()
	if err != nil {
		return nil, err
	}
	updated, err := service.Get(ctx, t.UID)
	if err != nil {
		return nil, err
	}
	return t.replace(updated), nil
}

// Update changes the template and reloads this resource object.
func (t *SandboxTemplate) Update(ctx context.Context, params UpdateSandboxTemplateParams) (*SandboxTemplate, error) {
	service, err := t.boundService()
	if err != nil {
		return nil, err
	}
	updated, err := service.Update(ctx, t.UID, params)
	if err != nil {
		return nil, err
	}
	return t.replace(updated), nil
}

// Delete removes this template.
func (t *SandboxTemplate) Delete(ctx context.Context) error {
	service, err := t.boundService()
	if err != nil {
		return err
	}
	return service.Delete(ctx, t.UID)
}

// CreateSandbox creates a sandbox from this template.
func (t *SandboxTemplate) CreateSandbox(ctx context.Context, params SandboxCreateParams) (*Sandbox, error) {
	service, err := t.boundService()
	if err != nil {
		return nil, err
	}
	params.Template = t
	return service.client.Sandboxes.Create(ctx, params)
}
