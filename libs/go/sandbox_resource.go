package targon

import (
	"context"
)

func hydrateSandbox(service *SandboxesService, workload *Workload) *Sandbox {
	if workload == nil {
		return nil
	}
	return &Sandbox{Workload: *workload, service: service}
}

func (s *Sandbox) boundService() (*SandboxesService, error) {
	if s == nil || s.service == nil {
		return nil, validation("sandbox is not bound to a client service", "sandbox", nil)
	}
	if _, err := requireNonEmpty(s.UID, "workload_uid"); err != nil {
		return nil, err
	}
	return s.service, nil
}

func (s *Sandbox) replace(updated *Sandbox) *Sandbox {
	if updated != nil {
		*s = *updated
	}
	return s
}

// Refresh reloads the full workload representation into this object.
func (s *Sandbox) Refresh(ctx context.Context) (*Sandbox, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	updated, err := service.Get(ctx, s.UID)
	if err != nil {
		return nil, err
	}
	return s.replace(updated), nil
}

func (s *Sandbox) GetState(ctx context.Context) (*WorkloadStateResponse, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	return service.GetState(ctx, s.UID)
}

func (s *Sandbox) Update(ctx context.Context, params SandboxUpdateParams) (*Sandbox, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	updated, err := service.Update(ctx, s.UID, params)
	if err != nil {
		return nil, err
	}
	return s.replace(updated), nil
}

func (s *Sandbox) Freeze(ctx context.Context, options ...WaitOptions) (*Sandbox, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	updated, err := service.Freeze(ctx, s.UID, options...)
	if err != nil {
		return nil, err
	}
	return s.replace(updated), nil
}

func (s *Sandbox) Thaw(ctx context.Context, options ...WaitOptions) (*Sandbox, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	updated, err := service.Thaw(ctx, s.UID, options...)
	if err != nil {
		return nil, err
	}
	return s.replace(updated), nil
}

func (s *Sandbox) WaitForStatus(ctx context.Context, status string, options WaitOptions) (*Sandbox, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	updated, err := service.WaitForStatus(ctx, s.UID, status, options)
	if err != nil {
		return nil, err
	}
	return s.replace(updated), nil
}

func (s *Sandbox) Delete(ctx context.Context) error {
	service, err := s.boundService()
	if err != nil {
		return err
	}
	return service.Delete(ctx, s.UID)
}

func (s *Sandbox) Fork(ctx context.Context, params ForkSandboxParams) (*Sandbox, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	return service.Fork(ctx, s.UID, params)
}

func (s *Sandbox) Publish(ctx context.Context, params PublishSandboxParams) (*SandboxTemplate, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	return service.Publish(ctx, s.UID, params)
}

func (s *Sandbox) Exec(ctx context.Context, command string, timeoutSec int) (*SandboxExecResult, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	return service.Exec(ctx, s.UID, command, timeoutSec)
}

func (s *Sandbox) ReadFile(ctx context.Context, guestPath string) ([]byte, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	return service.Files.Read(ctx, s.UID, guestPath)
}

func (s *Sandbox) WriteFile(ctx context.Context, guestPath string, data []byte) error {
	service, err := s.boundService()
	if err != nil {
		return err
	}
	return service.Files.Write(ctx, s.UID, guestPath, data)
}

func (s *Sandbox) MintAccessTicket(ctx context.Context, ttlSec int) (*SandboxAccessTicket, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	return service.MintAccessTicket(ctx, s.UID, ttlSec)
}

func (s *Sandbox) GetDesktop(ctx context.Context) (*SandboxDesktopInfo, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	return service.GetDesktop(ctx, s.UID)
}

func (s *Sandbox) ListTerminals(ctx context.Context) ([]TerminalSession, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	return service.Terminals.List(ctx, s.UID)
}

func (s *Sandbox) CreateTerminal(ctx context.Context, cols, rows int) (*TerminalSession, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	return service.Terminals.Create(ctx, s.UID, cols, rows)
}

func (s *Sandbox) DeleteTerminal(ctx context.Context, terminalID string) error {
	service, err := s.boundService()
	if err != nil {
		return err
	}
	return service.Terminals.Delete(ctx, s.UID, terminalID)
}

func (s *Sandbox) ConnectTerminal(ctx context.Context, terminalID string, options ConnectTerminalOptions) (*TerminalConnection, error) {
	service, err := s.boundService()
	if err != nil {
		return nil, err
	}
	return service.Terminals.Connect(ctx, s.UID, terminalID, options)
}

func (s *Sandbox) AttachSSHKey(ctx context.Context, keyUID string) error {
	service, err := s.boundService()
	if err != nil {
		return err
	}
	return service.AttachSSHKey(ctx, s.UID, keyUID)
}

func (s *Sandbox) DetachSSHKey(ctx context.Context, keyUID string) error {
	service, err := s.boundService()
	if err != nil {
		return err
	}
	return service.DetachSSHKey(ctx, s.UID, keyUID)
}
