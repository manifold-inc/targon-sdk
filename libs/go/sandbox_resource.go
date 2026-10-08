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

func sandboxCall[T any](s *Sandbox, call func(*SandboxesService, string) (T, error)) (T, error) {
	service, err := s.boundService()
	if err != nil {
		var zero T
		return zero, err
	}
	return call(service, s.UID)
}

func (s *Sandbox) mutate(call func(*SandboxesService, string) (*Sandbox, error)) (*Sandbox, error) {
	updated, err := sandboxCall(s, call)
	if err != nil {
		return nil, err
	}
	return s.replace(updated), nil
}

func (s *Sandbox) do(call func(*SandboxesService, string) error) error {
	_, err := sandboxCall(s, func(service *SandboxesService, uid string) (struct{}, error) {
		return struct{}{}, call(service, uid)
	})
	return err
}

// Refresh reloads the full workload representation into this object.
func (s *Sandbox) Refresh(ctx context.Context) (*Sandbox, error) {
	return s.mutate(func(service *SandboxesService, uid string) (*Sandbox, error) {
		return service.Get(ctx, uid)
	})
}

func (s *Sandbox) GetState(ctx context.Context) (*WorkloadStateResponse, error) {
	return sandboxCall(s, func(service *SandboxesService, uid string) (*WorkloadStateResponse, error) {
		return service.GetState(ctx, uid)
	})
}

func (s *Sandbox) Update(ctx context.Context, params SandboxUpdateParams) (*Sandbox, error) {
	return s.mutate(func(service *SandboxesService, uid string) (*Sandbox, error) {
		return service.Update(ctx, uid, params)
	})
}

func (s *Sandbox) Freeze(ctx context.Context, options ...WaitOptions) (*Sandbox, error) {
	return s.mutate(func(service *SandboxesService, uid string) (*Sandbox, error) {
		return service.Freeze(ctx, uid, options...)
	})
}

func (s *Sandbox) Thaw(ctx context.Context, options ...WaitOptions) (*Sandbox, error) {
	return s.mutate(func(service *SandboxesService, uid string) (*Sandbox, error) {
		return service.Thaw(ctx, uid, options...)
	})
}

func (s *Sandbox) WaitForStatus(ctx context.Context, status string, options WaitOptions) (*Sandbox, error) {
	return s.mutate(func(service *SandboxesService, uid string) (*Sandbox, error) {
		return service.WaitForStatus(ctx, uid, status, options)
	})
}

func (s *Sandbox) Delete(ctx context.Context) error {
	return s.do(func(service *SandboxesService, uid string) error {
		return service.Delete(ctx, uid)
	})
}

func (s *Sandbox) Fork(ctx context.Context, params ForkSandboxParams) (*Sandbox, error) {
	return sandboxCall(s, func(service *SandboxesService, uid string) (*Sandbox, error) {
		return service.Fork(ctx, uid, params)
	})
}

func (s *Sandbox) Publish(ctx context.Context, params PublishSandboxParams) (*SandboxTemplate, error) {
	return sandboxCall(s, func(service *SandboxesService, uid string) (*SandboxTemplate, error) {
		return service.Publish(ctx, uid, params)
	})
}

func (s *Sandbox) Exec(ctx context.Context, command string, timeoutSec int) (*SandboxExecResult, error) {
	return sandboxCall(s, func(service *SandboxesService, uid string) (*SandboxExecResult, error) {
		return service.Exec(ctx, uid, command, timeoutSec)
	})
}

func (s *Sandbox) ReadFile(ctx context.Context, guestPath string) ([]byte, error) {
	return sandboxCall(s, func(service *SandboxesService, uid string) ([]byte, error) {
		return service.Files.Read(ctx, uid, guestPath)
	})
}

func (s *Sandbox) WriteFile(ctx context.Context, guestPath string, data []byte) error {
	return s.do(func(service *SandboxesService, uid string) error {
		return service.Files.Write(ctx, uid, guestPath, data)
	})
}

func (s *Sandbox) MintAccessTicket(ctx context.Context, ttlSec int) (*SandboxAccessTicket, error) {
	return sandboxCall(s, func(service *SandboxesService, uid string) (*SandboxAccessTicket, error) {
		return service.MintAccessTicket(ctx, uid, ttlSec)
	})
}

func (s *Sandbox) GetDesktop(ctx context.Context) (*SandboxDesktopInfo, error) {
	return sandboxCall(s, func(service *SandboxesService, uid string) (*SandboxDesktopInfo, error) {
		return service.GetDesktop(ctx, uid)
	})
}

func (s *Sandbox) ListTerminals(ctx context.Context) ([]TerminalSession, error) {
	return sandboxCall(s, func(service *SandboxesService, uid string) ([]TerminalSession, error) {
		return service.Terminals.List(ctx, uid)
	})
}

func (s *Sandbox) CreateTerminal(ctx context.Context, cols, rows int) (*TerminalSession, error) {
	return sandboxCall(s, func(service *SandboxesService, uid string) (*TerminalSession, error) {
		return service.Terminals.Create(ctx, uid, cols, rows)
	})
}

func (s *Sandbox) DeleteTerminal(ctx context.Context, terminalID string) error {
	return s.do(func(service *SandboxesService, uid string) error {
		return service.Terminals.Delete(ctx, uid, terminalID)
	})
}

func (s *Sandbox) ConnectTerminal(ctx context.Context, terminalID string, options ConnectTerminalOptions) (*TerminalConnection, error) {
	return sandboxCall(s, func(service *SandboxesService, uid string) (*TerminalConnection, error) {
		return service.Terminals.Connect(ctx, uid, terminalID, options)
	})
}

func (s *Sandbox) AttachSSHKey(ctx context.Context, keyUID string) error {
	return s.do(func(service *SandboxesService, uid string) error {
		return service.AttachSSHKey(ctx, uid, keyUID)
	})
}

func (s *Sandbox) DetachSSHKey(ctx context.Context, keyUID string) error {
	return s.do(func(service *SandboxesService, uid string) error {
		return service.DetachSSHKey(ctx, uid, keyUID)
	})
}
