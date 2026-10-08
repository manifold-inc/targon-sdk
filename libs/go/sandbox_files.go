package targon

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/http"
	"net/url"
)

type SandboxFilesService struct {
	sandboxes *SandboxesService
}

func (s *SandboxFilesService) Read(ctx context.Context, workloadUID, guestPath string) ([]byte, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	guestPath, err = requireAbsolutePath(guestPath, "path")
	if err != nil {
		return nil, err
	}
	path, err := s.sandboxes.path(workloadUID, "files")
	if err != nil {
		return nil, err
	}
	var out SandboxFileInfo
	if err := s.sandboxes.client.do(ctx, http.MethodGet, path, url.Values{"path": {guestPath}}, nil, &out); err != nil {
		return nil, err
	}
	data, err := base64.StdEncoding.DecodeString(out.ContentB64)
	if err != nil {
		return nil, fmt.Errorf("decode sandbox file content: %w", err)
	}
	return data, nil
}

func (s *SandboxFilesService) Write(ctx context.Context, workloadUID, guestPath string, data []byte) error {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return err
	}
	guestPath, err = requireAbsolutePath(guestPath, "path")
	if err != nil {
		return err
	}
	if len(data) > MaxSandboxFileBytes {
		return validation(fmt.Sprintf("file content must not exceed %d bytes", MaxSandboxFileBytes), "data", len(data))
	}
	path, err := s.sandboxes.path(workloadUID, "files")
	if err != nil {
		return err
	}
	payload := SandboxFileInfo{Path: guestPath, ContentB64: base64.StdEncoding.EncodeToString(data)}
	return s.sandboxes.client.doNoRetry(ctx, http.MethodPut, path, nil, payload, nil)
}
