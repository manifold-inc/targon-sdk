package targon

import (
	"context"
	"fmt"
	"net/http"
	"strings"
)

// terminalWorkloadStates are backend statuses that fail common waits. stopped
// and frozen are deliberately excluded: they can transition later (start/thaw),
// so waiting through them is valid. Typed waits check their requested target
// before this set.
var terminalWorkloadStates = map[string]struct{}{"error": {}, "deleted": {}, "suspended": {}}

type typedWorkloadService interface {
	path(parts ...string) (string, error)
	get(context.Context, string) (*Workload, error)
}

func registerTyped(
	ctx context.Context,
	client *Client,
	payload any,
	expectedType string,
) (*Workload, error) {
	service := typedWorkloadService(client.Workloads)
	registered, err := registerWorkload(ctx, client, service, payload)
	if err != nil {
		return nil, err
	}
	return getTyped(ctx, service, registered.UID, expectedType)
}

func registerWorkload(
	ctx context.Context,
	client *Client,
	service typedWorkloadService,
	payload any,
) (*Workload, error) {
	path, err := service.path()
	if err != nil {
		return nil, err
	}
	var registered Workload
	if err := client.doNoRetry(ctx, http.MethodPost, path, nil, payload, &registered); err != nil {
		return nil, err
	}
	return &registered, nil
}

func deployTyped(
	ctx context.Context,
	client *Client,
	uid string,
	expectedType string,
	wait WaitOptions,
) (*Workload, error) {
	uid, err := requireNonEmpty(uid, "workload_uid")
	if err != nil {
		return nil, err
	}
	service := typedWorkloadService(client.Workloads)
	path, err := service.path(uid, "deploy")
	if err != nil {
		return nil, err
	}
	var ignored WorkloadOperationResponse
	if err := client.doNoRetry(ctx, http.MethodPost, path, nil, nil, &ignored); err != nil {
		return nil, err
	}
	if wait.NoWait {
		return getTyped(ctx, service, uid, expectedType)
	}
	return waitTyped(ctx, client, uid, expectedType, "running", wait)
}

func waitTyped(
	ctx context.Context,
	client *Client,
	uid string,
	expectedType string,
	target string,
	options WaitOptions,
) (*Workload, error) {
	uid, err := requireNonEmpty(uid, "workload_uid")
	if err != nil {
		return nil, err
	}
	target, err = requireNonEmpty(target, "status")
	if err != nil {
		return nil, err
	}
	timeout, interval := waitDurations(options)
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	lastStatus := ""
	service := typedWorkloadService(client.Workloads)
	return pollUntil(ctx, interval, func() (*Workload, bool, error) {
		state, err := client.Workloads.GetState(ctx, uid)
		if err != nil {
			return nil, false, err
		}
		if err := requireTypedType(state.UID, state.WorkloadType, expectedType); err != nil {
			return nil, false, err
		}
		lastStatus = state.Status
		status := strings.ToLower(state.Status)
		if status == strings.ToLower(target) {
			workload, err := getTyped(ctx, service, uid, expectedType)
			return workload, true, err
		}
		if _, ok := terminalWorkloadStates[status]; ok {
			return nil, false, typedStateError(uid, expectedType, state.Status, state.Message)
		}
		return nil, false, nil
	}, func(*Workload) error {
		return typedTimeoutError(uid, expectedType, target, lastStatus, timeout.Seconds())
	})
}

func getTyped(
	ctx context.Context,
	service typedWorkloadService,
	uid string,
	expectedType string,
) (*Workload, error) {
	workload, err := service.get(ctx, uid)
	if err != nil {
		return nil, err
	}
	if err := requireTypedWorkload(workload, expectedType); err != nil {
		return nil, err
	}
	return workload, nil
}

func requireTypedWorkload(workload *Workload, expectedType string) error {
	if strings.EqualFold(strings.TrimSpace(expectedType), "SANDBOX") {
		return requireSandboxWorkload(workload)
	}
	if workload == nil {
		return &APIError{
			StatusCode: http.StatusBadGateway,
			Message:    fmt.Sprintf("%s endpoint returned an empty workload", expectedType),
			Reason:     "WORKLOAD_TYPE_MISMATCH",
		}
	}
	return requireTypedType(workload.UID, workload.Type, expectedType)
}

func requireTypedType(workloadUID, workloadType, expectedType string) error {
	if strings.EqualFold(strings.TrimSpace(expectedType), "SANDBOX") {
		return requireSandboxType(workloadUID, workloadType)
	}
	if strings.EqualFold(strings.TrimSpace(workloadType), strings.TrimSpace(expectedType)) {
		return nil
	}
	return &APIError{
		StatusCode:  http.StatusBadGateway,
		Message:     fmt.Sprintf("expected %s workload, got %q", expectedType, workloadType),
		Reason:      "WORKLOAD_TYPE_MISMATCH",
		WorkloadUID: workloadUID,
	}
}

func typedStateError(workloadUID, expectedType, status, message string) error {
	if strings.EqualFold(strings.TrimSpace(expectedType), "SANDBOX") {
		return &SandboxStateError{APIError: APIError{
			StatusCode:  http.StatusConflict,
			Message:     fmt.Sprintf("sandbox %s entered terminal state %q: %s", workloadUID, status, message),
			Reason:      "WORKLOAD_SANDBOX_INVALID_STATE",
			WorkloadUID: workloadUID,
		}}
	}
	return &APIError{
		StatusCode:  http.StatusConflict,
		Message:     fmt.Sprintf("workload %s entered terminal state %q: %s", workloadUID, status, message),
		Reason:      "WORKLOAD_INVALID_STATE",
		WorkloadUID: workloadUID,
	}
}

func typedTimeoutError(workloadUID, expectedType, target, lastStatus string, timeoutSeconds float64) error {
	resource := "workload"
	if strings.EqualFold(strings.TrimSpace(expectedType), "SANDBOX") {
		resource = "sandbox"
	}
	return &TimeoutError{
		Message: fmt.Sprintf("%s %s did not reach %q within %.0fs (last status: %q)", resource, workloadUID, target, timeoutSeconds, lastStatus),
		Timeout: timeoutSeconds,
	}
}
