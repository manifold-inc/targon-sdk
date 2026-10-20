package targon

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// Error is the base Targon SDK error.
type Error struct {
	Message string
	Details map[string]any
	Cause   error
}

func (e *Error) Error() string {
	if e == nil {
		return ""
	}
	if len(e.Details) == 0 {
		return e.Message
	}
	parts := make([]string, 0, len(e.Details))
	for k, v := range e.Details {
		parts = append(parts, fmt.Sprintf("%s=%v", k, v))
	}
	return fmt.Sprintf("%s (%s)", e.Message, strings.Join(parts, ", "))
}

func (e *Error) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

// APIError is returned for non-success HTTP responses.
type APIError struct {
	StatusCode  int
	Message     string
	Reason      string
	RequestID   string
	WorkloadUID string
	Cause       error
}

func (e *APIError) Error() string {
	if e == nil {
		return ""
	}
	if e.Reason != "" {
		return fmt.Sprintf("api error %d (%s): %s", e.StatusCode, e.Reason, e.Message)
	}
	return fmt.Sprintf("api error %d: %s", e.StatusCode, e.Message)
}

func (e *APIError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

func (e *APIError) IsRetryable() bool {
	return e != nil && (e.StatusCode >= 500 || e.StatusCode == http.StatusTooManyRequests)
}

func (e *APIError) IsNotFound() bool     { return e != nil && e.StatusCode == http.StatusNotFound }
func (e *APIError) IsUnauthorized() bool { return e != nil && e.StatusCode == http.StatusUnauthorized }
func (e *APIError) IsForbidden() bool    { return e != nil && e.StatusCode == http.StatusForbidden }
func (e *APIError) IsRateLimited() bool {
	return e != nil && e.StatusCode == http.StatusTooManyRequests
}
func (e *APIError) IsClientError() bool { return e != nil && e.StatusCode >= 400 && e.StatusCode < 500 }
func (e *APIError) IsServerError() bool { return e != nil && e.StatusCode >= 500 && e.StatusCode < 600 }

func mapAPIError(status int, message, reason, requestID string, workloadUID ...string) error {
	err := &APIError{StatusCode: status, Message: message, Reason: reason, RequestID: requestID}
	if len(workloadUID) > 0 {
		err.WorkloadUID = workloadUID[0]
	}
	category := sandboxErrorCategory(reason, status)
	if category == nil && err.WorkloadUID != "" {
		switch status {
		case http.StatusRequestEntityTooLarge:
			category = ErrPayloadTooLarge
		case http.StatusBadGateway:
			category = ErrGateway
		case http.StatusServiceUnavailable:
			category = ErrSandboxUnavailable
		}
	}
	switch category {
	case ErrSandboxUnavailable:
		return &SandboxUnavailableError{APIError: *err}
	case ErrSandboxState:
		return &SandboxStateError{APIError: *err}
	case ErrSandboxTemplate:
		return &SandboxTemplateError{APIError: *err}
	case ErrTerminalLimit:
		return &TerminalLimitError{APIError: *err}
	case ErrAccessTicket:
		return &AccessTicketError{APIError: *err}
	case ErrPayloadTooLarge:
		return &PayloadTooLargeError{APIError: *err}
	case ErrGateway:
		return &GatewayError{APIError: *err}
	}
	switch status {
	case http.StatusUnauthorized:
		return &AuthenticationError{APIError: *err}
	case http.StatusForbidden:
		return &AuthorizationError{APIError: *err}
	case http.StatusNotFound:
		return &ResourceNotFoundError{APIError: *err}
	case http.StatusTooManyRequests:
		return &RateLimitError{APIError: *err}
	default:
		return err
	}
}

var (
	ErrSandboxUnavailable = errors.New("sandbox unavailable")
	ErrSandboxState       = errors.New("invalid sandbox state")
	ErrSandboxTemplate    = errors.New("sandbox template error")
	ErrTerminalLimit      = errors.New("sandbox terminal limit reached")
	ErrAccessTicket       = errors.New("sandbox access ticket error")
	ErrPayloadTooLarge    = errors.New("sandbox payload too large")
	ErrGateway            = errors.New("sandbox gateway error")
)

func sandboxErrorCategory(reason string, status int) error {
	if strings.HasPrefix(reason, "SANDBOX_TEMPLATE_") ||
		strings.HasPrefix(reason, "WORKLOAD_SANDBOX_TEMPLATE_") {
		return ErrSandboxTemplate
	}
	switch reason {
	case "WORKLOAD_SANDBOX_UNAVAILABLE", "WORKLOAD_SANDBOX_NO_CAPACITY", "WORKLOAD_SANDBOX_NO_HOST_PORTS":
		return ErrSandboxUnavailable
	case "WORKLOAD_SANDBOX_INVALID_STATE", "WORKLOAD_SANDBOX_NOT_DEPLOYED",
		"WORKLOAD_SANDBOX_NOT_RUNNING", "WORKLOAD_SANDBOX_PARENT_INVALID_STATE",
		"WORKLOAD_SANDBOX_PARENT_NOT_DEPLOYED", "WORKLOAD_SANDBOX_CREATE_CONFLICT":
		return ErrSandboxState
	case "WORKLOAD_SANDBOX_SESSION_LIMIT":
		return ErrTerminalLimit
	case "WORKLOAD_ACCESS_TICKET_TTL_INVALID", "WORKLOAD_ACCESS_TICKET_STORE_FAILED":
		return ErrAccessTicket
	case "WORKLOAD_SANDBOX_PAYLOAD_TOO_LARGE":
		return ErrPayloadTooLarge
	}
	if status == http.StatusBadGateway && strings.Contains(reason, "SANDBOX") {
		return ErrGateway
	}
	return nil
}

type SandboxUnavailableError struct{ APIError }

func (e *SandboxUnavailableError) Unwrap() error        { return &e.APIError }
func (e *SandboxUnavailableError) Is(target error) bool { return target == ErrSandboxUnavailable }

type SandboxStateError struct{ APIError }

func (e *SandboxStateError) Unwrap() error        { return &e.APIError }
func (e *SandboxStateError) Is(target error) bool { return target == ErrSandboxState }

type SandboxTemplateError struct{ APIError }

func (e *SandboxTemplateError) Unwrap() error        { return &e.APIError }
func (e *SandboxTemplateError) Is(target error) bool { return target == ErrSandboxTemplate }

type TerminalLimitError struct{ APIError }

func (e *TerminalLimitError) Unwrap() error        { return &e.APIError }
func (e *TerminalLimitError) Is(target error) bool { return target == ErrTerminalLimit }

type AccessTicketError struct{ APIError }

func (e *AccessTicketError) Unwrap() error        { return &e.APIError }
func (e *AccessTicketError) Is(target error) bool { return target == ErrAccessTicket }

type PayloadTooLargeError struct{ APIError }

func (e *PayloadTooLargeError) Unwrap() error        { return &e.APIError }
func (e *PayloadTooLargeError) Is(target error) bool { return target == ErrPayloadTooLarge }

type GatewayError struct{ APIError }

func (e *GatewayError) Unwrap() error        { return &e.APIError }
func (e *GatewayError) Is(target error) bool { return target == ErrGateway }

// AuthenticationError is a 401 response.
type AuthenticationError struct{ APIError }

func (e *AuthenticationError) Error() string {
	if e == nil {
		return ""
	}
	if e.Message != "" {
		return e.APIError.Error()
	}
	return "authentication failed. Check your API key is valid. Set TARGON_API_KEY or pass an API key to New()."
}

func (e *AuthenticationError) Unwrap() error { return &e.APIError }

// AuthorizationError is a 403 response.
type AuthorizationError struct{ APIError }

func (e *AuthorizationError) Unwrap() error { return &e.APIError }

// ResourceNotFoundError is a 404 response.
type ResourceNotFoundError struct {
	APIError
	ResourceType string
	ResourceID   string
}

func (e *ResourceNotFoundError) Unwrap() error { return &e.APIError }

// RateLimitError is a 429 response.
type RateLimitError struct {
	APIError
	RetryAfter int
}

func (e *RateLimitError) Unwrap() error { return &e.APIError }

// ValidationError is a client-side input error.
type ValidationError struct {
	Message string
	Field   string
	Value   any
}

func (e *ValidationError) Error() string {
	if e == nil {
		return ""
	}
	if e.Field != "" {
		return fmt.Sprintf("%s (field=%s)", e.Message, e.Field)
	}
	return e.Message
}

// ConfigError is a missing or invalid client configuration.
type ConfigError struct {
	Message   string
	ConfigKey string
}

func (e *ConfigError) Error() string {
	if e == nil {
		return ""
	}
	if e.ConfigKey != "" {
		return fmt.Sprintf("%s (config_key=%s)", e.Message, e.ConfigKey)
	}
	return e.Message
}

// TimeoutError is returned when a wait/poll deadline is exceeded.
type TimeoutError struct {
	Message string
	Timeout float64
}

func (e *TimeoutError) Error() string {
	if e == nil {
		return ""
	}
	return e.Message
}

func (e *TimeoutError) IsRetryable() bool { return true }

// NetworkError wraps a transport failure.
type NetworkError struct {
	Message string
	Cause   error
}

func (e *NetworkError) Error() string {
	if e == nil {
		return ""
	}
	if e.Cause != nil {
		return fmt.Sprintf("%s: %v", e.Message, e.Cause)
	}
	return e.Message
}

func (e *NetworkError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

func (e *NetworkError) IsRetryable() bool { return true }

// IsNotFound reports whether err is a 404.
func IsNotFound(err error) bool {
	var api *APIError
	if errors.As(err, &api) {
		return api.IsNotFound()
	}
	var nf *ResourceNotFoundError
	return errors.As(err, &nf)
}

// IsUnauthorized reports whether err is a 401.
func IsUnauthorized(err error) bool {
	var api *APIError
	if errors.As(err, &api) {
		return api.IsUnauthorized()
	}
	var auth *AuthenticationError
	return errors.As(err, &auth)
}

// IsRateLimited reports whether err is a 429.
func IsRateLimited(err error) bool {
	var api *APIError
	if errors.As(err, &api) {
		return api.IsRateLimited()
	}
	var rl *RateLimitError
	return errors.As(err, &rl)
}

func requireNonEmpty(value, field string) (string, error) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return "", &ValidationError{
			Message: field + " must be a non-empty string",
			Field:   field,
			Value:   value,
		}
	}
	return trimmed, nil
}
