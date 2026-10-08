package targon

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type retryTransport struct {
	base       http.RoundTripper
	maxRetries int
}

type noRetryContextKey struct{}

func (t *retryTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	if t.maxRetries <= 0 || noRetryFromContext(req.Context()) {
		return base.RoundTrip(req)
	}

	var lastErr error
	for attempt := 0; attempt <= t.maxRetries; attempt++ {
		if err := req.Context().Err(); err != nil {
			return nil, err
		}
		cloned, err := cloneRequest(req)
		if err != nil {
			return nil, err
		}
		resp, err := base.RoundTrip(cloned)
		if err != nil {
			lastErr = err
			if attempt == t.maxRetries || !retryableMethod(req.Method) {
				return nil, err
			}
			if err := sleepBackoff(req.Context(), attempt); err != nil {
				return nil, err
			}
			continue
		}
		if !retryableStatus(resp.StatusCode) || !retryableMethod(req.Method) || attempt == t.maxRetries {
			return resp, nil
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if err := sleepBackoff(req.Context(), attempt); err != nil {
			return nil, err
		}
	}
	if lastErr != nil {
		return nil, lastErr
	}
	return base.RoundTrip(req)
}

func retryableMethod(method string) bool {
	switch method {
	case http.MethodGet, http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete:
		return true
	default:
		return false
	}
}

func retryableStatus(code int) bool {
	switch code {
	case http.StatusTooManyRequests, http.StatusInternalServerError, http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return true
	default:
		return false
	}
}

func sleepBackoff(ctx context.Context, attempt int) error {
	d := 500 * time.Millisecond * time.Duration(1<<uint(attempt))
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func cloneRequest(req *http.Request) (*http.Request, error) {
	cloned := req.Clone(req.Context())
	if req.Body == nil || req.Body == http.NoBody {
		return cloned, nil
	}
	if req.GetBody != nil {
		body, err := req.GetBody()
		if err != nil {
			return nil, err
		}
		cloned.Body = body
		return cloned, nil
	}
	buf, err := io.ReadAll(req.Body)
	req.Body.Close()
	if err != nil {
		return nil, err
	}
	req.Body = io.NopCloser(bytes.NewReader(buf))
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(buf)), nil
	}
	cloned.Body = io.NopCloser(bytes.NewReader(buf))
	cloned.GetBody = req.GetBody
	cloned.ContentLength = int64(len(buf))
	return cloned, nil
}

func newHTTPClient(cfg Config, retries bool) *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	if cfg.SkipTLSVerify {
		transport.TLSClientConfig = &tls.Config{InsecureSkipVerify: true} //nolint:gosec
	}
	var rt http.RoundTripper = transport
	if retries && cfg.MaxRetries > 0 {
		rt = &retryTransport{base: transport, maxRetries: cfg.MaxRetries}
	}
	return &http.Client{
		Timeout:   cfg.Timeout,
		Transport: rt,
	}
}

func (c *Client) do(ctx context.Context, method, path string, query url.Values, body any, out any) error {
	return c.doJSON(ctx, method, path, query, body, out)
}

// doNoRetry executes one request without transport retries. Sandbox operations
// use this for non-idempotent POSTs that could create duplicate resources or
// repeat side effects.
func (c *Client) doNoRetry(ctx context.Context, method, path string, query url.Values, body any, out any) error {
	return c.doJSON(context.WithValue(ctx, noRetryContextKey{}, true), method, path, query, body, out)
}

func noRetryFromContext(ctx context.Context) bool {
	disabled, _ := ctx.Value(noRetryContextKey{}).(bool)
	return disabled
}

func (c *Client) doJSON(ctx context.Context, method, path string, query url.Values, body any, out any) error {
	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("encode request: %w", err)
		}
		reader = bytes.NewReader(raw)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.url(path, query), reader)
	if err != nil {
		return &NetworkError{Message: "failed to build request", Cause: err}
	}
	for k, v := range c.cfg.headers() {
		req.Header.Set(k, v)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return &NetworkError{Message: "request failed", Cause: err}
	}
	defer resp.Body.Close()
	return c.handleJSON(resp, out)
}

func (c *Client) handleJSON(resp *http.Response, out any) error {
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return &NetworkError{Message: "failed to read response", Cause: err}
	}
	if err := apiErrorFromResponse(resp, raw); err != nil {
		return err
	}
	if out == nil || resp.StatusCode == http.StatusNoContent || len(bytes.TrimSpace(raw)) == 0 {
		return nil
	}
	ct := resp.Header.Get("Content-Type")
	if strings.Contains(ct, "text/plain") {
		if s, ok := out.(*string); ok {
			*s = string(raw)
			return nil
		}
	}
	if err := json.Unmarshal(raw, out); err != nil {
		if s, ok := out.(*string); ok {
			*s = string(raw)
			return nil
		}
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}

func (c *Client) stream(ctx context.Context, method, path string, query url.Values) (io.ReadCloser, error) {
	req, err := http.NewRequestWithContext(ctx, method, c.url(path, query), nil)
	if err != nil {
		return nil, &NetworkError{Message: "failed to build request", Cause: err}
	}
	for k, v := range c.cfg.headers() {
		req.Header.Set(k, v)
	}
	req.Header.Set("Accept", "text/plain")
	resp, err := c.streamHTTP.Do(req)
	if err != nil {
		return nil, &NetworkError{Message: "request failed", Cause: err}
	}
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		return nil, apiErrorFromResponse(resp, raw)
	}
	return resp.Body, nil
}

func (c *Client) getText(ctx context.Context, path string, query url.Values) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.url(path, query), nil)
	if err != nil {
		return "", &NetworkError{Message: "failed to build request", Cause: err}
	}
	for k, v := range c.cfg.headers() {
		req.Header.Set(k, v)
	}
	req.Header.Set("Accept", "text/plain")
	resp, err := c.http.Do(req)
	if err != nil {
		return "", &NetworkError{Message: "request failed", Cause: err}
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", &NetworkError{Message: "failed to read response", Cause: err}
	}
	if err := apiErrorFromResponse(resp, raw); err != nil {
		return "", err
	}
	return string(raw), nil
}

func (c *Client) url(path string, query url.Values) string {
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	u := c.cfg.BaseURL + path
	if len(query) > 0 {
		u += "?" + query.Encode()
	}
	return u
}

type errorBody struct {
	Error  string `json:"error"`
	Reason string `json:"reason"`
}

func apiErrorFromResponse(resp *http.Response, raw []byte) error {
	if resp.StatusCode < 400 {
		return nil
	}
	message := strings.TrimSpace(string(raw))
	reason := ""
	var parsed errorBody
	if json.Unmarshal(raw, &parsed) == nil {
		if parsed.Error != "" {
			message = parsed.Error
		}
		reason = parsed.Reason
	}
	if message == "" {
		message = resp.Status
	}
	return mapAPIError(resp.StatusCode, message, reason, resp.Header.Get("X-Request-Id"), workloadUIDFromResponse(resp))
}

func workloadUIDFromResponse(resp *http.Response) string {
	if resp == nil || resp.Request == nil || resp.Request.URL == nil {
		return ""
	}
	parts := strings.Split(strings.Trim(resp.Request.URL.Path, "/"), "/")
	for i := range parts {
		if parts[i] == "workloads" && i+1 < len(parts) {
			switch parts[i+1] {
			case "", "verify", "vm-images", "bm-images":
				return ""
			default:
				return parts[i+1]
			}
		}
	}
	return ""
}
