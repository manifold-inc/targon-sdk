package targon

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type commonWorkloads interface {
	Get(context.Context, string) (*Workload, error)
	List(context.Context, ListWorkloadsParams) (List[WorkloadListItem], error)
	Delete(context.Context, string) error
	GetState(context.Context, string) (*WorkloadStateResponse, error)
	GetEvents(context.Context, string, Page) (List[WorkloadEvent], error)
	WaitUntilReady(context.Context, string, WaitOptions) (*WorkloadStateResponse, error)
	Verify(context.Context, string, string) (bool, error)
	AttachSSHKey(context.Context, string, string) (*SSHKeyAttachResponse, error)
	DetachSSHKey(context.Context, string, string) error
}

var _ commonWorkloads = (*WorkloadService)(nil)

func TestWorkloadVerifyPostsUIDAndDigest(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/tha/v3/orgs/acme/workloads/verify" {
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if want := map[string]any{"uid": "wrk-1", "digest": "abc123"}; !reflect.DeepEqual(body, want) {
			t.Fatalf("body=%#v", body)
		}
		_, _ = w.Write([]byte(`{"verified":true}`))
	})
	ok, err := c.Workloads.Verify(context.Background(), " wrk-1 ", "abc123")
	if err != nil || !ok {
		t.Fatalf("verified=%v err=%v", ok, err)
	}
	if _, err := c.Workloads.Verify(context.Background(), "wrk-1", " "); err == nil {
		t.Fatal("expected digest validation error")
	}
}

func TestWorkloadSSHKeyAttachDetachRoutes(t *testing.T) {
	var methods []string
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tha/v3/orgs/acme/workloads/wrk-1/ssh-keys/ssh-9" {
			t.Errorf("path=%s", r.URL.Path)
		}
		methods = append(methods, r.Method)
		if r.Method == http.MethodPut {
			_, _ = w.Write([]byte(`{"workload_uid":"wrk-1","ssh_key_uid":"ssh-9"}`))
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})
	attached, err := c.Workloads.AttachSSHKey(context.Background(), "wrk-1", "ssh-9")
	if err != nil || attached.WorkloadUID != "wrk-1" || attached.SSHKeyUID != "ssh-9" {
		t.Fatalf("attached=%+v err=%v", attached, err)
	}
	if err := c.Workloads.DetachSSHKey(context.Background(), "wrk-1", "ssh-9"); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(methods, []string{http.MethodPut, http.MethodDelete}) {
		t.Fatalf("methods=%v", methods)
	}
}

func TestWorkloadWaitUntilReadyForAnyType(t *testing.T) {
	for _, workloadType := range []string{"VM", "BM", "SANDBOX"} {
		var calls atomic.Int32
		c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
			status := "provisioning"
			if calls.Add(1) > 1 {
				status = "running"
			}
			_ = json.NewEncoder(w).Encode(WorkloadStateResponse{UID: "wrk-1", WorkloadType: workloadType, Status: status})
		})
		state, err := c.Workloads.WaitUntilReady(context.Background(), "wrk-1", WaitOptions{
			Timeout: time.Second, PollInterval: time.Millisecond,
		})
		if err != nil || state.Status != "running" || calls.Load() != 2 {
			t.Fatalf("type=%s state=%+v err=%v calls=%d", workloadType, state, err, calls.Load())
		}
	}
}

func TestWorkloadWaitUntilReadyTerminalAndTimeout(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(WorkloadStateResponse{UID: "wrk-1", WorkloadType: "VM", Status: "error", Message: "boom"})
	})
	_, err := c.Workloads.WaitUntilReady(context.Background(), "wrk-1", WaitOptions{Timeout: time.Second, PollInterval: time.Millisecond})
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Reason != "WORKLOAD_INVALID_STATE" || !strings.Contains(apiErr.Message, "boom") {
		t.Fatalf("err=%v", err)
	}

	c = testClient(t, func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(WorkloadStateResponse{UID: "wrk-1", WorkloadType: "VM", Status: "provisioning"})
	})
	_, err = c.Workloads.WaitUntilReady(context.Background(), "wrk-1", WaitOptions{Timeout: 20 * time.Millisecond, PollInterval: time.Millisecond})
	var timeoutErr *TimeoutError
	if !errors.As(err, &timeoutErr) {
		t.Fatalf("err=%T %v", err, err)
	}
}

func TestWaitContinuesThroughStoppedAndFrozen(t *testing.T) {
	for _, status := range []string{"stopped", "frozen", "Frozen"} {
		var calls atomic.Int32
		c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
			if strings.HasSuffix(r.URL.Path, "/state") {
				next := status
				if calls.Add(1) > 1 {
					next = "running"
				}
				_ = json.NewEncoder(w).Encode(WorkloadStateResponse{UID: "wrk-1", WorkloadType: "VM", Status: next})
				return
			}
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-1", Type: "VM"})
		})
		options := WaitOptions{Timeout: time.Second, PollInterval: time.Millisecond}
		state, err := c.Workloads.WaitUntilReady(context.Background(), "wrk-1", options)
		if err != nil || state.Status != "running" || calls.Load() != 2 {
			t.Fatalf("WaitUntilReady status=%s state=%+v err=%v calls=%d", status, state, err, calls.Load())
		}
		calls.Store(0)
		workload, err := waitTyped(context.Background(), c, "wrk-1", "VM", "running", options)
		if err != nil || workload.UID != "wrk-1" || calls.Load() != 2 {
			t.Fatalf("waitTyped status=%s workload=%+v err=%v calls=%d", status, workload, err, calls.Load())
		}
	}
}

func TestWaitUntilReadyKeepsErrorSuspendedDeletedTerminal(t *testing.T) {
	for _, status := range []string{"error", "suspended", "deleted", "Deleted"} {
		var calls atomic.Int32
		c := testClient(t, func(w http.ResponseWriter, _ *http.Request) {
			calls.Add(1)
			_ = json.NewEncoder(w).Encode(WorkloadStateResponse{UID: "wrk-1", WorkloadType: "VM", Status: status})
		})
		options := WaitOptions{Timeout: time.Second, PollInterval: time.Millisecond}
		_, err := c.Workloads.WaitUntilReady(context.Background(), "wrk-1", options)
		var apiErr *APIError
		if !errors.As(err, &apiErr) || apiErr.Reason != "WORKLOAD_INVALID_STATE" || calls.Load() != 1 {
			t.Fatalf("WaitUntilReady status=%s err=%v calls=%d", status, err, calls.Load())
		}
	}
}

func TestWaitTypedChecksTargetBeforeTerminalStates(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/state") {
			_ = json.NewEncoder(w).Encode(WorkloadStateResponse{UID: "wrk-1", WorkloadType: "VM", Status: "frozen"})
			return
		}
		_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-1", Type: "VM"})
	})
	got, err := waitTyped(context.Background(), c, "wrk-1", "VM", "frozen", WaitOptions{
		Timeout: time.Second, PollInterval: time.Millisecond,
	})
	if err != nil || got.UID != "wrk-1" {
		t.Fatalf("workload=%+v err=%v", got, err)
	}
}

func TestWorkloadListDecodesBackendOperationShape(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tha/v3/orgs/acme/workloads" {
			t.Errorf("path=%s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"items":[{"uid":"wrk-bm","type":"BM","name":"metal","image":"ubuntu-24.04",
			"reservation_uid":"res-1","created_by":{"token_type":"USER","username":"dev"},
			"state":{"status":"running","message":"","ready_replicas":1,"total_replicas":1},
			"created_at":"2026-10-28T12:00:00Z","updated_at":"2026-10-28T12:00:00Z"}],"next_cursor":null}`))
	})
	page, err := c.Workloads.List(context.Background(), ListWorkloadsParams{})
	if err != nil {
		t.Fatal(err)
	}
	item := page.Items[0]
	if item.ReservationUID == nil || *item.ReservationUID != "res-1" || item.CreatedBy == nil || item.CreatedBy.Username != "dev" {
		t.Fatalf("item=%+v", item)
	}
}

func TestCurrentWorkloadTypes(t *testing.T) {
	for _, workloadType := range []string{"VM", "BM", "SANDBOX"} {
		if _, ok := validWorkloadTypes[workloadType]; !ok {
			t.Errorf("%s should be valid", workloadType)
		}
	}
	for _, workloadType := range []string{
		"RENTAL", "SERVERLESS", "INFERENCE", "FUNCTION",
	} {
		if _, ok := deprecatedWorkloadTypes[workloadType]; !ok {
			t.Errorf("%s should be deprecated", workloadType)
		}
	}
}

func TestTypedRegisterPostsOnceAndGetsFullResource(t *testing.T) {
	var postCalls, getCalls atomic.Int32
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/tha/v3/orgs/acme/workloads":
			postCalls.Add(1)
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-registered"})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/workloads/wrk-registered"):
			getCalls.Add(1)
			_ = json.NewEncoder(w).Encode(Workload{
				UID: "wrk-registered", Type: "SANDBOX", Name: "full",
				SandboxConfig: &SandboxConfig{TemplateUID: "sbt-base"},
			})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	})

	got, err := registerTyped(context.Background(), c, map[string]any{
		"type": "SANDBOX", "name": "full",
	}, "SANDBOX")
	if err != nil {
		t.Fatal(err)
	}
	if got.UID != "wrk-registered" || got.SandboxConfig == nil {
		t.Fatalf("workload=%+v", got)
	}
	if postCalls.Load() != 1 || getCalls.Load() != 1 {
		t.Fatalf("post calls=%d get calls=%d", postCalls.Load(), getCalls.Load())
	}
}

func TestTypedDeployNoWaitPostsAndGetsFullResource(t *testing.T) {
	var deployCalls, getCalls atomic.Int32
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/workloads/wrk-deploy/deploy"):
			deployCalls.Add(1)
			_ = json.NewEncoder(w).Encode(WorkloadOperationResponse{UID: "wrk-deploy"})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/workloads/wrk-deploy"):
			getCalls.Add(1)
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-deploy", Type: "SANDBOX", Name: "full"})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	})

	got, err := deployTyped(context.Background(), c, "wrk-deploy", "SANDBOX", WaitOptions{NoWait: true})
	if err != nil {
		t.Fatal(err)
	}
	if got.UID != "wrk-deploy" || deployCalls.Load() != 1 || getCalls.Load() != 1 {
		t.Fatalf("workload=%+v deploy calls=%d get calls=%d", got, deployCalls.Load(), getCalls.Load())
	}
}

func TestTypedDeployWaitsForRunning(t *testing.T) {
	var stateCalls atomic.Int32
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/workloads/wrk-wait/deploy"):
			_ = json.NewEncoder(w).Encode(WorkloadOperationResponse{UID: "wrk-wait"})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/workloads/wrk-wait/state"):
			status := "provisioning"
			if stateCalls.Add(1) > 1 {
				status = "running"
			}
			_ = json.NewEncoder(w).Encode(WorkloadStateResponse{
				UID: "wrk-wait", WorkloadType: "SANDBOX", Status: status,
			})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/workloads/wrk-wait"):
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-wait", Type: "SANDBOX", Name: "running"})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	})

	got, err := deployTyped(context.Background(), c, "wrk-wait", "SANDBOX", WaitOptions{
		Timeout: time.Second, PollInterval: time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.UID != "wrk-wait" || stateCalls.Load() != 2 {
		t.Fatalf("workload=%+v state calls=%d", got, stateCalls.Load())
	}
}

func TestTypedTypeMismatchUsesSandboxError(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-wrong"})
		case http.MethodGet:
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-wrong", Type: "VM"})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	})

	_, err := registerTyped(context.Background(), c, map[string]string{"type": "SANDBOX"}, "SANDBOX")
	if !errors.Is(err, ErrSandboxState) {
		t.Fatalf("expected sandbox type error, got %T %v", err, err)
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Reason != "WORKLOAD_SANDBOX_TYPE_MISMATCH" {
		t.Fatalf("api error=%+v", apiErr)
	}
}

func TestTypedMutationsDisableRetries(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"temporary","reason":"WORKLOAD_SANDBOX_UNAVAILABLE"}`))
	}))
	defer srv.Close()
	c, err := New(Config{
		APIKey: "test", Org: "acme", BaseURL: srv.URL, MaxRetries: 2, Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	if _, err := registerTyped(context.Background(), c, map[string]string{"type": "SANDBOX"}, "SANDBOX"); err == nil {
		t.Fatal("expected register error")
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("register requests=%d, want 1", got)
	}
	if _, err := deployTyped(context.Background(), c, "wrk-1", "SANDBOX", WaitOptions{NoWait: true}); err == nil {
		t.Fatal("expected deploy error")
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("total mutation requests=%d, want 2", got)
	}
}
