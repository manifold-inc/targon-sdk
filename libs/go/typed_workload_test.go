package targon

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
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
}

var _ commonWorkloads = (*WorkloadService)(nil)

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
