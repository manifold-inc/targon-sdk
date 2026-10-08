package targon

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coder/websocket"
)

func readySandboxTemplate(c *Client, uid string) *SandboxTemplate {
	return hydrateSandboxTemplate(c.Sandboxes.Templates, &SandboxTemplate{
		UID: uid, Status: SandboxTemplateStatusReady,
	})
}

func TestSandboxCreateWireContractAndNoPOSTRetry(t *testing.T) {
	var calls atomic.Int32
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.Method != http.MethodPost || r.URL.Path != "/tha/v3/orgs/acme/workloads" {
			t.Fatalf("%s %s", r.Method, r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["type"] != "SANDBOX" || body["image"] != "sbt-base" {
			t.Fatalf("unexpected body: %#v", body)
		}
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"full","reason":"WORKLOAD_SANDBOX_NO_CAPACITY"}`))
	})

	_, err := c.Sandboxes.Create(context.Background(), SandboxCreateParams{
		Name:     "dev-box",
		Template: readySandboxTemplate(c, "sbt-base"),
		Wait:     WaitOptions{NoWait: true},
	})
	if !errors.Is(err, ErrSandboxUnavailable) {
		t.Fatalf("expected unavailable error, got %T %v", err, err)
	}
	if calls.Load() != 1 {
		t.Fatalf("non-idempotent POST retried %d times", calls.Load())
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T", err)
	}
}

func TestSandboxValidation(t *testing.T) {
	c := testClient(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("validation should fail before HTTP")
	})
	_, err := c.Sandboxes.Create(context.Background(), SandboxCreateParams{
		Name: "Bad Name", Template: readySandboxTemplate(c, "sbt-base"),
	})
	var validationErr *ValidationError
	if !errors.As(err, &validationErr) {
		t.Fatalf("expected validation error, got %v", err)
	}
	_, err = c.Sandboxes.Exec(context.Background(), "wrk-1", strings.Repeat("x", MaxSandboxExecCommandBytes+1), 60)
	if !errors.As(err, &validationErr) {
		t.Fatalf("expected command validation error, got %v", err)
	}
	_, err = c.Sandboxes.Exec(context.Background(), "wrk-1", " \t\n", 60)
	if !errors.As(err, &validationErr) {
		t.Fatalf("expected whitespace command validation error, got %v", err)
	}
}

func TestSandboxTemplatesExecFilesAndDesktop(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/sandbox-templates"):
			if r.URL.Query().Get("kind") != "USER" {
				t.Errorf("kind=%q", r.URL.Query().Get("kind"))
			}
			_ = json.NewEncoder(w).Encode(List[SandboxTemplate]{Items: []SandboxTemplate{{
				UID: "sbt-user", Kind: SandboxTemplateKindUser, Status: SandboxTemplateStatusReady,
			}}})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/exec"):
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			if body["cmd"] != "echo ok" {
				t.Errorf("body=%#v", body)
			}
			_ = json.NewEncoder(w).Encode(SandboxExecResult{Stdout: "ok\n", Code: 0})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/files"):
			if r.URL.Query().Get("path") != "/tmp/a" {
				t.Errorf("path=%q", r.URL.Query().Get("path"))
			}
			_ = json.NewEncoder(w).Encode(SandboxFileInfo{Path: "/tmp/a", ContentB64: base64.StdEncoding.EncodeToString([]byte{0, 1, 2})})
		case r.Method == http.MethodPut && strings.HasSuffix(r.URL.Path, "/files"):
			var body SandboxFileInfo
			_ = json.NewDecoder(r.Body).Decode(&body)
			if body.Path != "/tmp/b" || body.ContentB64 != base64.StdEncoding.EncodeToString([]byte("data")) {
				t.Errorf("body=%#v", body)
			}
			w.WriteHeader(http.StatusNoContent)
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/desktop"):
			_ = json.NewEncoder(w).Encode(SandboxDesktopInfo{Available: true, Port: 6080, WSURL: "wss://example.test/desktop/ws"})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	})

	templates, err := c.Sandboxes.Templates.List(context.Background(), ListSandboxTemplatesParams{
		Page: Page{Limit: 10}, Kind: SandboxTemplateKindUser,
	})
	if err != nil || len(templates.Items) != 1 {
		t.Fatalf("templates=%+v err=%v", templates, err)
	}
	if templates.Items[0].service != c.Sandboxes.Templates {
		t.Fatal("listed template was not bound to its service")
	}
	execResult, err := c.Sandboxes.Exec(context.Background(), "wrk-1", "echo ok", 60)
	if err != nil || execResult.Stdout != "ok\n" {
		t.Fatalf("exec=%+v err=%v", execResult, err)
	}
	data, err := c.Sandboxes.Files.Read(context.Background(), "wrk-1", "/tmp/a")
	if err != nil || len(data) != 3 || data[2] != 2 {
		t.Fatalf("read=%v err=%v", data, err)
	}
	if err := c.Sandboxes.Files.Write(context.Background(), "wrk-1", "/tmp/b", []byte("data")); err != nil {
		t.Fatal(err)
	}
	desktop, err := c.Sandboxes.GetDesktop(context.Background(), "wrk-1")
	if err != nil || !desktop.Available || desktop.Port != 6080 {
		t.Fatalf("desktop=%+v err=%v", desktop, err)
	}
}

func TestSandboxCreateRequiresBoundReadyTemplate(t *testing.T) {
	c := testClient(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("template validation should fail before HTTP")
	})
	other := testClient(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("template validation should fail before HTTP")
	})

	tests := []struct {
		name     string
		template *SandboxTemplate
	}{
		{name: "nil"},
		{name: "unbound", template: &SandboxTemplate{UID: "sbt-user", Status: SandboxTemplateStatusReady}},
		{name: "empty uid", template: readySandboxTemplate(c, "")},
		{name: "not ready", template: hydrateSandboxTemplate(c.Sandboxes.Templates, &SandboxTemplate{
			UID: "sbt-user", Status: SandboxTemplateStatusPending,
		})},
		{name: "different service", template: readySandboxTemplate(other, "sbt-user")},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := c.Sandboxes.Create(context.Background(), SandboxCreateParams{
				Name: "valid-name", Template: test.template,
			})
			var validationErr *ValidationError
			if !errors.As(err, &validationErr) {
				t.Fatalf("expected validation error, got %T %v", err, err)
			}
		})
	}
}

func TestSandboxTemplateResourceMethodsAndHydration(t *testing.T) {
	display := "Updated"
	var getCalls atomic.Int32
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/sandbox-templates/sbt-user"):
			getCalls.Add(1)
			_ = json.NewEncoder(w).Encode(SandboxTemplate{
				UID: "sbt-user", Name: "user", DisplayName: &display,
				Kind: SandboxTemplateKindUser, Status: SandboxTemplateStatusReady,
			})
		case r.Method == http.MethodPatch && strings.HasSuffix(r.URL.Path, "/sandbox-templates/sbt-user"):
			_ = json.NewEncoder(w).Encode(SandboxTemplate{
				UID: "sbt-user", Name: "user", DisplayName: &display,
				Kind: SandboxTemplateKindUser, Status: SandboxTemplateStatusReady,
			})
		case r.Method == http.MethodDelete && strings.HasSuffix(r.URL.Path, "/sandbox-templates/sbt-user"):
			w.WriteHeader(http.StatusNoContent)
		case r.Method == http.MethodPost && r.URL.Path == "/tha/v3/orgs/acme/workloads":
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			if body["image"] != "sbt-user" {
				t.Errorf("image=%v", body["image"])
			}
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-template"})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/workloads/wrk-template/deploy"):
			_ = json.NewEncoder(w).Encode(WorkloadOperationResponse{UID: "wrk-template"})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/workloads/wrk-template"):
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-template", Name: "from-template", Type: "SANDBOX"})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	})

	template, err := c.Sandboxes.Templates.Get(context.Background(), "sbt-user")
	if err != nil || template.service != c.Sandboxes.Templates {
		t.Fatalf("template=%+v err=%v", template, err)
	}
	if refreshed, err := template.Refresh(context.Background()); err != nil || refreshed != template {
		t.Fatalf("refreshed=%+v err=%v", refreshed, err)
	}
	if updated, err := template.Update(context.Background(), UpdateSandboxTemplateParams{DisplayName: &display}); err != nil || updated != template {
		t.Fatalf("updated=%+v err=%v", updated, err)
	}
	sandbox, err := template.CreateSandbox(context.Background(), SandboxCreateParams{
		Name: "from-template", Wait: WaitOptions{NoWait: true},
	})
	if err != nil || sandbox.UID != "wrk-template" {
		t.Fatalf("sandbox=%+v err=%v", sandbox, err)
	}
	if err := template.Delete(context.Background()); err != nil {
		t.Fatal(err)
	}
	if getCalls.Load() != 2 {
		t.Fatalf("template GET calls=%d, want 2", getCalls.Load())
	}
}

func TestSandboxPublishReturnsBoundTemplate(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/workloads/wrk-1/publish"):
			_ = json.NewEncoder(w).Encode(SandboxTemplate{
				UID: "sbt-published", Name: "published",
				Kind: SandboxTemplateKindUser, Status: SandboxTemplateStatusReady,
			})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/sandbox-templates/sbt-published"):
			_ = json.NewEncoder(w).Encode(SandboxTemplate{
				UID: "sbt-published", Name: "published",
				Kind: SandboxTemplateKindUser, Status: SandboxTemplateStatusReady,
			})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	})

	template, err := c.Sandboxes.Publish(context.Background(), "wrk-1", PublishSandboxParams{
		Name: "published", Wait: WaitOptions{NoWait: true},
	})
	if err != nil || template.service != c.Sandboxes.Templates {
		t.Fatalf("template=%+v err=%v", template, err)
	}
	if _, err := template.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestTerminalWebSocketMintsFreshTicketAndUsesBinaryFrames(t *testing.T) {
	var tickets atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/access-tickets"):
			n := tickets.Add(1)
			_ = json.NewEncoder(w).Encode(SandboxAccessTicket{
				Ticket: "sat-test-" + string(rune('0'+n)), ExpiresAt: time.Now().Add(time.Minute),
			})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/terminals/term-1/ws"):
			if r.URL.Query().Get("ticket") == "" {
				t.Error("missing access ticket")
			}
			conn, err := websocket.Accept(w, r, nil)
			if err != nil {
				t.Error(err)
				return
			}
			defer conn.Close(websocket.StatusNormalClosure, "")
			messageType, data, err := conn.Read(r.Context())
			if err != nil {
				t.Error(err)
				return
			}
			if messageType != websocket.MessageBinary {
				t.Errorf("message type=%v", messageType)
			}
			if err := conn.Write(r.Context(), websocket.MessageBinary, append([]byte("echo:"), data...)); err != nil {
				t.Error(err)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	c, err := New(Config{APIKey: "test", Org: "acme", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	conn, err := c.Sandboxes.Terminals.Connect(context.Background(), "wrk-1", "term-1", ConnectTerminalOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if err := conn.Write(context.Background(), []byte("hello")); err != nil {
		t.Fatal(err)
	}
	data, err := conn.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "echo:hello" {
		t.Fatalf("data=%q", data)
	}
	if tickets.Load() != 1 {
		t.Fatalf("ticket calls=%d", tickets.Load())
	}
}

func TestRouteScopedErrorCarriesWorkloadUID(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusConflict)
		_, _ = w.Write([]byte(`{"error":"not deployed","reason":"WORKLOAD_SANDBOX_NOT_DEPLOYED"}`))
	})
	_, err := c.Sandboxes.Exec(context.Background(), "wrk-123", "true", 60)
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.WorkloadUID != "wrk-123" {
		t.Fatalf("error=%T %+v", err, apiErr)
	}
	if !errors.Is(err, ErrSandboxState) {
		t.Fatalf("expected state category: %v", err)
	}
}

func TestSandboxResourceObjectMethods(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/workloads/wrk-object"):
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-object", Name: "object-box", Type: "SANDBOX"})
		case r.Method == http.MethodGet && r.URL.Path == "/tha/v3/orgs/acme/workloads":
			_ = json.NewEncoder(w).Encode(List[WorkloadListItem]{Items: []WorkloadListItem{{
				UID: "wrk-object", Name: "object-box", Type: "SANDBOX",
			}}})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/exec"):
			_ = json.NewEncoder(w).Encode(SandboxExecResult{Stdout: "from-object\n"})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/files"):
			_ = json.NewEncoder(w).Encode(SandboxFileInfo{
				Path: r.URL.Query().Get("path"), ContentB64: base64.StdEncoding.EncodeToString([]byte("object-data")),
			})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/terminals"):
			_ = json.NewEncoder(w).Encode([]TerminalSession{{ID: "term-object", PID: 42}})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	})

	sandbox, err := c.Sandboxes.Get(context.Background(), "wrk-object")
	if err != nil || sandbox.UID != "wrk-object" || sandbox.Workload.Name != "object-box" {
		t.Fatalf("sandbox=%+v err=%v", sandbox, err)
	}
	result, err := sandbox.Exec(context.Background(), "echo object", 60)
	if err != nil || result.Stdout != "from-object\n" {
		t.Fatalf("exec=%+v err=%v", result, err)
	}
	data, err := sandbox.ReadFile(context.Background(), "/tmp/object")
	if err != nil || string(data) != "object-data" {
		t.Fatalf("data=%q err=%v", data, err)
	}
	terminals, err := sandbox.ListTerminals(context.Background())
	if err != nil || len(terminals) != 1 || terminals[0].ID != "term-object" {
		t.Fatalf("terminals=%+v err=%v", terminals, err)
	}

	list, err := c.Sandboxes.List(context.Background(), ListSandboxesParams{})
	if err != nil || len(list.Items) != 1 {
		t.Fatalf("list=%+v err=%v", list, err)
	}
	if list.Items[0].UID != "wrk-object" || list.Items[0].Type != "SANDBOX" {
		t.Fatalf("summary=%+v", list.Items[0])
	}
}

func TestPOSTRetryScope(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/deploy") {
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":"retry","reason":"TEMPORARY"}`))
			return
		}
		_ = json.NewEncoder(w).Encode(WorkloadOperationResponse{UID: "wrk-retry"})
	}))
	defer srv.Close()

	c, err := New(Config{
		APIKey: "test", Org: "acme", BaseURL: srv.URL, MaxRetries: 1, Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if _, err := c.Workloads.Deploy(context.Background(), "wrk-retry"); err != nil {
		t.Fatalf("existing non-sandbox POST did not retain retries: %v", err)
	}
	if calls.Load() != 2 {
		t.Fatalf("POST calls=%d, want 2", calls.Load())
	}
}

func TestSandboxListAndTimeoutValidation(t *testing.T) {
	c := testClient(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("validation should fail before HTTP")
	})
	_, err := c.Sandboxes.List(context.Background(), ListSandboxesParams{Page: Page{Limit: 1001}})
	var validationErr *ValidationError
	if !errors.As(err, &validationErr) {
		t.Fatalf("expected list limit validation, got %v", err)
	}
	ttl, idle := 60, 60
	_, err = c.Sandboxes.Create(context.Background(), SandboxCreateParams{
		Name: "timeouts", Template: readySandboxTemplate(c, "sbt-base"),
		SandboxConfig: &SandboxConfigInput{TTLSec: &ttl, IdleTimeoutSec: &idle},
	})
	if !errors.As(err, &validationErr) {
		t.Fatalf("expected create timeout validation, got %v", err)
	}
	name := "updated"
	_, err = c.Sandboxes.Update(context.Background(), "wrk-1", SandboxUpdateParams{
		Name: &name, SandboxConfig: &SandboxConfigInput{TTLSec: &ttl, IdleTimeoutSec: &idle},
	})
	if !errors.As(err, &validationErr) {
		t.Fatalf("expected update timeout validation, got %v", err)
	}
}

func TestSandboxRejectsNonSandboxResponses(t *testing.T) {
	name := "renamed"
	tests := []struct {
		name string
		call func(*Client) error
	}{
		{
			name: "get",
			call: func(c *Client) error {
				_, err := c.Sandboxes.Get(context.Background(), "wrk-wrong")
				return err
			},
		},
		{
			name: "update",
			call: func(c *Client) error {
				_, err := c.Sandboxes.Update(context.Background(), "wrk-wrong", SandboxUpdateParams{Name: &name})
				return err
			},
		},
		{
			name: "state",
			call: func(c *Client) error {
				_, err := c.Sandboxes.GetState(context.Background(), "wrk-wrong")
				return err
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				if strings.HasSuffix(r.URL.Path, "/state") {
					_ = json.NewEncoder(w).Encode(WorkloadStateResponse{UID: "wrk-wrong", WorkloadType: "VM", Status: "running"})
					return
				}
				_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-wrong", Type: "VM", Name: "wrong"})
			})
			err := test.call(c)
			if !errors.Is(err, ErrSandboxState) {
				t.Fatalf("expected sandbox type error, got %T %v", err, err)
			}
			var apiErr *APIError
			if !errors.As(err, &apiErr) || apiErr.Reason != "WORKLOAD_SANDBOX_TYPE_MISMATCH" {
				t.Fatalf("api error=%+v", apiErr)
			}
		})
	}
}

func TestSandboxCreateNoWaitFetchesFullResource(t *testing.T) {
	var getCalls atomic.Int32
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/tha/v3/orgs/acme/workloads":
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-full"})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/deploy"):
			_ = json.NewEncoder(w).Encode(WorkloadOperationResponse{UID: "wrk-full"})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/workloads/wrk-full"):
			getCalls.Add(1)
			_ = json.NewEncoder(w).Encode(Workload{
				UID: "wrk-full", Type: "SANDBOX", Name: "full",
				SandboxConfig: &SandboxConfig{TemplateUID: "sbt-base"},
			})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	})
	sandbox, err := c.Sandboxes.Create(context.Background(), SandboxCreateParams{
		Name: "full", Template: readySandboxTemplate(c, "sbt-base"), Wait: WaitOptions{NoWait: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if getCalls.Load() != 1 || sandbox.SandboxConfig == nil || sandbox.SandboxConfig.TemplateUID != "sbt-base" {
		t.Fatalf("sandbox=%+v get calls=%d", sandbox, getCalls.Load())
	}
}

func TestSandboxOperationNoWaitFetchesFullResource(t *testing.T) {
	var gets atomic.Int32
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			uid := "wrk-parent"
			if strings.HasSuffix(r.URL.Path, "/fork") {
				uid = "wrk-child"
			}
			_ = json.NewEncoder(w).Encode(WorkloadOperationResponse{UID: uid})
		case http.MethodGet:
			gets.Add(1)
			uid := "wrk-parent"
			if strings.HasSuffix(r.URL.Path, "/wrk-child") {
				uid = "wrk-child"
			}
			_ = json.NewEncoder(w).Encode(Workload{
				UID: uid, Type: "SANDBOX", Name: "full",
				SandboxConfig: &SandboxConfig{TemplateUID: "sbt-base"},
			})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	})

	forked, err := c.Sandboxes.Fork(context.Background(), "wrk-parent", ForkSandboxParams{Wait: WaitOptions{NoWait: true}})
	if err != nil || forked.UID != "wrk-child" || forked.SandboxConfig == nil {
		t.Fatalf("forked=%+v err=%v", forked, err)
	}
	frozen, err := c.Sandboxes.Freeze(context.Background(), "wrk-parent", WaitOptions{NoWait: true})
	if err != nil || frozen.UID != "wrk-parent" || frozen.SandboxConfig == nil {
		t.Fatalf("frozen=%+v err=%v", frozen, err)
	}
	thawed, err := c.Sandboxes.Thaw(context.Background(), "wrk-parent", WaitOptions{NoWait: true})
	if err != nil || thawed.UID != "wrk-parent" || thawed.SandboxConfig == nil {
		t.Fatalf("thawed=%+v err=%v", thawed, err)
	}
	if gets.Load() != 3 {
		t.Fatalf("full workload GET calls=%d", gets.Load())
	}
}

func TestAllSandboxMutationsDisableRetries(t *testing.T) {
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

	name, display := "renamed", "Display"
	mutations := []struct {
		name string
		call func() error
	}{
		{"update", func() error {
			_, err := c.Sandboxes.Update(context.Background(), "wrk-1", SandboxUpdateParams{Name: &name})
			return err
		}},
		{"delete", func() error { return c.Sandboxes.Delete(context.Background(), "wrk-1") }},
		{"attach ssh", func() error { return c.Sandboxes.AttachSSHKey(context.Background(), "wrk-1", "ssh-1") }},
		{"detach ssh", func() error { return c.Sandboxes.DetachSSHKey(context.Background(), "wrk-1", "ssh-1") }},
		{"freeze", func() error {
			_, err := c.Sandboxes.Freeze(context.Background(), "wrk-1", WaitOptions{NoWait: true})
			return err
		}},
		{"thaw", func() error {
			_, err := c.Sandboxes.Thaw(context.Background(), "wrk-1", WaitOptions{NoWait: true})
			return err
		}},
		{"fork", func() error {
			_, err := c.Sandboxes.Fork(context.Background(), "wrk-1", ForkSandboxParams{Wait: WaitOptions{NoWait: true}})
			return err
		}},
		{"publish", func() error {
			_, err := c.Sandboxes.Publish(context.Background(), "wrk-1", PublishSandboxParams{Name: "template", Wait: WaitOptions{NoWait: true}})
			return err
		}},
		{"exec", func() error { _, err := c.Sandboxes.Exec(context.Background(), "wrk-1", "true", 60); return err }},
		{"ticket", func() error { _, err := c.Sandboxes.MintAccessTicket(context.Background(), "wrk-1", 60); return err }},
		{"terminal create", func() error {
			_, err := c.Sandboxes.Terminals.Create(context.Background(), "wrk-1", 80, 24)
			return err
		}},
		{"terminal delete", func() error { return c.Sandboxes.Terminals.Delete(context.Background(), "wrk-1", "term-1") }},
		{"file write", func() error { return c.Sandboxes.Files.Write(context.Background(), "wrk-1", "/tmp/a", []byte("x")) }},
		{"template update", func() error {
			_, err := c.Sandboxes.Templates.Update(context.Background(), "sbt-user", UpdateSandboxTemplateParams{DisplayName: &display})
			return err
		}},
		{"template delete", func() error { return c.Sandboxes.Templates.Delete(context.Background(), "sbt-user") }},
	}
	for _, mutation := range mutations {
		t.Run(mutation.name, func(t *testing.T) {
			before := calls.Load()
			if err := mutation.call(); err == nil {
				t.Fatal("expected mutation error")
			}
			if delta := calls.Load() - before; delta != 1 {
				t.Fatalf("mutation made %d requests, want 1", delta)
			}
		})
	}
}

func TestSandboxDeployDisablesRetries(t *testing.T) {
	var createCalls, deployCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/tha/v3/orgs/acme/workloads":
			createCalls.Add(1)
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-deploy"})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/deploy"):
			deployCalls.Add(1)
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":"temporary","reason":"WORKLOAD_SANDBOX_UNAVAILABLE"}`))
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	}))
	defer srv.Close()
	c, err := New(Config{
		APIKey: "test", Org: "acme", BaseURL: srv.URL, MaxRetries: 2, Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	_, err = c.Sandboxes.Create(context.Background(), SandboxCreateParams{
		Name: "deploy", Template: readySandboxTemplate(c, "sbt-base"), Wait: WaitOptions{NoWait: true},
	})
	if err == nil {
		t.Fatal("expected deploy error")
	}
	if createCalls.Load() != 1 || deployCalls.Load() != 1 {
		t.Fatalf("create calls=%d deploy calls=%d", createCalls.Load(), deployCalls.Load())
	}
}

func TestSandboxExecPreservesOriginalCommandBytes(t *testing.T) {
	const command = "  printf 'hello' \n"
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Cmd string `json:"cmd"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body.Cmd != command {
			t.Fatalf("cmd=%q, want %q", body.Cmd, command)
		}
		_ = json.NewEncoder(w).Encode(SandboxExecResult{Code: 0})
	})
	if _, err := c.Sandboxes.Exec(context.Background(), "wrk-1", command, 60); err != nil {
		t.Fatal(err)
	}
}
