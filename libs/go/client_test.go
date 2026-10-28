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

func testClient(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	c, err := New(Config{
		APIKey:  "test-key",
		Org:     "acme",
		BaseURL: srv.URL,
		Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.Close)
	return c
}

func TestInventoryList(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tha/v3/inventory" {
			t.Errorf("path %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("auth %s", got)
		}
		if r.URL.Query().Get("type") != "rental" {
			t.Errorf("type %s", r.URL.Query().Get("type"))
		}
		_ = json.NewEncoder(w).Encode([]Inventory{{
			Name:        CPUSmall,
			DisplayName: "CPU Small",
			Type:        "rental",
			Available:   4,
		}})
	})
	items, err := c.Inventory.List(context.Background(), "rental", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || items[0].Name != CPUSmall {
		t.Fatalf("%+v", items)
	}
}

func TestInventoryRejectsDeprecatedType(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("should not call API")
	})
	_, err := c.Inventory.List(context.Background(), "serverless", nil)
	var ve *ValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("got %v", err)
	}
}

func TestAPIErrorMapping(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"missing","reason":"NOT_FOUND"}`))
	})
	_, err := c.Projects.Get(context.Background(), "proj_1")
	if !IsNotFound(err) {
		t.Fatalf("expected not found, got %v", err)
	}
}

func TestForOrgSharesTransport(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/orgs/other/projects") {
			t.Errorf("path %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(List[Project]{Items: []Project{{UID: "p1", Name: "n"}}})
	})
	scoped, err := c.ForOrg("other")
	if err != nil {
		t.Fatal(err)
	}
	if scoped.http != c.http {
		t.Fatal("expected shared http client")
	}
	list, err := scoped.Projects.List(context.Background(), Page{})
	if err != nil {
		t.Fatal(err)
	}
	if len(list.Items) != 1 {
		t.Fatalf("%+v", list)
	}
}

func TestOrgCRUD(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == "POST" && r.URL.Path == "/tha/v3/orgs":
			_ = json.NewEncoder(w).Encode(Org{UID: "o1", Slug: "acme", Name: "Acme"})
		case r.Method == "GET" && r.URL.Path == "/tha/v3/orgs/acme":
			_ = json.NewEncoder(w).Encode(Org{UID: "o1", Slug: "acme"})
		default:
			t.Errorf("%s %s", r.Method, r.URL.Path)
			w.WriteHeader(404)
		}
	})
	org, err := c.Orgs.Create(context.Background(), "Acme", "acme")
	if err != nil || org.Slug != "acme" {
		t.Fatalf("%v %+v", err, org)
	}
	got, err := c.Orgs.Get(context.Background(), "acme")
	if err != nil || got.UID != "o1" {
		t.Fatalf("%v %+v", err, got)
	}
}

func TestTokens(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/tha/v3/me/api-tokens":
			_ = json.NewEncoder(w).Encode(APIToken{UID: "tok_1", Name: "laptop"})
		case "/tha/v3/orgs/acme/tokens":
			_ = json.NewEncoder(w).Encode(ServiceToken{UID: "svc_1", Name: "ci"})
		default:
			t.Errorf("path %s", r.URL.Path)
		}
	})
	tok, err := c.APITokens.Create(context.Background(), "laptop")
	if err != nil || tok.UID != "tok_1" {
		t.Fatalf("%v %+v", err, tok)
	}
	svc, err := c.ServiceTokens.Create(context.Background(), "ci")
	if err != nil || svc.Name != "ci" {
		t.Fatalf("%v %+v", err, svc)
	}
}

func TestSandboxCreate(t *testing.T) {
	var stateCalls atomic.Int32
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == "POST" && r.URL.Path == "/tha/v3/orgs/acme/workloads":
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk_sb", Name: "sandbox"})
		case strings.HasSuffix(r.URL.Path, "/deploy"):
			_ = json.NewEncoder(w).Encode(WorkloadOperationResponse{UID: "wrk_sb"})
		case strings.HasSuffix(r.URL.Path, "/state"):
			stateCalls.Add(1)
			status := "provisioning"
			if stateCalls.Load() > 1 {
				status = "running"
			}
			_ = json.NewEncoder(w).Encode(WorkloadStateResponse{UID: "wrk_sb", WorkloadType: "SANDBOX", Status: status})
		case r.Method == "GET" && strings.HasSuffix(r.URL.Path, "/workloads/wrk_sb"):
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk_sb", Name: "sandbox", Type: "SANDBOX"})
		default:
			t.Errorf("%s %s", r.Method, r.URL.Path)
		}
	})
	sb, err := CreateSandbox(context.Background(), SandboxCreateParams{
		Name:     "sandbox",
		Template: readySandboxTemplate(c, "sbt-base"),
		Wait: WaitOptions{
			Timeout:      time.Second,
			PollInterval: 10 * time.Millisecond,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if sb.UID != "wrk_sb" || sb.Type != "SANDBOX" {
		t.Fatalf("%+v", sb)
	}
}

func TestSSHKey(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/ssh-keys") && r.Method == "POST" {
			_ = json.NewEncoder(w).Encode(SSHKey{UID: "ssh_1", Name: "laptop"})
		} else {
			t.Errorf("%s %s", r.Method, r.URL.Path)
		}
	})
	key, err := c.SSHKeys.Create(context.Background(), "laptop", "ssh-ed25519 AAAA")
	if err != nil || key.UID != "ssh_1" {
		t.Fatalf("%v %+v", err, key)
	}
}
