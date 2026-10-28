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

// inventoryResponse is a /tha/v3/inventory row in the shape the backend
// returns for current vm and bm resources.
const inventoryResponse = `[
  {
    "name": "h100-small",
    "display_name": "H100 Small",
    "description": "1x H100",
    "type": "vm",
    "gpu": true,
    "spec": {
      "gpu_vendor": "nvidia", "gpu_model": "h100", "gpu_type": "H100", "gpu_count": 1,
      "cpu_millicores": 16000, "memory_mib": 131072, "disk_size_mib": 512000,
      "network_mode": "nat", "cc_enabled": true,
      "vcpu": 16, "memory": 128, "storage": 500
    },
    "cost_per_hour": 2.5,
    "available": 3
  },
  {
    "name": "bm-h100-8x",
    "display_name": "H100 8x Bare Metal",
    "description": "8x H100",
    "type": "bm",
    "region": {"uid": "us-east", "display_name": "US East", "country": "US", "continent": "NA"},
    "gpu": true,
    "spec": {"gpu_count": 8, "cpu_sockets": 2, "memory_mib": 2097152},
    "cost_per_hour": 20,
    "available": 1
  }
]`

func TestInventoryListSendsCurrentFiltersAndDecodesBackendShape(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tha/v3/inventory" {
			t.Errorf("path %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("auth %s", got)
		}
		want := "gpu=true&region=us-east&type=bm"
		if got := r.URL.Query().Encode(); got != want {
			t.Errorf("query=%q, want %q", got, want)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(inventoryResponse))
	})
	gpu := true
	items, err := c.Inventory.List(context.Background(), InventoryListParams{
		Type: " BM ", GPU: &gpu, Region: " US-East ",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 2 || items[0].Name != H100Small || items[0].Spec.MemoryMiB != 131072 ||
		items[0].Spec.CCEnabled == nil || !*items[0].Spec.CCEnabled || items[0].Region != nil {
		t.Fatalf("vm item=%+v", items[0])
	}
	if items[1].Region == nil || items[1].Region.UID != "us-east" || items[1].Spec.GPUCount != 8 {
		t.Fatalf("bm item=%+v", items[1])
	}
}

func TestInventoryAcceptsVMType(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Query().Get("type"); got != "vm" {
			t.Errorf("type=%q", got)
		}
		_, _ = w.Write([]byte(`[]`))
	})
	if _, err := c.Inventory.Capacity(context.Background(), InventoryListParams{Type: "vm"}); err != nil {
		t.Fatal(err)
	}
}

func TestInventoryRejectsDeprecatedAndInvalidFilters(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("should not call API")
	})
	for _, tt := range []struct {
		params InventoryListParams
		field  string
		text   string
	}{
		{InventoryListParams{Type: "rental"}, "type", "deprecated"},
		{InventoryListParams{Type: "storage"}, "type", "deprecated"},
		{InventoryListParams{Type: "serverless"}, "type", "deprecated"},
		{InventoryListParams{Type: "sandbox"}, "type", "one of vm, bm"},
		{InventoryListParams{Type: "   "}, "type", "non-empty"},
		{InventoryListParams{Region: "us_east"}, "region", "slug"},
		{InventoryListParams{Region: strings.Repeat("a", 33)}, "region", "slug"},
	} {
		_, err := c.Inventory.List(context.Background(), tt.params)
		var ve *ValidationError
		if !errors.As(err, &ve) || ve.Field != tt.field || !strings.Contains(ve.Message, tt.text) {
			t.Fatalf("params=%+v err=%v", tt.params, err)
		}
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
