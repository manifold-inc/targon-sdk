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

// bareMetalWorkloadResponse is GET /workloads/{uid} for BM in the backend
// shape: region and hostname are nested under bm_config.
const bareMetalWorkloadResponse = `{
  "uid": "wrk-bm",
  "type": "BM",
  "created_by": {"token_type": "USER", "email": "dev@example.com", "username": "dev"},
  "name": "metal-example",
  "image": "ubuntu-24.04",
  "resource_name": "bm-a100",
  "reservation_uid": "res-bm",
  "ssh_keys": [{"uid": "ssh-1", "name": "laptop", "public_key_raw": "ssh-ed25519 AAAA"}],
  "bm_config": {"region": "us-central", "hostname": "metal-01"},
  "state": {"status": "registered", "message": "Workload registered, awaiting deploy", "ready_replicas": 0, "total_replicas": 0},
  "created_at": "2026-10-28T12:00:00Z",
  "updated_at": "2026-10-28T12:00:00Z"
}`

func TestBareMetalRegisterExactWireContractAndHydration(t *testing.T) {
	var posts, gets atomic.Int32
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/tha/v3/orgs/acme/workloads":
			posts.Add(1)
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			want := map[string]any{
				"type":            "BM",
				"name":            "metal-example",
				"image":           "ubuntu-24.04",
				"resource_name":   "bm-a100",
				"region":          "us-central",
				"bm_config":       map[string]any{"hostname": "metal-01"},
				"project_id":      "proj-1",
				"reservation_uid": "res-bm",
				"ssh_keys":        []any{"ssh-1", "ssh-2"},
			}
			if !reflect.DeepEqual(body, want) {
				t.Fatalf("body=%#v, want %#v", body, want)
			}
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-bm"})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/workloads/wrk-bm"):
			gets.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(bareMetalWorkloadResponse))
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	})

	metal, err := c.BareMetal.Register(context.Background(), BareMetalCreateParams{
		Name: "metal-example", Image: "ubuntu-24.04", ResourceName: "bm-a100",
		Region: "us-central", BMConfig: &BMConfigInput{Hostname: "Metal-01"},
		ProjectID: " proj-1 ", ReservationUID: "res-bm",
		SSHKeys: []string{"ssh-1", "ssh-2"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if metal.UID != "wrk-bm" || metal.BMConfig == nil || metal.BMConfig.Region != "us-central" ||
		metal.BMConfig.Hostname != "metal-01" || metal.service != c.BareMetal {
		t.Fatalf("bare metal=%+v", metal)
	}
	if metal.ReservationUID == nil || *metal.ReservationUID != "res-bm" ||
		metal.CreatedBy == nil || metal.CreatedBy.Email != "dev@example.com" || metal.VMConfig != nil {
		t.Fatalf("bare metal=%+v", metal)
	}
	if posts.Load() != 1 || gets.Load() != 1 {
		t.Fatalf("posts=%d gets=%d", posts.Load(), gets.Load())
	}
}

func TestBareMetalRegistrationValidationAndTypedSurface(t *testing.T) {
	c := testClient(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("validation should fail before HTTP")
	})
	base := BareMetalCreateParams{
		Name: "metal", Image: "ubuntu", ResourceName: "bm-a100",
		Region: "us-central", SSHKeys: []string{"ssh-1"},
	}
	tests := []struct {
		name   string
		params BareMetalCreateParams
		field  string
	}{
		{name: "missing name", params: func() BareMetalCreateParams {
			p := base
			p.Name = ""
			return p
		}(), field: "name"},
		{name: "missing image", params: func() BareMetalCreateParams {
			p := base
			p.Image = ""
			return p
		}(), field: "image"},
		{name: "missing resource name", params: func() BareMetalCreateParams {
			p := base
			p.ResourceName = ""
			return p
		}(), field: "resource_name"},
		{name: "missing region", params: func() BareMetalCreateParams {
			p := base
			p.Region = ""
			return p
		}(), field: "region"},
		{name: "missing SSH keys", params: func() BareMetalCreateParams {
			p := base
			p.SSHKeys = nil
			return p
		}(), field: "ssh_keys"},
		{name: "invalid hostname", params: func() BareMetalCreateParams {
			p := base
			p.BMConfig = &BMConfigInput{Hostname: "-metal"}
			return p
		}(), field: "bm_config.hostname"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := c.BareMetal.Register(context.Background(), tt.params)
			var validationErr *ValidationError
			if !errors.As(err, &validationErr) || validationErr.Field != tt.field {
				t.Fatalf("error=%T %v", err, err)
			}
		})
	}

	typ := reflect.TypeOf(BareMetalCreateParams{})
	for _, field := range []string{
		"Ports", "Envs", "Environment", "Commands", "RegistryAuth", "VMConfig", "Hostname",
	} {
		if _, ok := typ.FieldByName(field); ok {
			t.Errorf("BareMetalCreateParams must not expose %s", field)
		}
	}
}

func TestBareMetalCreateDeploysAndWaits(t *testing.T) {
	var states atomic.Int32
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/tha/v3/orgs/acme/workloads":
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-bm"})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/deploy"):
			_ = json.NewEncoder(w).Encode(WorkloadOperationResponse{UID: "wrk-bm"})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/state"):
			status := "provisioning"
			if states.Add(1) == 2 {
				status = "running"
			}
			_ = json.NewEncoder(w).Encode(WorkloadStateResponse{
				UID: "wrk-bm", WorkloadType: "BM", Status: status,
			})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/workloads/wrk-bm"):
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-bm", Type: "BM", Name: "ready"})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	})

	metal, err := c.BareMetal.Create(context.Background(), BareMetalCreateParams{
		Name: "metal", Image: "ubuntu", ResourceName: "bm-a100",
		Region: "us-central", SSHKeys: []string{"ssh-1"},
		Wait: WaitOptions{Timeout: time.Second, PollInterval: time.Millisecond},
	})
	if err != nil || metal.Name != "ready" || metal.service != c.BareMetal {
		t.Fatalf("bare metal=%+v err=%v", metal, err)
	}
	if states.Load() != 2 {
		t.Fatalf("state calls=%d", states.Load())
	}
}

func TestBareMetalRejectsTypeMismatch(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-wrong", Type: "VM"})
	})
	_, err := c.BareMetal.Get(context.Background(), "wrk-wrong")
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Reason != "WORKLOAD_TYPE_MISMATCH" {
		t.Fatalf("error=%T %+v", err, apiErr)
	}
}

func TestBareMetalOperationsUseExactRoutesAndPayloads(t *testing.T) {
	var actions []string
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost:
			actions = append(actions, r.URL.Path)
			if strings.HasSuffix(r.URL.Path, "/reinstall") {
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				want := map[string]any{
					"image": "ubuntu-24.04", "ssh_keys": []any{"ssh-2"}, "hostname": "metal-new",
				}
				if !reflect.DeepEqual(body, want) {
					t.Fatalf("body=%#v, want %#v", body, want)
				}
			}
			_ = json.NewEncoder(w).Encode(WorkloadOperationResponse{UID: "wrk-bm"})
		case r.Method == http.MethodGet:
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-bm", Type: "BM"})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	})

	metal, err := c.BareMetal.Get(context.Background(), "wrk-bm")
	if err != nil {
		t.Fatal(err)
	}
	operations := []func() (*BareMetal, error){
		func() (*BareMetal, error) { return metal.Reboot(context.Background()) },
		func() (*BareMetal, error) { return metal.PowerOn(context.Background()) },
		func() (*BareMetal, error) { return metal.PowerOff(context.Background()) },
		func() (*BareMetal, error) {
			return metal.Reinstall(context.Background(), BareMetalReinstallParams{
				Image: "ubuntu-24.04", SSHKeys: []string{"ssh-2"}, Hostname: "metal-new",
			})
		},
	}
	for _, operation := range operations {
		got, err := operation()
		if err != nil || got != metal {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	}
	want := []string{
		"/tha/v3/orgs/acme/workloads/wrk-bm/reboot",
		"/tha/v3/orgs/acme/workloads/wrk-bm/power-on",
		"/tha/v3/orgs/acme/workloads/wrk-bm/power-off",
		"/tha/v3/orgs/acme/workloads/wrk-bm/reinstall",
	}
	if !reflect.DeepEqual(actions, want) {
		t.Fatalf("actions=%#v, want %#v", actions, want)
	}
}

func TestBareMetalReinstallValidation(t *testing.T) {
	c := testClient(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("validation should fail before HTTP")
	})
	tests := []struct {
		name   string
		params BareMetalReinstallParams
		field  string
	}{
		{name: "missing image", params: BareMetalReinstallParams{SSHKeys: []string{"ssh-1"}}, field: "image"},
		{name: "invalid hostname", params: BareMetalReinstallParams{Image: "ubuntu", Hostname: "bad_host"}, field: "hostname"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := c.BareMetal.Reinstall(context.Background(), "wrk-bm", tt.params)
			var validationErr *ValidationError
			if !errors.As(err, &validationErr) || validationErr.Field != tt.field {
				t.Fatalf("error=%T %v", err, err)
			}
		})
	}
}

func TestBareMetalReinstallReusesAttachedKeysWhenOmitted(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if want := map[string]any{"image": "ubuntu-24.04"}; !reflect.DeepEqual(body, want) {
				t.Fatalf("body=%#v, want %#v", body, want)
			}
			_ = json.NewEncoder(w).Encode(WorkloadOperationResponse{UID: "wrk-bm"})
		default:
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-bm", Type: "BM"})
		}
	})
	if _, err := c.BareMetal.Reinstall(context.Background(), "wrk-bm", BareMetalReinstallParams{Image: "ubuntu-24.04"}); err != nil {
		t.Fatal(err)
	}
}

func TestBareMetalImagesUsesBMImagesRoute(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/tha/v3/orgs/acme/workloads/bm-images" {
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{"name":"ubuntu-22.04-bm","display_name":"Ubuntu 22.04","os_family":"linux"}]`))
	})
	images, err := c.BareMetal.Images(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(images) != 1 || images[0].Name != "ubuntu-22.04-bm" || images[0].OSFamily != "linux" {
		t.Fatalf("images=%+v", images)
	}
}

func TestBareMetalDefaultWaitIsThirtyMinutes(t *testing.T) {
	if DefaultBareMetalWaitTimeout != 30*time.Minute {
		t.Fatalf("DefaultBareMetalWaitTimeout=%s", DefaultBareMetalWaitTimeout)
	}
	if got := bareMetalWait(WaitOptions{}); got.Timeout != 30*time.Minute {
		t.Fatalf("default timeout=%s", got.Timeout)
	}
	explicit := WaitOptions{Timeout: time.Minute, PollInterval: time.Second, NoWait: true}
	if got := bareMetalWait(explicit); got != explicit {
		t.Fatalf("explicit wait changed: %+v", got)
	}
}

func TestBareMetalMutationsDisableRetries(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"temporary","reason":"UNAVAILABLE"}`))
	}))
	defer srv.Close()
	c, err := New(Config{
		APIKey: "test", Org: "acme", BaseURL: srv.URL, MaxRetries: 2, Timeout: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	mutations := []func() error{
		func() error {
			_, err := c.BareMetal.Register(context.Background(), BareMetalCreateParams{
				Name: "metal", Image: "ubuntu", ResourceName: "bm-a100",
				Region: "us-central", SSHKeys: []string{"ssh-1"},
			})
			return err
		},
		func() error {
			_, err := c.BareMetal.Deploy(context.Background(), "wrk-bm", WaitOptions{NoWait: true})
			return err
		},
		func() error { _, err := c.BareMetal.Reboot(context.Background(), "wrk-bm"); return err },
		func() error { _, err := c.BareMetal.PowerOn(context.Background(), "wrk-bm"); return err },
		func() error { _, err := c.BareMetal.PowerOff(context.Background(), "wrk-bm"); return err },
		func() error {
			_, err := c.BareMetal.Reinstall(context.Background(), "wrk-bm", BareMetalReinstallParams{
				Image: "ubuntu", SSHKeys: []string{"ssh-1"},
			})
			return err
		},
	}
	for _, mutation := range mutations {
		before := calls.Load()
		if err := mutation(); err == nil {
			t.Fatal("expected mutation error")
		}
		if delta := calls.Load() - before; delta != 1 {
			t.Fatalf("mutation requests=%d, want 1", delta)
		}
	}
}
