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

func TestVMRegisterExactWireContractAndHydration(t *testing.T) {
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
				"type":          "VM",
				"name":          "vm-example",
				"image":         "ubuntu-24.04",
				"resource_name": "cpu-small",
				"vm_config":     map[string]any{"password": "secret"},
			}
			if !reflect.DeepEqual(body, want) {
				t.Fatalf("body=%#v, want %#v", body, want)
			}
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm"})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/workloads/wrk-vm"):
			gets.Add(1)
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm", Type: "VM", Name: "vm-example"})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	})

	vm, err := c.VMs.Register(context.Background(), VMCreateParams{
		Name: "vm-example", Image: "ubuntu-24.04",
		ResourceName: "cpu-small", Password: "secret",
	})
	if err != nil {
		t.Fatal(err)
	}
	if vm.UID != "wrk-vm" || vm.Name != "vm-example" || vm.service != c.VMs {
		t.Fatalf("vm=%+v", vm)
	}
	if posts.Load() != 1 || gets.Load() != 1 {
		t.Fatalf("posts=%d gets=%d", posts.Load(), gets.Load())
	}
}

func TestVMRegisterOptionalWireFields(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if body["project_id"] != "proj-1" || body["reservation_uid"] != "res-1" {
				t.Fatalf("body=%#v", body)
			}
			if !reflect.DeepEqual(body["ssh_keys"], []any{"ssh-1"}) {
				t.Fatalf("ssh_keys=%#v", body["ssh_keys"])
			}
			if !reflect.DeepEqual(body["ports"], []any{map[string]any{
				"port": float64(22), "protocol": "TCP", "routing": "PUBLIC",
			}}) {
				t.Fatalf("ports=%#v", body["ports"])
			}
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm"})
		case http.MethodGet:
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm", Type: "VM"})
		}
	})

	_, err := c.VMs.Register(context.Background(), VMCreateParams{
		Name: "vm", Image: "ubuntu", ResourceName: "cpu-small", Password: "secret",
		ProjectID: "proj-1", ReservationUID: "res-1", SSHKeys: []string{"ssh-1"},
		Ports: []PortConfig{{Port: 22, Protocol: "TCP", Routing: "PUBLIC"}},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestVMPasswordValidation(t *testing.T) {
	c := testClient(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("validation should fail before HTTP")
	})
	for _, password := range []string{"", "abc"} {
		_, err := c.VMs.Register(context.Background(), VMCreateParams{
			Name: "vm", Image: "ubuntu", ResourceName: "cpu-small", Password: password,
		})
		var validationErr *ValidationError
		if !errors.As(err, &validationErr) || validationErr.Field != "vm_config.password" {
			t.Fatalf("password=%q error=%T %v", password, err, err)
		}
	}
}

func TestVMCreateDeploysAndWaits(t *testing.T) {
	var states atomic.Int32
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/tha/v3/orgs/acme/workloads":
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm"})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/deploy"):
			_ = json.NewEncoder(w).Encode(WorkloadOperationResponse{UID: "wrk-vm"})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/state"):
			status := "provisioning"
			if states.Add(1) == 2 {
				status = "running"
			}
			_ = json.NewEncoder(w).Encode(WorkloadStateResponse{
				UID: "wrk-vm", WorkloadType: "VM", Status: status,
			})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/workloads/wrk-vm"):
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm", Type: "VM", Name: "ready"})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	})

	vm, err := c.VMs.Create(context.Background(), VMCreateParams{
		Name: "vm", Image: "ubuntu", ResourceName: "cpu-small", Password: "secret",
		Wait: WaitOptions{Timeout: time.Second, PollInterval: time.Millisecond},
	})
	if err != nil || vm.Name != "ready" || vm.service != c.VMs {
		t.Fatalf("vm=%+v err=%v", vm, err)
	}
	if states.Load() != 2 {
		t.Fatalf("state calls=%d", states.Load())
	}
}

func TestVMRejectsTypeMismatch(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-wrong", Type: "BM"})
	})
	_, err := c.VMs.Get(context.Background(), "wrk-wrong")
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Reason != "WORKLOAD_TYPE_MISMATCH" {
		t.Fatalf("error=%T %+v", err, apiErr)
	}
}

func TestVMUpdateAllowsOnlyNameAndPorts(t *testing.T) {
	name := "renamed"
	ports := []PortConfig{{Port: 443, Protocol: "TCP"}}
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPatch {
			t.Fatalf("method=%s", r.Method)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		want := map[string]any{
			"name":  "renamed",
			"ports": []any{map[string]any{"port": float64(443), "protocol": "TCP"}},
		}
		if !reflect.DeepEqual(body, want) {
			t.Fatalf("body=%#v, want %#v", body, want)
		}
		_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm", Type: "VM", Name: name})
	})
	vm, err := c.VMs.Update(context.Background(), "wrk-vm", VMUpdateParams{Name: &name, Ports: &ports})
	if err != nil || vm.Name != name {
		t.Fatalf("vm=%+v err=%v", vm, err)
	}
}

func TestVMResourceForwardsLifecycleAndLogs(t *testing.T) {
	var rebootCalls atomic.Int32
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/logs"):
			if got := r.URL.Query().Get("tail"); got != "25" {
				t.Errorf("tail=%q", got)
			}
			if got := r.URL.Query().Get("type"); got != "serial" {
				t.Errorf("type=%q", got)
			}
			w.Header().Set("Content-Type", "text/plain")
			_, _ = w.Write([]byte("booted\n"))
		case r.Method == http.MethodPatch:
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm", Type: "VM", Name: "renamed"})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/deploy"):
			_ = json.NewEncoder(w).Encode(WorkloadOperationResponse{UID: "wrk-vm"})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/reboot"):
			rebootCalls.Add(1)
			_ = json.NewEncoder(w).Encode(WorkloadOperationResponse{UID: "wrk-vm"})
		case r.Method == http.MethodGet:
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm", Type: "VM", Name: "current"})
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	})

	vm, err := c.VMs.Get(context.Background(), "wrk-vm")
	if err != nil {
		t.Fatal(err)
	}
	name := "renamed"
	if got, err := vm.Update(context.Background(), VMUpdateParams{Name: &name}); err != nil || got != vm || vm.Name != name {
		t.Fatalf("updated=%+v err=%v", got, err)
	}
	if got, err := vm.Deploy(context.Background(), WaitOptions{NoWait: true}); err != nil || got != vm {
		t.Fatalf("deployed=%+v err=%v", got, err)
	}
	if got, err := vm.Reboot(context.Background()); err != nil || got != vm {
		t.Fatalf("rebooted=%+v err=%v", got, err)
	}
	logs, err := vm.Logs(context.Background(), LogOptions{Tail: 25, LogType: "SERIAL", Follow: true})
	if err != nil || logs != "booted\n" {
		t.Fatalf("logs=%q err=%v", logs, err)
	}
	if rebootCalls.Load() != 1 {
		t.Fatalf("reboot calls=%d", rebootCalls.Load())
	}
}

func TestVMMutationsDisableRetries(t *testing.T) {
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

	name := "renamed"
	mutations := []func() error{
		func() error {
			_, err := c.VMs.Register(context.Background(), VMCreateParams{
				Name: "vm", Image: "ubuntu", ResourceName: "cpu-small", Password: "secret",
			})
			return err
		},
		func() error {
			_, err := c.VMs.Deploy(context.Background(), "wrk-vm", WaitOptions{NoWait: true})
			return err
		},
		func() error {
			_, err := c.VMs.Update(context.Background(), "wrk-vm", VMUpdateParams{Name: &name})
			return err
		},
		func() error {
			_, err := c.VMs.Reboot(context.Background(), "wrk-vm")
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
