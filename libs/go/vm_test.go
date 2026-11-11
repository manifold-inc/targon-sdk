package targon

import (
	"context"
	"encoding/json"
	"errors"
	"io"
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
				"resource_name": "h100-small",
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
		ResourceName: H100Small,
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
			want := map[string]any{
				"type":            "VM",
				"name":            "vm",
				"image":           "ubuntu-24.04",
				"resource_name":   "h100-small",
				"vm_config":       map[string]any{"hostname": "vm-host.example"},
				"project_id":      "proj-1",
				"reservation_uid": "res-1",
				"ssh_keys":        []any{"ssh-1"},
				"ports": []any{map[string]any{
					"port": float64(8080), "protocol": "TCP",
				}},
			}
			if !reflect.DeepEqual(body, want) {
				t.Fatalf("body=%#v, want %#v", body, want)
			}
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm"})
		case http.MethodGet:
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm", Type: "VM"})
		}
	})

	_, err := c.VMs.Register(context.Background(), VMCreateParams{
		Name: "vm", Image: "ubuntu-24.04", ResourceName: H100Small,
		VMConfig:  &VMConfigInput{Hostname: " VM-Host.Example "},
		ProjectID: " proj-1 ", ReservationUID: " res-1 ", SSHKeys: []string{"ssh-1"},
		Ports: []PortConfig{{Port: 8080, Protocol: "TCP"}},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestVMRegisterOmitsEmptyOptionalFields(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			for _, key := range []string{"vm_config", "project_id", "reservation_uid", "ssh_keys", "ports", "password"} {
				if _, ok := body[key]; ok {
					t.Errorf("payload must omit %s: %#v", key, body)
				}
			}
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm"})
		case http.MethodGet:
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm", Type: "VM"})
		}
	})
	_, err := c.VMs.Register(context.Background(), VMCreateParams{
		Name: "vm", Image: "ubuntu-24.04", ResourceName: H100Small,
		VMConfig: &VMConfigInput{Hostname: "  "}, ProjectID: " ", ReservationUID: "",
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestVMRegistrationValidationAndNoPassword(t *testing.T) {
	c := testClient(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("validation should fail before HTTP")
	})
	for _, tt := range []struct {
		name   string
		params VMCreateParams
		field  string
	}{
		{"invalid hostname", VMCreateParams{VMConfig: &VMConfigInput{Hostname: "bad_host"}}, "vm_config.hostname"},
		{"long hostname", VMCreateParams{VMConfig: &VMConfigInput{Hostname: strings.Repeat("a", 64)}}, "vm_config.hostname"},
		{"ssh port", VMCreateParams{Ports: []PortConfig{{Port: 22, Protocol: "TCP"}}}, "ports"},
		{"port range", VMCreateParams{Ports: []PortConfig{{Port: 70000}}}, "ports"},
		{"sctp", VMCreateParams{Ports: []PortConfig{{Port: 8080, Protocol: "SCTP"}}}, "ports"},
		{"duplicate", VMCreateParams{Ports: []PortConfig{{Port: 8080}, {Port: 8080, Protocol: "tcp"}}}, "ports"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			params := tt.params
			params.Name, params.Image, params.ResourceName = "vm", "ubuntu-24.04", H100Small
			_, err := c.VMs.Register(context.Background(), params)
			var validationErr *ValidationError
			if !errors.As(err, &validationErr) || validationErr.Field != tt.field {
				t.Fatalf("error=%T %v", err, err)
			}
		})
	}
	ports := []PortConfig{{Port: 22}}
	_, err := c.VMs.Update(context.Background(), "wrk-vm", VMUpdateParams{Ports: &ports})
	var validationErr *ValidationError
	if !errors.As(err, &validationErr) || validationErr.Field != "ports" {
		t.Fatalf("update error=%T %v", err, err)
	}

	for _, typ := range []reflect.Type{reflect.TypeOf(VMCreateParams{}), reflect.TypeOf(VMConfigInput{})} {
		if _, ok := typ.FieldByName("Password"); ok {
			t.Errorf("%s must not expose Password", typ.Name())
		}
	}
	if fields := reflect.TypeOf(VMConfigInput{}).NumField(); fields != 1 {
		t.Errorf("VMConfigInput fields=%d, want only Hostname", fields)
	}
}

func TestVMPortsNormalizeAndAllowSamePortOnTCPAndUDP(t *testing.T) {
	var got map[string]any
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
				t.Fatal(err)
			}
			_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm", Type: "VM"})
			return
		}
		_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-vm", Type: "VM"})
	})
	_, err := c.VMs.Register(context.Background(), VMCreateParams{
		Name: "vm", Image: "ubuntu-24.04", ResourceName: H100Small,
		Ports: []PortConfig{
			{Port: 8080, Protocol: "udp", Routing: "DIRECT"},
			{Port: 8080},
			{Port: 9090, Protocol: " TCP "},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []any{
		map[string]any{"port": float64(8080), "protocol": "UDP"},
		map[string]any{"port": float64(8080), "protocol": "TCP"},
		map[string]any{"port": float64(9090), "protocol": "TCP"},
	}
	if !reflect.DeepEqual(got["ports"], want) {
		t.Fatalf("ports=%#v, want %#v", got["ports"], want)
	}

	for name, ports := range map[string][]PortConfig{
		"sctp":      {{Port: 8080, Protocol: "SCTP"}},
		"duplicate": {{Port: 8080, Protocol: "TCP"}, {Port: 8080}},
	} {
		ports := ports
		_, err := c.VMs.Update(context.Background(), "wrk-vm", VMUpdateParams{Ports: &ports})
		var validationErr *ValidationError
		if !errors.As(err, &validationErr) || validationErr.Field != "ports" {
			t.Fatalf("%s update error=%T %v", name, err, err)
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
		Name: "vm", Image: "ubuntu", ResourceName: H100Small,
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
			q := r.URL.Query()
			if got := q.Get("tail"); got != "25" {
				t.Errorf("tail=%q", got)
			}
			if got := q.Get("type"); got != "serial" {
				t.Errorf("type=%q", got)
			}
			w.Header().Set("Content-Type", "text/plain")
			if q.Get("follow") == "true" {
				_, _ = w.Write([]byte("streamed\n"))
				return
			}
			if q.Has("follow") {
				t.Errorf("Logs must not send follow: %q", r.URL.RawQuery)
			}
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
	logs, err := vm.Logs(context.Background(), LogOptions{Tail: 25, LogType: "SERIAL"})
	if err != nil || logs != "booted\n" {
		t.Fatalf("logs=%q err=%v", logs, err)
	}
	stream, err := vm.StreamLogs(context.Background(), LogOptions{Tail: 25, LogType: "serial"})
	if err != nil {
		t.Fatal(err)
	}
	streamed, err := io.ReadAll(stream)
	_ = stream.Close()
	if err != nil || string(streamed) != "streamed\n" {
		t.Fatalf("streamed=%q err=%v", streamed, err)
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
				Name: "vm", Image: "ubuntu", ResourceName: H100Small,
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

func TestVMImagesUsesVMImagesRoute(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/tha/v3/orgs/acme/workloads/vm-images" {
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{"name":"ubuntu-24.04","display_name":"Ubuntu 24.04","os_family":"linux","description":"LTS"}]`))
	})
	images, err := c.VMs.Images(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := []VMImage{{Name: "ubuntu-24.04", DisplayName: "Ubuntu 24.04", OSFamily: "linux", Description: "LTS"}}
	if !reflect.DeepEqual(images, want) {
		t.Fatalf("images=%+v", images)
	}
}

// vmWorkloadResponse is GET /workloads/{uid} for a VM in the backend shape.
const vmWorkloadResponse = `{
  "uid": "wrk-vm",
  "type": "VM",
  "created_by": {"token_type": "SERVICE", "token_name": "ci"},
  "project_id": "proj-1",
  "name": "vm",
  "image": "ubuntu-24.04",
  "resource_name": "h100-small",
  "reservation_uid": "res-1",
  "cost_per_hour": 2.5,
  "resource": {"name": "h100-small", "display_name": "H100 Small", "cpu_millicores": 16000, "memory_mib": 131072, "vcpu": 16, "memory": 128},
  "ports": [{"port": 8080, "protocol": "TCP"}],
  "ssh_keys": [{"uid": "ssh-1", "name": "laptop", "public_key_raw": "ssh-ed25519 AAAA"}],
  "vm_config": {"hostname": "vm-host"},
  "state": {"status": "running", "message": "", "public_ip": "203.0.113.10", "ssh_port": 22, "ready_replicas": 1, "total_replicas": 1},
  "created_at": "2026-10-28T12:00:00Z",
  "updated_at": "2026-10-28T12:05:00Z"
}`

func TestVMHydratesBackendResponseShape(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(vmWorkloadResponse))
	})
	vm, err := c.VMs.Get(context.Background(), "wrk-vm")
	if err != nil {
		t.Fatal(err)
	}
	if vm.VMConfig == nil || vm.VMConfig.Hostname != "vm-host" || vm.BMConfig != nil {
		t.Fatalf("vm_config=%+v bm_config=%+v", vm.VMConfig, vm.BMConfig)
	}
	if vm.ReservationUID == nil || *vm.ReservationUID != "res-1" {
		t.Fatalf("reservation_uid=%v", vm.ReservationUID)
	}
	if vm.CreatedBy == nil || vm.CreatedBy.TokenType != "SERVICE" || vm.CreatedBy.TokenName != "ci" {
		t.Fatalf("created_by=%+v", vm.CreatedBy)
	}
	if vm.State == nil || vm.State.SSHPort == nil || *vm.State.SSHPort != 22 {
		t.Fatalf("state=%+v", vm.State)
	}
}
