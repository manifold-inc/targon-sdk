package targon

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"testing"
)

func TestNormalizeSSHKeys(t *testing.T) {
	if got, err := normalizeSSHKeys(nil); got != nil || err != nil {
		t.Fatalf("nil: got=%v err=%v", got, err)
	}
	got, err := normalizeSSHKeys([]string{})
	if err != nil || got == nil || len(got) != 0 {
		t.Fatalf("empty: got=%#v err=%v", got, err)
	}
	got, err = normalizeSSHKeys([]string{" key-1 ", "key-2"})
	if err != nil || !reflect.DeepEqual(got, []string{"key-1", "key-2"}) {
		t.Fatalf("trim: got=%v err=%v", got, err)
	}
	for _, keys := range [][]string{{""}, {"key-1", "  "}, {"\t\n"}} {
		_, err := normalizeSSHKeys(keys)
		var validationErr *ValidationError
		if !errors.As(err, &validationErr) || validationErr.Field != "ssh_keys" {
			t.Fatalf("keys=%q err=%T %v", keys, err, err)
		}
	}
}

func TestBlankSSHKeysAreRejectedBeforeRequests(t *testing.T) {
	c := testClient(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("validation should fail before HTTP")
	})
	ctx := context.Background()
	blank := []string{"ssh-1", " "}
	sandboxBlank := []string{""}

	calls := map[string]func() error{
		"vm register": func() error {
			_, err := c.VMs.Register(ctx, VMCreateParams{
				Name: "vm", Image: "ubuntu-24.04", ResourceName: H100Small, SSHKeys: blank,
			})
			return err
		},
		"vm create": func() error {
			_, err := c.VMs.Create(ctx, VMCreateParams{
				Name: "vm", Image: "ubuntu-24.04", ResourceName: H100Small, SSHKeys: blank,
			})
			return err
		},
		"bare metal register": func() error {
			_, err := c.BareMetal.Register(ctx, BareMetalCreateParams{
				Name: "metal", Image: "ubuntu", ResourceName: "bm-a100",
				Region: "us-central", SSHKeys: blank,
			})
			return err
		},
		"bare metal blank only": func() error {
			_, err := c.BareMetal.Register(ctx, BareMetalCreateParams{
				Name: "metal", Image: "ubuntu", ResourceName: "bm-a100",
				Region: "us-central", SSHKeys: []string{"  "},
			})
			return err
		},
		"bare metal reinstall": func() error {
			_, err := c.BareMetal.Reinstall(ctx, "wrk-bm", BareMetalReinstallParams{
				Image: "ubuntu", SSHKeys: blank,
			})
			return err
		},
		"sandbox create": func() error {
			_, err := c.Sandboxes.Create(ctx, SandboxCreateParams{
				Name: "valid", Template: readySandboxTemplate(c, "sbt-base"), SSHKeys: sandboxBlank,
			})
			return err
		},
		"sandbox update": func() error {
			_, err := c.Sandboxes.Update(ctx, "wrk-1", SandboxUpdateParams{SSHKeys: &blank})
			return err
		},
	}
	for name, call := range calls {
		t.Run(name, func(t *testing.T) {
			err := call()
			var validationErr *ValidationError
			if !errors.As(err, &validationErr) || validationErr.Field != "ssh_keys" {
				t.Fatalf("error=%T %v", err, err)
			}
		})
	}
}

func TestSSHKeysAreTrimmedInRegistrationPayloads(t *testing.T) {
	var bodies []map[string]any
	lastType := "VM"
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			bodies = append(bodies, body)
			lastType = body["type"].(string)
		}
		_ = json.NewEncoder(w).Encode(Workload{UID: "wrk-1", Type: lastType})
	})
	ctx := context.Background()
	if _, err := c.VMs.Register(ctx, VMCreateParams{
		Name: "vm", Image: "ubuntu-24.04", ResourceName: H100Small, SSHKeys: []string{" ssh-1 "},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := c.BareMetal.Register(ctx, BareMetalCreateParams{
		Name: "metal", Image: "ubuntu", ResourceName: "bm-a100",
		Region: "us-central", SSHKeys: []string{" ssh-2", "ssh-3 "},
	}); err != nil {
		t.Fatal(err)
	}
	if len(bodies) != 2 {
		t.Fatalf("bodies=%d", len(bodies))
	}
	if got := bodies[0]["ssh_keys"]; !reflect.DeepEqual(got, []any{"ssh-1"}) {
		t.Fatalf("vm ssh_keys=%v", got)
	}
	if got := bodies[1]["ssh_keys"]; !reflect.DeepEqual(got, []any{"ssh-2", "ssh-3"}) {
		t.Fatalf("bm ssh_keys=%v", got)
	}
}
