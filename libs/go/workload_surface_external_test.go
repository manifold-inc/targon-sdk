package targon_test

import (
	"context"
	"reflect"
	"testing"

	targon "github.com/manifold-inc/targon-sdk/libs/go/v4"
)

type commonWorkloads interface {
	Get(context.Context, string) (*targon.Workload, error)
	List(context.Context, targon.ListWorkloadsParams) (targon.List[targon.WorkloadListItem], error)
	Delete(context.Context, string) error
	GetState(context.Context, string) (*targon.WorkloadStateResponse, error)
	GetEvents(context.Context, string, targon.Page) (targon.List[targon.WorkloadEvent], error)
	WaitUntilReady(context.Context, string, targon.WaitOptions) (*targon.WorkloadStateResponse, error)
	Verify(context.Context, string, string) (bool, error)
	AttachSSHKey(context.Context, string, string) (*targon.SSHKeyAttachResponse, error)
	DetachSSHKey(context.Context, string, string) error
}

// TestCommonWorkloadServiceSurface pins the common service to operations the
// backend accepts for every workload type. Generic Create, Update, Deploy,
// Suspend, Exec, and Volume operations stay removed; logs and images live on
// the typed VM and Bare Metal services.
func TestCommonWorkloadServiceSurface(t *testing.T) {
	var _ commonWorkloads = (*targon.WorkloadService)(nil)

	service := reflect.TypeOf((*targon.WorkloadService)(nil))
	wantMethods := map[string]struct{}{
		"AttachSSHKey":   {},
		"Delete":         {},
		"DetachSSHKey":   {},
		"Get":            {},
		"GetEvents":      {},
		"GetState":       {},
		"List":           {},
		"Verify":         {},
		"WaitUntilReady": {},
	}
	for i := 0; i < service.NumMethod(); i++ {
		method := service.Method(i)
		if _, ok := wantMethods[method.Name]; !ok {
			t.Errorf("WorkloadService still exports removed method %s", method.Name)
		}
		delete(wantMethods, method.Name)
	}
	for method := range wantMethods {
		t.Errorf("WorkloadService is missing common method %s", method)
	}

	assertNoField(t, reflect.TypeOf(targon.Client{}), "Volumes")
	assertNoField(t, reflect.TypeOf(targon.Client{}), "Rentals")
	for _, field := range []string{"Volumes", "Envs", "Commands", "Args", "RegistryAuth", "Region", "Hostname"} {
		assertNoField(t, reflect.TypeOf(targon.Workload{}), field)
	}
	assertNoField(t, reflect.TypeOf(targon.WorkloadListItem{}), "Volumes")
}

func TestTypedServicesExposeRestoredTypeSpecificOperations(t *testing.T) {
	for _, tt := range []struct {
		typ     reflect.Type
		methods []string
	}{
		{reflect.TypeOf((*targon.VMService)(nil)), []string{"Logs", "StreamLogs", "Images", "Reboot"}},
		{reflect.TypeOf((*targon.VM)(nil)), []string{"Logs", "StreamLogs", "Reboot"}},
		{reflect.TypeOf((*targon.BareMetalService)(nil)), []string{"Images", "Reboot", "PowerOn", "PowerOff", "Reinstall"}},
	} {
		for _, method := range tt.methods {
			if _, ok := tt.typ.MethodByName(method); !ok {
				t.Errorf("%s is missing %s", tt.typ, method)
			}
		}
	}
	for _, method := range []string{"PowerOn", "PowerOff", "Reinstall", "Password"} {
		if _, ok := reflect.TypeOf((*targon.VMService)(nil)).MethodByName(method); ok {
			t.Errorf("VMService must not expose %s", method)
		}
	}
	for _, method := range []string{"Logs", "StreamLogs"} {
		if _, ok := reflect.TypeOf((*targon.BareMetalService)(nil)).MethodByName(method); ok {
			t.Errorf("BareMetalService must not expose %s", method)
		}
	}
}

func assertNoField(t *testing.T, typ reflect.Type, field string) {
	t.Helper()
	if _, ok := typ.FieldByName(field); ok {
		t.Errorf("%s still exports removed field %s", typ.Name(), field)
	}
}
