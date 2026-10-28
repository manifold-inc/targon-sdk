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
}

func TestCommonWorkloadServiceSurface(t *testing.T) {
	var _ commonWorkloads = (*targon.WorkloadService)(nil)

	service := reflect.TypeOf((*targon.WorkloadService)(nil))
	wantMethods := map[string]struct{}{
		"Delete":    {},
		"Get":       {},
		"GetEvents": {},
		"GetState":  {},
		"List":      {},
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
	for _, field := range []string{"Volumes", "Envs", "Commands", "Args", "RegistryAuth"} {
		assertNoField(t, reflect.TypeOf(targon.Workload{}), field)
	}
	assertNoField(t, reflect.TypeOf(targon.WorkloadListItem{}), "Volumes")
}

func assertNoField(t *testing.T, typ reflect.Type, field string) {
	t.Helper()
	if _, ok := typ.FieldByName(field); ok {
		t.Errorf("%s still exports removed field %s", typ.Name(), field)
	}
}
