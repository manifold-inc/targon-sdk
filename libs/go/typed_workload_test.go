package targon

import (
	"context"
	"testing"
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
