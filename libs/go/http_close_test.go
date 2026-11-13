package targon

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestClientCloseClosesIdleConnections(t *testing.T) {
	for _, retry := range []bool{false, true} {
		name := "success"
		if retry {
			name = "retry"
		}
		t.Run(name, func(t *testing.T) {
			var calls atomic.Int32
			client, idle, closed := newCloseTestClient(t, func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet || r.URL.Path != APIVersion+"/inventory" {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
				}
				if calls.Add(1) == 1 && retry {
					w.WriteHeader(http.StatusServiceUnavailable)
					_, _ = w.Write([]byte(`{"error":"temporary"}`))
					return
				}
				_, _ = w.Write([]byte(`[{"name":"test-capacity"}]`))
			})

			listCloseTestInventory(t, client)
			wantCalls := int32(1)
			if retry {
				wantCalls++
			}
			if got := calls.Load(); got != wantCalls {
				t.Fatalf("requests = %d, want %d", got, wantCalls)
			}
			conn := waitForCloseTestEvent(t, idle, "idle connection")
			if retry {
				if got := waitForCloseTestEvent(t, idle, "idle connection after retry"); got != conn {
					t.Fatal("retry did not reuse the drained response's connection")
				}
			}

			client.Close()
			if got := waitForCloseTestEvent(t, closed, "Client.Close to close the idle connection"); got != conn {
				t.Fatal("closed a different connection")
			}
		})
	}
}

func TestForOrgClosePreservesIdleConnections(t *testing.T) {
	client, idle, closed := newCloseTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`[{"name":"test-capacity"}]`))
	})
	listCloseTestInventory(t, client)
	conn := waitForCloseTestEvent(t, idle, "idle connection")

	scoped, err := client.ForOrg("other")
	if err != nil {
		t.Fatal(err)
	}
	scoped.Close()
	listCloseTestInventory(t, scoped)
	if got := waitForCloseTestEvent(t, idle, "reused idle connection"); got != conn {
		t.Fatal("ForOrg.Close closed the shared idle connection")
	}
	select {
	case <-closed:
		t.Fatal("shared connection closed after ForOrg.Close")
	default:
	}
}

func TestClientClosePreservesActiveRequests(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{}, 1)
	var calls atomic.Int32
	client, _, _ := newCloseTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			close(started)
		}
		select {
		case <-release:
			_, _ = w.Write([]byte(`[{"name":"test-capacity"}]`))
		case <-r.Context().Done():
		}
	})
	// Release the handler before server cleanup even if an assertion fails.
	t.Cleanup(func() { close(release) })
	ctx, cancel := context.WithTimeout(context.Background(), closeTestTimeout)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		items, err := client.Inventory.List(ctx, InventoryListParams{})
		if err == nil && (len(items) != 1 || items[0].Name != "test-capacity") {
			err = errors.New("unexpected inventory response")
		}
		done <- err
	}()
	select {
	case <-started:
	case <-ctx.Done():
		t.Fatal("timed out waiting for the active request")
	}
	client.Close()
	release <- struct{}{}
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("active request failed after Client.Close: %v", err)
		}
	case <-ctx.Done():
		t.Fatal("timed out waiting for the active request to finish")
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("active request was retried: requests = %d, want 1", got)
	}
}

func TestRetryTransportCloseIdleConnections(t *testing.T) {
	configured := &closeCountingTransport{}
	fallback := &closeCountingTransport{}
	original := http.DefaultTransport
	http.DefaultTransport = fallback
	t.Cleanup(func() { http.DefaultTransport = original })

	for _, tt := range []struct {
		name           string
		base           http.RoundTripper
		wantConfigured int
		wantFallback   int
	}{
		{"configured base", configured, 1, 0},
		{"nil base", nil, 0, 1},
		{"non-closable base", roundTripOnlyTransport{}, 0, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			configured.closes, fallback.closes = 0, 0
			client := &http.Client{Transport: &retryTransport{base: tt.base}}
			client.CloseIdleConnections()
			if configured.closes != tt.wantConfigured || fallback.closes != tt.wantFallback {
				t.Fatalf("close calls: configured = %d, fallback = %d; want %d, %d",
					configured.closes, fallback.closes, tt.wantConfigured, tt.wantFallback)
			}
		})
	}
}

// Test helpers begin.

const closeTestTimeout = 5 * time.Second

// ConnState events let the tests observe socket cleanup before server shutdown.
func newCloseTestClient(t *testing.T, handler http.HandlerFunc) (*Client, <-chan net.Conn, <-chan net.Conn) {
	t.Helper()
	idle := make(chan net.Conn, 16)
	closed := make(chan net.Conn, 16)
	server := httptest.NewUnstartedServer(handler)
	server.Config.ConnState = func(conn net.Conn, state http.ConnState) {
		switch state {
		case http.StateIdle:
			idle <- conn
		case http.StateClosed:
			closed <- conn
		}
	}
	server.Start()
	t.Cleanup(server.Close)
	client, err := New(Config{APIKey: "test-key", Org: "acme", Profile: "test-profile", BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(client.Close)
	return client, idle, closed
}

func listCloseTestInventory(t *testing.T, client *Client) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), closeTestTimeout)
	defer cancel()
	items, err := client.Inventory.List(ctx, InventoryListParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || items[0].Name != "test-capacity" {
		t.Fatalf("unexpected inventory response: %+v", items)
	}
}

func waitForCloseTestEvent(t *testing.T, events <-chan net.Conn, description string) net.Conn {
	t.Helper()
	timer := time.NewTimer(closeTestTimeout)
	defer timer.Stop()
	select {
	case conn := <-events:
		return conn
	case <-timer.C:
		t.Fatalf("timed out waiting for %s", description)
		return nil
	}
}

type roundTripOnlyTransport struct{}

func (roundTripOnlyTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("unexpected RoundTrip")
}

type closeCountingTransport struct {
	roundTripOnlyTransport
	closes int
}

func (t *closeCountingTransport) CloseIdleConnections() {
	t.closes++
}

// Test helpers end.
