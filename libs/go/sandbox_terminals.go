package targon

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/coder/websocket"
)

var terminalIDPattern = regexp.MustCompile(`^[A-Za-z0-9._-]{1,64}$`)

type SandboxTerminalsService struct {
	sandboxes *SandboxesService
}

func (s *SandboxTerminalsService) List(ctx context.Context, workloadUID string) ([]TerminalSession, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.sandboxes.path(workloadUID, "terminals")
	if err != nil {
		return nil, err
	}
	var out []TerminalSession
	if err := s.sandboxes.client.do(ctx, http.MethodGet, path, nil, nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *SandboxTerminalsService) Create(ctx context.Context, workloadUID string, cols, rows int) (*TerminalSession, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	if cols == 0 {
		cols = 80
	}
	if rows == 0 {
		rows = 24
	}
	if cols < 1 || cols > MaxSandboxTerminalDimension {
		return nil, validation("cols must be between 1 and 1000", "cols", cols)
	}
	if rows < 1 || rows > MaxSandboxTerminalDimension {
		return nil, validation("rows must be between 1 and 1000", "rows", rows)
	}
	path, err := s.sandboxes.path(workloadUID, "terminals")
	if err != nil {
		return nil, err
	}
	var out TerminalSession
	err = s.sandboxes.client.doNoRetry(ctx, http.MethodPost, path, nil, map[string]int{"cols": cols, "rows": rows}, &out)
	return &out, err
}

func (s *SandboxTerminalsService) Delete(ctx context.Context, workloadUID, terminalID string) error {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return err
	}
	if !terminalIDPattern.MatchString(strings.TrimSpace(terminalID)) {
		return validation("terminal_id must be 1-64 letters, digits, '.', '_', or '-'", "terminal_id", terminalID)
	}
	path, err := s.sandboxes.path(workloadUID, "terminals", terminalID)
	if err != nil {
		return err
	}
	return s.sandboxes.client.doNoRetry(ctx, http.MethodDelete, path, nil, nil, nil)
}

// Connect opens the terminal's raw binary PTY WebSocket. Unless UseBearer is
// set, it mints a fresh single-use access ticket immediately before dialing.
func (s *SandboxTerminalsService) Connect(ctx context.Context, workloadUID, terminalID string, options ConnectTerminalOptions) (*TerminalConnection, error) {
	workloadUID, err := requireNonEmpty(workloadUID, "workload_uid")
	if err != nil {
		return nil, err
	}
	if !terminalIDPattern.MatchString(strings.TrimSpace(terminalID)) {
		return nil, validation("terminal_id must be 1-64 letters, digits, '.', '_', or '-'", "terminal_id", terminalID)
	}
	path, err := s.sandboxes.path(workloadUID, "terminals", terminalID, "ws")
	if err != nil {
		return nil, err
	}
	rawURL, err := websocketURL(s.sandboxes.client.url(path, nil))
	if err != nil {
		return nil, err
	}
	headers := http.Header{}
	if options.UseBearer {
		headers.Set("Authorization", "Bearer "+s.sandboxes.client.cfg.APIKey)
	} else {
		ticket, err := s.sandboxes.MintAccessTicket(ctx, workloadUID, options.TicketTTL)
		if err != nil {
			return nil, err
		}
		if time.Until(ticket.ExpiresAt) < 5*time.Second {
			return nil, &AccessTicketError{APIError: APIError{
				StatusCode:  http.StatusUnauthorized,
				Message:     "newly minted access ticket expires too soon to connect",
				Reason:      "WORKLOAD_ACCESS_TICKET_EXPIRES_TOO_SOON",
				WorkloadUID: workloadUID,
			}}
		}
		rawURL, err = addTicket(rawURL, ticket.Ticket)
		if err != nil {
			return nil, err
		}
	}
	conn, response, err := websocket.Dial(ctx, rawURL, &websocket.DialOptions{
		HTTPClient: s.sandboxes.client.streamHTTP,
		HTTPHeader: headers,
	})
	if err != nil {
		if response != nil {
			defer response.Body.Close()
			raw, _ := io.ReadAll(response.Body)
			if apiErr := apiErrorFromResponse(response, raw); apiErr != nil {
				if !options.UseBearer && response.StatusCode == http.StatusUnauthorized {
					var base *APIError
					if errors.As(apiErr, &base) {
						return nil, &AccessTicketError{APIError: *base}
					}
				}
				return nil, apiErr
			}
		}
		return nil, &NetworkError{Message: "terminal websocket dial failed", Cause: err}
	}
	return &TerminalConnection{conn: conn}, nil
}

type TerminalConnection struct {
	conn *websocket.Conn
}

// Read reads one raw binary PTY frame.
func (c *TerminalConnection) Read(ctx context.Context) ([]byte, error) {
	if c == nil || c.conn == nil {
		return nil, &NetworkError{Message: "terminal connection is closed"}
	}
	messageType, data, err := c.conn.Read(ctx)
	if err != nil {
		return nil, err
	}
	if messageType != websocket.MessageBinary {
		return nil, fmt.Errorf("terminal websocket received non-binary message type %d", messageType)
	}
	return data, nil
}

// Write sends one raw binary PTY frame.
func (c *TerminalConnection) Write(ctx context.Context, data []byte) error {
	if c == nil || c.conn == nil {
		return &NetworkError{Message: "terminal connection is closed"}
	}
	return c.conn.Write(ctx, websocket.MessageBinary, data)
}

func (c *TerminalConnection) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	err := c.conn.Close(websocket.StatusNormalClosure, "")
	c.conn = nil
	return err
}

func websocketURL(raw string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return "", err
	}
	switch u.Scheme {
	case "http":
		u.Scheme = "ws"
	case "https":
		u.Scheme = "wss"
	default:
		return "", fmt.Errorf("unsupported websocket base URL scheme %q", u.Scheme)
	}
	return u.String(), nil
}
