package targon

import (
	"net/http"
	"strings"
)

// Client is the Targon API client. Resource services share this client's
// HTTP transport and organization context.
type Client struct {
	cfg        Config
	http       *http.Client
	streamHTTP *http.Client
	ownsHTTP   bool

	Inventory     *InventoryService
	Workloads     *WorkloadService
	Sandboxes     *SandboxesService
	Volumes       *VolumeService
	SSHKeys       *SSHKeyService
	Projects      *ProjectService
	Orgs          *OrgService
	Members       *MemberService
	APITokens     *APITokenService
	ServiceTokens *ServiceTokenService
	Wallet        *WalletService
	Credits       *CreditsService
}

// New builds a client from cfg. Empty APIKey/Org fall back to the environment
// and the active CLI profile.
func New(cfg Config) (*Client, error) {
	resolved, err := cfg.resolve()
	if err != nil {
		return nil, err
	}
	c := &Client{
		cfg:        resolved,
		http:       newHTTPClient(resolved, true),
		streamHTTP: newHTTPClient(resolved, false),
		ownsHTTP:   true,
	}
	c.streamHTTP.Timeout = 0
	c.initServices()
	return c, nil
}

// NewFromEnv is [New] with credentials taken from the environment / CLI profile.
func NewFromEnv() (*Client, error) {
	return New(DefaultConfig())
}

func (c *Client) initServices() {
	c.Inventory = &InventoryService{client: c}
	c.Workloads = &WorkloadService{client: c}
	c.Sandboxes = &SandboxesService{client: c}
	c.Sandboxes.Templates = &SandboxTemplatesService{client: c}
	c.Sandboxes.Files = &SandboxFilesService{sandboxes: c.Sandboxes}
	c.Sandboxes.Terminals = &SandboxTerminalsService{sandboxes: c.Sandboxes}
	c.Volumes = &VolumeService{client: c}
	c.SSHKeys = &SSHKeyService{client: c}
	c.Projects = &ProjectService{client: c}
	c.Orgs = &OrgService{client: c}
	c.Members = &MemberService{client: c}
	c.APITokens = &APITokenService{client: c}
	c.ServiceTokens = &ServiceTokenService{client: c}
	c.Wallet = &WalletService{client: c}
	c.Credits = &CreditsService{client: c}
}

// Config returns a copy of the resolved client configuration.
func (c *Client) Config() Config {
	return c.cfg
}

// Org returns the bound organization slug, which may be empty.
func (c *Client) Org() string {
	return c.cfg.Org
}

// RequireOrg returns the bound organization or a [ConfigError].
func (c *Client) RequireOrg() (string, error) {
	return c.cfg.RequireOrg()
}

func (c *Client) orgResourcePath(resource string, parts ...string) (string, error) {
	org, err := c.RequireOrg()
	if err != nil {
		return "", err
	}
	path, err := orgPath(org, resource)
	if err != nil {
		return "", err
	}
	return joinPath(path, parts...), nil
}

// ForOrg returns a client bound to slug that shares this client's HTTP transport.
func (c *Client) ForOrg(slug string) (*Client, error) {
	slug = strings.TrimSpace(slug)
	if slug == "" {
		return nil, &ConfigError{Message: "Organization slug must be a non-empty string.", ConfigKey: "org"}
	}
	clone := *c
	clone.cfg.Org = slug
	clone.ownsHTTP = false
	clone.initServices()
	return &clone, nil
}

// Close idle connections. Safe to call on a ForOrg clone (it is a no-op).
func (c *Client) Close() {
	if c == nil || !c.ownsHTTP {
		return
	}
	c.http.CloseIdleConnections()
	c.streamHTTP.CloseIdleConnections()
}
