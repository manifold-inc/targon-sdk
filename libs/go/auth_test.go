package targon

import (
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"testing"
)

func TestRequireNonEmpty(t *testing.T) {
	_, err := requireNonEmpty("  ", "name")
	var ve *ValidationError
	if !errors.As(err, &ve) || ve.Field != "name" {
		t.Fatalf("expected ValidationError, got %v", err)
	}
}

func TestMapAPIError(t *testing.T) {
	cases := []struct {
		status int
		check  func(error) bool
	}{
		{401, IsUnauthorized},
		{404, IsNotFound},
		{429, IsRateLimited},
	}
	for _, tc := range cases {
		err := mapAPIError(tc.status, "nope", "reason", "req-1")
		if !tc.check(err) {
			t.Fatalf("status %d: %T %v", tc.status, err, err)
		}
	}
}

func TestAuthPrecedence(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("TARGON_API_KEY", "")
	t.Setenv("TARGON_ORG", "")
	dir := filepath.Join(home, ".targon")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.toml"), []byte(`
current = "work"
[profiles.work]
org = "acme"
base_url = "https://api.targon.com"
`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "credentials-work"), []byte("file-key\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	if got := getAPIKey(""); got != "file-key" {
		t.Fatalf("file key: got %q", got)
	}
	t.Setenv("TARGON_API_KEY", "env-key")
	if got := getAPIKey(""); got != "env-key" {
		t.Fatalf("env key should win: got %q", got)
	}
	if got := getOrg("", ""); got != "acme" {
		t.Fatalf("profile org: got %q", got)
	}
	t.Setenv("TARGON_ORG", "from-env")
	if got := getOrg("", ""); got != "from-env" {
		t.Fatalf("env org should win: got %q", got)
	}
	if got := getOrg("explicit", ""); got != "explicit" {
		t.Fatalf("explicit org should win: got %q", got)
	}
}

func TestConfigRequiresAPIKey(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("TARGON_API_KEY", "")
	_, err := Config{}.resolve()
	var ce *ConfigError
	if !errors.As(err, &ce) || ce.ConfigKey != "api_key" {
		t.Fatalf("expected api_key ConfigError, got %v", err)
	}
}

func TestConfigRequireOrg(t *testing.T) {
	cfg := Config{APIKey: "k"}
	_, err := cfg.RequireOrg()
	var ce *ConfigError
	if !errors.As(err, &ce) || ce.ConfigKey != "org" {
		t.Fatalf("expected org ConfigError, got %v", err)
	}
}

func TestDefaultUserAgent(t *testing.T) {
	cfg, err := Config{APIKey: "k"}.resolve()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.UserAgent != "targon-sdk-go/"+Version {
		t.Fatalf("user agent: %q", cfg.UserAgent)
	}
	if cfg.MaxRetries != 3 {
		t.Fatalf("retries: %d", cfg.MaxRetries)
	}
	_ = http.StatusOK
}
