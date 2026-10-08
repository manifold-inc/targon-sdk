package targon

import (
	"fmt"
	"strings"
	"time"
)

// Config holds client authentication and transport settings.
type Config struct {
	APIKey     string
	Org        string
	Profile    string
	BaseURL    string
	Timeout    time.Duration
	MaxRetries int
	// SkipTLSVerify disables TLS certificate verification. Prefer leaving this
	// false except in tests.
	SkipTLSVerify bool
	UserAgent     string
}

// DefaultConfig returns production defaults. APIKey and Org are still resolved
// by [New] / [NewFromEnv].
func DefaultConfig() Config {
	return Config{
		BaseURL:    DefaultBaseURL,
		Timeout:    30 * time.Second,
		MaxRetries: 3,
	}
}

func (c Config) resolve() (Config, error) {
	if c.BaseURL == "" {
		c.BaseURL = DefaultBaseURL
	}
	c.BaseURL = strings.TrimRight(strings.TrimSpace(c.BaseURL), "/")
	if c.BaseURL == "" {
		return Config{}, &ConfigError{Message: "base_url must be a non-empty string", ConfigKey: "base_url"}
	}
	if c.Timeout < 0 {
		return Config{}, &ConfigError{Message: "timeout must be non-negative", ConfigKey: "timeout"}
	}
	if c.Timeout == 0 {
		c.Timeout = 30 * time.Second
	}
	if c.MaxRetries < 0 {
		return Config{}, &ConfigError{Message: "max_retries must be non-negative", ConfigKey: "max_retries"}
	}
	if c.MaxRetries == 0 {
		c.MaxRetries = 3
	}

	c.Profile = getProfile(c.Profile)
	key := strings.TrimSpace(c.APIKey)
	if key == "" {
		key = getAPIKey(c.Profile)
	}
	if key == "" {
		return Config{}, &ConfigError{
			Message:   "API key is required. Provide it via Config.APIKey or set TARGON_API_KEY.",
			ConfigKey: "api_key",
		}
	}
	c.APIKey = key
	c.Org = getOrg(c.Org, c.Profile)
	c.UserAgent = strings.TrimSpace(c.UserAgent)
	if c.UserAgent == "" {
		c.UserAgent = fmt.Sprintf("targon-sdk-go/%s", Version)
	}
	return c, nil
}

// RequireOrg returns the configured organization slug.
func (c Config) RequireOrg() (string, error) {
	if strings.TrimSpace(c.Org) != "" {
		return strings.TrimSpace(c.Org), nil
	}
	return "", &ConfigError{
		Message:   "organization is required. Pass Org on Config, set TARGON_ORG, or select one with `targon org use <slug>`.",
		ConfigKey: "org",
	}
}

func (c Config) headers() map[string]string {
	return map[string]string{
		"Authorization": "Bearer " + c.APIKey,
		"Content-Type":  "application/json",
		"Accept":        "application/json",
		"User-Agent":    c.UserAgent,
	}
}
