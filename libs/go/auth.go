package targon

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/pelletier/go-toml/v2"
)

const (
	apiKeyEnv      = "TARGON_API_KEY"
	orgEnv         = "TARGON_ORG"
	defaultProfile = "default"
	configFileName = "config.toml"
)

type cliConfigFile struct {
	Current  string                    `toml:"current"`
	Profiles map[string]cliProfileFile `toml:"profiles"`
}

type cliProfileFile struct {
	Org     string `toml:"org"`
	BaseURL string `toml:"base_url"`
}

func targonDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".targon"), nil
}

func credentialsFile(profile string) (string, error) {
	dir, err := targonDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "credentials-"+profile), nil
}

func readConfigFile() (*cliConfigFile, error) {
	dir, err := targonDir()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(dir, configFileName))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var parsed cliConfigFile
	if err := toml.Unmarshal(data, &parsed); err != nil {
		return nil, err
	}
	return &parsed, nil
}

func getProfile(profile string) string {
	if strings.TrimSpace(profile) != "" {
		return strings.TrimSpace(profile)
	}
	cfg, err := readConfigFile()
	if err != nil || cfg == nil {
		return defaultProfile
	}
	if strings.TrimSpace(cfg.Current) != "" {
		return strings.TrimSpace(cfg.Current)
	}
	return defaultProfile
}

func getAPIKey(profile string) string {
	if env := strings.TrimSpace(os.Getenv(apiKeyEnv)); env != "" {
		return env
	}
	path, err := credentialsFile(getProfile(profile))
	if err != nil {
		return ""
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func getProfileOrg(profile string) string {
	cfg, err := readConfigFile()
	if err != nil || cfg == nil {
		return ""
	}
	name := getProfile(profile)
	p, ok := cfg.Profiles[name]
	if !ok {
		return ""
	}
	return strings.TrimSpace(p.Org)
}

func getOrg(org, profile string) string {
	if strings.TrimSpace(org) != "" {
		return strings.TrimSpace(org)
	}
	if env := strings.TrimSpace(os.Getenv(orgEnv)); env != "" {
		return env
	}
	return getProfileOrg(profile)
}
