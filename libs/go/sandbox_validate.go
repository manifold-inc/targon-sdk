package targon

import (
	"regexp"
	"strconv"
	"strings"
)

var (
	sandboxNamePattern  = regexp.MustCompile(`^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$`)
	templateNamePattern = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$`)
	terminalIDPattern   = regexp.MustCompile(`^[A-Za-z0-9._-]{1,64}$`)
)

func validateSandboxName(name, field string) error {
	if !sandboxNamePattern.MatchString(strings.TrimSpace(name)) {
		return validation("name must be 1-32 lowercase alphanumeric characters or hyphens, without leading or trailing hyphens", field, name)
	}
	return nil
}

func validateTemplateName(name string) error {
	name = strings.TrimSpace(name)
	if !templateNamePattern.MatchString(name) {
		return validation("name must be 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit", "name", name)
	}
	return nil
}

func validateDisplayName(displayName string) error {
	if len(strings.TrimSpace(displayName)) > 128 {
		return validation("display_name must be at most 128 characters", "display_name", displayName)
	}
	return nil
}

func validateTerminalID(terminalID string) error {
	if !terminalIDPattern.MatchString(strings.TrimSpace(terminalID)) {
		return validation("terminal_id must be 1-64 letters, digits, '.', '_', or '-'", "terminal_id", terminalID)
	}
	return nil
}

func validateSandboxConfigForCreate(config *SandboxConfigInput) error {
	if config == nil {
		return nil
	}
	if config.TTLSec != nil && *config.TTLSec < 0 {
		return validation("sandbox_config.ttl_sec must be zero or positive", "sandbox_config.ttl_sec", *config.TTLSec)
	}
	if config.IdleTimeoutSec != nil && *config.IdleTimeoutSec < 0 {
		return validation("sandbox_config.idle_timeout_sec must be zero or positive", "sandbox_config.idle_timeout_sec", *config.IdleTimeoutSec)
	}
	if config.TTLSec != nil && config.IdleTimeoutSec != nil &&
		*config.TTLSec > 0 && *config.IdleTimeoutSec > 0 && *config.IdleTimeoutSec >= *config.TTLSec {
		return validation("sandbox_config.idle_timeout_sec must be less than ttl_sec", "sandbox_config.idle_timeout_sec", *config.IdleTimeoutSec)
	}
	return nil
}

func validateSandboxConfigForUpdate(config *SandboxConfigInput) error {
	if err := validateSandboxConfigForCreate(config); err != nil || config == nil {
		return err
	}
	if config.IdleTimeoutSec != nil && *config.IdleTimeoutSec == 0 {
		return validation("sandbox_config.idle_timeout_sec must be positive", "sandbox_config.idle_timeout_sec", *config.IdleTimeoutSec)
	}
	return nil
}

func validateSandboxPorts(ports []PortConfig) error {
	seen := map[string]struct{}{}
	for _, port := range ports {
		if port.Port < 1 || port.Port > 65535 {
			return validation("sandbox port must be between 1 and 65535", "ports", port.Port)
		}
		if port.Port == 22 {
			return validation("sandbox port 22 is reserved for SSH and cannot be forwarded", "ports", port.Port)
		}
		protocol := strings.ToUpper(strings.TrimSpace(port.Protocol))
		if protocol == "" {
			protocol = "TCP"
		}
		if protocol != "TCP" && protocol != "UDP" {
			return validation("sandbox ports only support TCP or UDP", "ports", port.Protocol)
		}
		key := strconv.Itoa(port.Port) + "/" + protocol
		if _, ok := seen[key]; ok {
			return validation("duplicate sandbox port and protocol combination", "ports", key)
		}
		seen[key] = struct{}{}
	}
	return nil
}

func normalizedSandboxPorts(ports []PortConfig) []PortConfig {
	out := append([]PortConfig(nil), ports...)
	for i := range out {
		out[i].Protocol = strings.ToUpper(strings.TrimSpace(out[i].Protocol))
		if out[i].Protocol == "" {
			out[i].Protocol = "TCP"
		}
		out[i].Routing = ""
	}
	return out
}
