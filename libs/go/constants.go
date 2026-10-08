package targon

import (
	"fmt"
	"strings"
)

// DefaultBaseURL is the production Targon API host.
const DefaultBaseURL = "https://api.targon.com"

// APIVersion is the v3 API path prefix.
const APIVersion = "/tha/v3"

func orgPath(org, resource string) (string, error) {
	org = strings.TrimSpace(org)
	if org == "" || strings.Contains(org, "/") {
		return "", &ValidationError{Message: "org must be a non-empty organization slug", Field: "org"}
	}
	resource = strings.TrimSpace(strings.TrimPrefix(resource, "/"))
	if resource == "" {
		return "", &ValidationError{Message: "resource must be a non-empty path", Field: "resource"}
	}
	return fmt.Sprintf("%s/orgs/%s/%s", APIVersion, org, resource), nil
}

func joinPath(base string, parts ...string) string {
	for _, part := range parts {
		part = strings.Trim(part, "/")
		if part == "" {
			continue
		}
		base = strings.TrimRight(base, "/") + "/" + part
	}
	return base
}
