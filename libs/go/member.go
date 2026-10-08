package targon

import (
	"context"
	"strings"
)

var validRoles = map[string]struct{}{"OWNER": {}, "ADMIN": {}, "MEMBER": {}}
var validStatuses = map[string]struct{}{"ACTIVE": {}, "INVITED": {}, "SUSPENDED": {}}

// MemberUser is a user nested in a membership.
type MemberUser struct {
	Username string `json:"username"`
	Email    string `json:"email"`
}

// Member is an organization membership.
type Member struct {
	UID           string      `json:"uid"`
	User          MemberUser  `json:"user"`
	Role          string      `json:"role"`
	Status        string      `json:"status"`
	InvitedByUser *MemberUser `json:"invited_by_user"`
	JoinedAt      *string     `json:"joined_at"`
	CreatedAt     string      `json:"created_at"`
	UpdatedAt     string      `json:"updated_at"`
}

// MemberService manages organization members.
type MemberService struct {
	client *Client
}

func enumValue(value, field string, allowed map[string]struct{}) (string, error) {
	if value == "" {
		return "", nil
	}
	normalized, err := requireNonEmpty(value, field)
	if err != nil {
		return "", err
	}
	normalized = strings.ToUpper(normalized)
	if _, ok := allowed[normalized]; !ok {
		return "", &ValidationError{Message: field + " is not a valid value", Field: field, Value: value}
	}
	return normalized, nil
}

func (s *MemberService) path(username string) (string, error) {
	org, err := s.client.RequireOrg()
	if err != nil {
		return "", err
	}
	p := APIVersion + "/orgs/" + org + "/members"
	if username != "" {
		p += "/" + username
	}
	return p, nil
}

func (s *MemberService) List(ctx context.Context, role, status string, page Page) (List[Member], error) {
	path, err := s.path("")
	if err != nil {
		return List[Member]{}, err
	}
	role, err = enumValue(role, "role", validRoles)
	if err != nil {
		return List[Member]{}, err
	}
	status, err = enumValue(status, "status", validStatuses)
	if err != nil {
		return List[Member]{}, err
	}
	q := page.query()
	if role != "" {
		q.Set("role", role)
	}
	if status != "" {
		q.Set("status", status)
	}
	var out List[Member]
	err = s.client.do(ctx, "GET", path, q, nil, &out)
	return out, err
}

func (s *MemberService) Get(ctx context.Context, username string) (*Member, error) {
	username, err := requireNonEmpty(username, "username")
	if err != nil {
		return nil, err
	}
	path, err := s.path(username)
	if err != nil {
		return nil, err
	}
	var out Member
	err = s.client.do(ctx, "GET", path, nil, nil, &out)
	return &out, err
}

func (s *MemberService) Update(ctx context.Context, username, role string) (*Member, error) {
	username, err := requireNonEmpty(username, "username")
	if err != nil {
		return nil, err
	}
	role, err = enumValue(role, "role", validRoles)
	if err != nil {
		return nil, err
	}
	path, err := s.path(username)
	if err != nil {
		return nil, err
	}
	var out Member
	err = s.client.do(ctx, "PATCH", path, nil, map[string]string{"role": role}, &out)
	return &out, err
}

func (s *MemberService) Delete(ctx context.Context, username string) error {
	username, err := requireNonEmpty(username, "username")
	if err != nil {
		return err
	}
	path, err := s.path(username)
	if err != nil {
		return err
	}
	return s.client.do(ctx, "DELETE", path, nil, nil, nil)
}
