package targon

import "context"

// APIToken is a personal API token owned by the authenticated user.
type APIToken struct {
	UID       string  `json:"uid"`
	Name      string  `json:"name"`
	Token     *string `json:"token"`
	CreatedAt string  `json:"created_at"`
	UpdatedAt string  `json:"updated_at"`
}

// TokenCreator is the user who created a service token.
type TokenCreator struct {
	Username  string `json:"username"`
	Email     string `json:"email"`
	FirstName string `json:"first_name"`
	LastName  string `json:"last_name"`
}

// ServiceToken is an organization-owned service token.
type ServiceToken struct {
	UID       string        `json:"uid"`
	Name      string        `json:"name"`
	Token     *string       `json:"token"`
	CreatedBy *TokenCreator `json:"created_by"`
	CreatedAt string        `json:"created_at"`
}

// APITokenService manages personal API tokens.
type APITokenService struct {
	client *Client
}

func (s *APITokenService) List(ctx context.Context, page Page) (List[APIToken], error) {
	var out List[APIToken]
	err := s.client.do(ctx, "GET", APIVersion+"/me/api-tokens", page.query(), nil, &out)
	return out, err
}

func (s *APITokenService) Create(ctx context.Context, name string) (*APIToken, error) {
	name, err := requireNonEmpty(name, "name")
	if err != nil {
		return nil, err
	}
	var out APIToken
	err = s.client.do(ctx, "POST", APIVersion+"/me/api-tokens", nil, map[string]string{"name": name}, &out)
	return &out, err
}

func (s *APITokenService) Update(ctx context.Context, tokenUID, name string) (*APIToken, error) {
	tokenUID, err := requireNonEmpty(tokenUID, "token_uid")
	if err != nil {
		return nil, err
	}
	name, err = requireNonEmpty(name, "name")
	if err != nil {
		return nil, err
	}
	var out APIToken
	err = s.client.do(ctx, "PATCH", APIVersion+"/me/api-tokens/"+tokenUID, nil, map[string]string{"name": name}, &out)
	return &out, err
}

func (s *APITokenService) Delete(ctx context.Context, tokenUID string) error {
	tokenUID, err := requireNonEmpty(tokenUID, "token_uid")
	if err != nil {
		return err
	}
	return s.client.do(ctx, "DELETE", APIVersion+"/me/api-tokens/"+tokenUID, nil, nil, nil)
}

// ServiceTokenService manages organization service tokens.
type ServiceTokenService struct {
	client *Client
}

func (s *ServiceTokenService) path(tokenUID string) (string, error) {
	org, err := s.client.RequireOrg()
	if err != nil {
		return "", err
	}
	p := APIVersion + "/orgs/" + org + "/tokens"
	if tokenUID != "" {
		p += "/" + tokenUID
	}
	return p, nil
}

func (s *ServiceTokenService) List(ctx context.Context, page Page) (List[ServiceToken], error) {
	path, err := s.path("")
	if err != nil {
		return List[ServiceToken]{}, err
	}
	var out List[ServiceToken]
	err = s.client.do(ctx, "GET", path, page.query(), nil, &out)
	return out, err
}

func (s *ServiceTokenService) Create(ctx context.Context, name string) (*ServiceToken, error) {
	name, err := requireNonEmpty(name, "name")
	if err != nil {
		return nil, err
	}
	path, err := s.path("")
	if err != nil {
		return nil, err
	}
	var out ServiceToken
	err = s.client.do(ctx, "POST", path, nil, map[string]string{"name": name}, &out)
	return &out, err
}

func (s *ServiceTokenService) Delete(ctx context.Context, tokenUID string) error {
	tokenUID, err := requireNonEmpty(tokenUID, "token_uid")
	if err != nil {
		return err
	}
	path, err := s.path(tokenUID)
	if err != nil {
		return err
	}
	return s.client.do(ctx, "DELETE", path, nil, nil, nil)
}
