package targon

import "context"

// SSHKey is an organization SSH public key.
type SSHKey struct {
	UID          string `json:"uid"`
	Name         string `json:"name"`
	PublicKeyRaw string `json:"public_key_raw"`
	CreatedAt    string `json:"created_at"`
	UpdatedAt    string `json:"updated_at"`
}

// SSHKeyService manages SSH keys.
type SSHKeyService struct {
	client *Client
}

func (s *SSHKeyService) path(uid string) (string, error) {
	org, err := s.client.RequireOrg()
	if err != nil {
		return "", err
	}
	p, err := orgPath(org, "ssh-keys")
	if err != nil {
		return "", err
	}
	return joinPath(p, uid), nil
}

func (s *SSHKeyService) Create(ctx context.Context, name, sshKey string) (*SSHKey, error) {
	name, err := requireNonEmpty(name, "name")
	if err != nil {
		return nil, err
	}
	sshKey, err = requireNonEmpty(sshKey, "ssh_key")
	if err != nil {
		return nil, err
	}
	path, err := s.path("")
	if err != nil {
		return nil, err
	}
	var out SSHKey
	err = s.client.do(ctx, "POST", path, nil, map[string]string{"name": name, "ssh_key": sshKey}, &out)
	return &out, err
}

func (s *SSHKeyService) List(ctx context.Context, page Page) (List[SSHKey], error) {
	path, err := s.path("")
	if err != nil {
		return List[SSHKey]{}, err
	}
	var out List[SSHKey]
	err = s.client.do(ctx, "GET", path, page.query(), nil, &out)
	return out, err
}

func (s *SSHKeyService) Get(ctx context.Context, sshKeyUID string) (*SSHKey, error) {
	sshKeyUID, err := requireNonEmpty(sshKeyUID, "ssh_key_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.path(sshKeyUID)
	if err != nil {
		return nil, err
	}
	var out SSHKey
	err = s.client.do(ctx, "GET", path, nil, nil, &out)
	return &out, err
}

func (s *SSHKeyService) Update(ctx context.Context, sshKeyUID, name string) (*SSHKey, error) {
	sshKeyUID, err := requireNonEmpty(sshKeyUID, "ssh_key_uid")
	if err != nil {
		return nil, err
	}
	name, err = requireNonEmpty(name, "name")
	if err != nil {
		return nil, err
	}
	path, err := s.path(sshKeyUID)
	if err != nil {
		return nil, err
	}
	var out SSHKey
	err = s.client.do(ctx, "PATCH", path, nil, map[string]string{"name": name}, &out)
	return &out, err
}

func (s *SSHKeyService) Delete(ctx context.Context, sshKeyUID string) error {
	sshKeyUID, err := requireNonEmpty(sshKeyUID, "ssh_key_uid")
	if err != nil {
		return err
	}
	path, err := s.path(sshKeyUID)
	if err != nil {
		return err
	}
	return s.client.do(ctx, "DELETE", path, nil, nil, nil)
}
