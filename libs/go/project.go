package targon

import "context"

// Project is an organization project.
type Project struct {
	UID       string `json:"uid"`
	Name      string `json:"name"`
	CreatedAt string `json:"created_at"`
	UpdatedAt string `json:"updated_at"`
}

// ProjectService manages projects.
type ProjectService struct {
	client *Client
}

func (s *ProjectService) path(uid string) (string, error) {
	return s.client.orgResourcePath("projects", uid)
}

func (s *ProjectService) Create(ctx context.Context, name string) (*Project, error) {
	name, err := requireNonEmpty(name, "name")
	if err != nil {
		return nil, err
	}
	path, err := s.path("")
	if err != nil {
		return nil, err
	}
	var out Project
	err = s.client.do(ctx, "POST", path, nil, map[string]string{"name": name}, &out)
	return &out, err
}

func (s *ProjectService) List(ctx context.Context, page Page) (List[Project], error) {
	path, err := s.path("")
	if err != nil {
		return List[Project]{}, err
	}
	var out List[Project]
	err = s.client.do(ctx, "GET", path, page.query(), nil, &out)
	return out, err
}

func (s *ProjectService) Get(ctx context.Context, projectUID string) (*Project, error) {
	projectUID, err := requireNonEmpty(projectUID, "project_uid")
	if err != nil {
		return nil, err
	}
	path, err := s.path(projectUID)
	if err != nil {
		return nil, err
	}
	var out Project
	err = s.client.do(ctx, "GET", path, nil, nil, &out)
	return &out, err
}

func (s *ProjectService) Update(ctx context.Context, projectUID, name string) (*Project, error) {
	projectUID, err := requireNonEmpty(projectUID, "project_uid")
	if err != nil {
		return nil, err
	}
	name, err = requireNonEmpty(name, "name")
	if err != nil {
		return nil, err
	}
	path, err := s.path(projectUID)
	if err != nil {
		return nil, err
	}
	var out Project
	err = s.client.do(ctx, "PATCH", path, nil, map[string]string{"name": name}, &out)
	return &out, err
}

func (s *ProjectService) Delete(ctx context.Context, projectUID string) error {
	projectUID, err := requireNonEmpty(projectUID, "project_uid")
	if err != nil {
		return err
	}
	path, err := s.path(projectUID)
	if err != nil {
		return err
	}
	return s.client.do(ctx, "DELETE", path, nil, nil, nil)
}
