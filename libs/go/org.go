package targon

import "context"

// Org is a Targon organization.
type Org struct {
	UID          string  `json:"uid"`
	Slug         string  `json:"slug"`
	Name         string  `json:"name"`
	OrgType      string  `json:"org_type"`
	Role         string  `json:"role"`
	BillingEmail string  `json:"billing_email"`
	Credits      float64 `json:"credits"`
	Overage      int64   `json:"overage"`
	CreatedAt    string  `json:"created_at"`
	UpdatedAt    string  `json:"updated_at"`
}

// Wallet is an organization's billing wallet.
type Wallet struct {
	Address string `json:"address"`
}

// Credits is an organization's credit balance.
type Credits struct {
	Credits  float64 `json:"credits"`
	Currency string  `json:"currency"`
}

// OrgService manages organizations.
type OrgService struct {
	client *Client
}

func (s *OrgService) List(ctx context.Context, page Page) (List[Org], error) {
	var out List[Org]
	err := s.client.do(ctx, "GET", APIVersion+"/orgs", page.query(), nil, &out)
	return out, err
}

func (s *OrgService) Get(ctx context.Context, slug string) (*Org, error) {
	slug, err := requireNonEmpty(slug, "slug")
	if err != nil {
		return nil, err
	}
	var out Org
	err = s.client.do(ctx, "GET", APIVersion+"/orgs/"+slug, nil, nil, &out)
	return &out, err
}

func (s *OrgService) Create(ctx context.Context, name, slug string) (*Org, error) {
	name, err := requireNonEmpty(name, "name")
	if err != nil {
		return nil, err
	}
	slug, err = requireNonEmpty(slug, "slug")
	if err != nil {
		return nil, err
	}
	var out Org
	err = s.client.do(ctx, "POST", APIVersion+"/orgs", nil, map[string]string{"name": name, "slug": slug}, &out)
	return &out, err
}

func (s *OrgService) Update(ctx context.Context, slug string, name, newSlug, billingEmail *string) (*Org, error) {
	slug, err := requireNonEmpty(slug, "slug")
	if err != nil {
		return nil, err
	}
	payload := map[string]string{}
	if name != nil {
		v, err := requireNonEmpty(*name, "name")
		if err != nil {
			return nil, err
		}
		payload["name"] = v
	}
	if newSlug != nil {
		v, err := requireNonEmpty(*newSlug, "new_slug")
		if err != nil {
			return nil, err
		}
		payload["slug"] = v
	}
	if billingEmail != nil {
		payload["billing_email"] = *billingEmail
	}
	if len(payload) == 0 {
		return nil, &ValidationError{Message: "At least one organization field is required"}
	}
	var out Org
	err = s.client.do(ctx, "PATCH", APIVersion+"/orgs/"+slug, nil, payload, &out)
	return &out, err
}

func (s *OrgService) Delete(ctx context.Context, slug string) error {
	slug, err := requireNonEmpty(slug, "slug")
	if err != nil {
		return err
	}
	return s.client.do(ctx, "DELETE", APIVersion+"/orgs/"+slug, nil, nil, nil)
}

// WalletService reads the organization wallet.
type WalletService struct {
	client *Client
}

func (s *WalletService) Get(ctx context.Context) (*Wallet, error) {
	org, err := s.client.RequireOrg()
	if err != nil {
		return nil, err
	}
	var out Wallet
	err = s.client.do(ctx, "GET", APIVersion+"/orgs/"+org+"/wallet", nil, nil, &out)
	return &out, err
}

// CreditsService reads the organization credit balance.
type CreditsService struct {
	client *Client
}

func (s *CreditsService) Get(ctx context.Context) (*Credits, error) {
	org, err := s.client.RequireOrg()
	if err != nil {
		return nil, err
	}
	var out Credits
	err = s.client.do(ctx, "GET", APIVersion+"/orgs/"+org+"/credits", nil, nil, &out)
	return &out, err
}
