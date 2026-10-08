package targon

import (
	"net/url"
	"strconv"
)

// Page is a cursor pagination request.
type Page struct {
	Limit  int
	Cursor string
}

func (p Page) query() url.Values {
	q := url.Values{}
	if p.Limit > 0 {
		q.Set("limit", strconv.Itoa(p.Limit))
	}
	if p.Cursor != "" {
		q.Set("cursor", p.Cursor)
	}
	return q
}

// List is a cursor-paginated response.
type List[T any] struct {
	Items      []T     `json:"items"`
	NextCursor *string `json:"next_cursor,omitempty"`
}
