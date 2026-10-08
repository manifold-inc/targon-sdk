package targon

import (
	"context"
	"time"
)

func pollUntil[T any](
	ctx context.Context,
	timeout, interval time.Duration,
	poll func() (T, bool, error),
	timeoutError func(T) error,
) (T, error) {
	deadline := time.Now().Add(timeout)
	for {
		value, done, err := poll()
		if err != nil || done {
			return value, err
		}
		if time.Now().After(deadline) {
			return value, timeoutError(value)
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			var zero T
			return zero, ctx.Err()
		case <-timer.C:
		}
	}
}
