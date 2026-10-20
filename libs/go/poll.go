package targon

import (
	"context"
	"errors"
	"time"
)

// pollUntil polls until done, the context is canceled, or its deadline passes.
// A deadline produces timeoutError. A cancel produces ctx.Err().
func pollUntil[T any](
	ctx context.Context,
	interval time.Duration,
	poll func() (T, bool, error),
	timeoutError func(T) error,
) (T, error) {
	var zero T
	for {
		if err := ctx.Err(); err != nil {
			if errors.Is(err, context.DeadlineExceeded) {
				return zero, timeoutError(zero)
			}
			return zero, err
		}
		value, done, err := poll()
		if err != nil {
			if errors.Is(ctx.Err(), context.DeadlineExceeded) {
				return value, timeoutError(value)
			}
			if errors.Is(ctx.Err(), context.Canceled) {
				return zero, ctx.Err()
			}
			return value, err
		}
		if done {
			return value, nil
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			if errors.Is(ctx.Err(), context.DeadlineExceeded) {
				return value, timeoutError(value)
			}
			return zero, ctx.Err()
		case <-timer.C:
		}
	}
}
