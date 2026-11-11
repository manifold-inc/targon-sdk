module github.com/manifold-inc/targon-sdk/examples/go

go 1.24

require github.com/manifold-inc/targon-sdk/libs/go/v4 v4.0.0-rc.1

require (
	github.com/coder/websocket v1.8.15 // indirect
	github.com/pelletier/go-toml/v2 v2.4.3 // indirect
)

replace github.com/manifold-inc/targon-sdk/libs/go/v4 => ../../libs/go
