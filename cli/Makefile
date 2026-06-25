.PHONY: build test lint fmt clean

OUT_DIR := out
BIN_NAME := targon

build:
	cargo build --release
	@mkdir -p $(OUT_DIR)
	@cp target/release/$(BIN_NAME) $(OUT_DIR)/$(BIN_NAME)
	@echo "Binary copied to $(OUT_DIR)/$(BIN_NAME)"

test:
	cargo test

lint:
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings

fmt:
	cargo fmt

clean:
	cargo clean
	rm -rf $(OUT_DIR)
