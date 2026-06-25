# root Makefile
PACKAGES := cli libs/python libs/typescript libs/go
TARGETS  := build test lint fmt clean

.DEFAULT_GOAL := help
.PHONY: help $(TARGETS) $(PACKAGES)

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets (run across all packages):"
	@echo "  build    Build every package"
	@echo "  test     Test every package"
	@echo "  lint     Lint every package"
	@echo "  fmt      Format every package"
	@echo "  clean    Remove build artifacts"
	@echo ""
	@echo "Packages: $(PACKAGES)"
	@echo "Run 'make <package>' to build a single package."

$(TARGETS):
	@for pkg in $(PACKAGES); do \
		printf '\n==> make %s (%s)\n' "$@" "$$pkg"; \
		$(MAKE) -C $$pkg $@ || exit 1; \
	done

# Run any target against a single package, e.g. `make cli` runs `build` in cli.
$(PACKAGES):
	$(MAKE) -C $@ build
