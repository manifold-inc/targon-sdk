# root Makefile
PACKAGES := cli libs/python libs/typescript libs/go
TARGETS  := build test lint fmt clean
PYTHON_PKG := libs/python

.DEFAULT_GOAL := help
.PHONY: help install install-dev $(TARGETS) $(PACKAGES)

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets (run across all packages):"
	@echo "  build        Build every package"
	@echo "  test         Test every package"
	@echo "  lint         Lint every package"
	@echo "  fmt          Format every package"
	@echo "  clean        Remove build artifacts"
	@echo ""
	@echo "Python-only targets:"
	@echo "  install      Editable install of the Python SDK ($(PYTHON_PKG))"
	@echo "  install-dev  Editable install with dev extras ($(PYTHON_PKG))"
	@echo ""
	@echo "Packages: $(PACKAGES)"
	@echo "Run 'make <package>' to build a single package."

$(TARGETS):
	@for pkg in $(PACKAGES); do \
		printf '\n==> make %s (%s)\n' "$@" "$$pkg"; \
		$(MAKE) -C $$pkg $@ || exit 1; \
	done

# Python-specific install targets (only the Python lib is pip-installable).
install install-dev:
	$(MAKE) -C $(PYTHON_PKG) $@

# Run any target against a single package, e.g. `make cli` runs `build` in cli.
$(PACKAGES):
	$(MAKE) -C $@ build
