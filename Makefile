## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.
LOGO_FILE=.rhiza/assets/rhiza-logo.svg

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Wire typecheck into make validate
post-validate::
	@$(MAKE) typecheck
