# rivet — the everyday commands in one place. Everything here is a thin
# wrapper over cargo, git and the CI scripts in .github/ci, so a target does
# exactly what CI does where CI does it.
#
#   make                    # fetch the submodules if missing, build release
#   make FEATURES=ndi,qsv   # ... with features
#   make help               # every target
#
# Needs git, a Rust toolchain (1.99 or newer) and GNU make; on Windows, run
# it from Git Bash or MSYS2.

CARGO    ?= cargo
FEATURES ?=
PROFILE  ?= release
ARGS     ?=
# Parallel submodule fetches.
JOBS     ?= 16

# The features every lint and doc build covers: everything that builds
# without hardware or a CUDA toolkit (the GPU and NDI backends are dlopened
# at run time, so they compile anywhere). The same list as CI's clippy job.
ALL_FEATURES := server,batch,image,thumbnail,ipc,ndi,qsv,nvidia,amd,av1-sw-fallback,h26x-fallback
# The library tests' features (CI's transcoder-lib group).
TEST_FEATURES := server,batch,ipc,thumbnail,image,ndi

# A literal comma, for the substitutions below.
, := ,
FEATURE_FLAGS := $(if $(FEATURES),--features $(FEATURES),)
EXE := $(if $(filter Windows_NT,$(OS)),.exe,)
PROFILE_DIR := $(if $(filter dev,$(PROFILE)),debug,$(PROFILE))
BIN := target/$(PROFILE_DIR)/rivet$(EXE)

.DEFAULT_GOAL := all
.PHONY: all help submodules submodules-update submodules-status build debug \
        release-fast install run ndi test test-ndi test-all msrv fmt fmt-check \
        clippy lint doc check clean distclean

all: submodules build ## Fetch missing submodules, then a release build.

help: ## List the targets.
	@echo "make [target] [FEATURES=a,b] [PROFILE=release|release-fast|dev] [ARGS=...]"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

# ── Submodules ────────────────────────────────────────────────────────
# The codec crates (and rivet-ndi) are git submodules: a clone without
# --recurse-submodules has empty directories under crates/ and cannot build.

submodules: ## Fetch every submodule at the commit rivet pins.
	git submodule sync --recursive
	git submodule update --init --recursive --jobs $(JOBS)

submodules-update: ## Move every submodule to the tip of its tracked branch (develop).
	git submodule update --init --recursive --remote --jobs $(JOBS)
	@echo
	@echo "Submodules moved; review with 'git diff --submodule' and commit the new pointers."

submodules-status: ## Show each submodule's commit and whether it moved.
	git submodule status --recursive

# ── Build ─────────────────────────────────────────────────────────────

build: ## Build the rivet CLI and library (PROFILE=..., FEATURES=...).
	$(CARGO) build --profile $(PROFILE) -p rivet-transcoder $(FEATURE_FLAGS)
	@echo "built $(BIN)"

debug: ## A debug build.
	$(MAKE) build PROFILE=dev

release-fast: ## A release-class build without fat LTO (faster to link).
	$(MAKE) build PROFILE=release-fast

ndi: ## A release build with NDI (and FEATURES on top).
	$(MAKE) build FEATURES=$(if $(FEATURES),$(FEATURES)$(,),)ndi


install: submodules ## cargo install the `rivet` binary (FEATURES=...).
	$(CARGO) install --locked --path crates/rivet $(FEATURE_FLAGS)

run: build ## Build, then run rivet with ARGS (make run ARGS="probe in.mp4").
	./$(BIN) $(ARGS)

# ── Test and lint (as CI runs them) ───────────────────────────────────

test: ## The transcoder's unit tests, every hardware-free feature.
	$(CARGO) test -p rivet-transcoder --lib --features $(TEST_FEATURES) --locked

test-ndi: ## Live and NDI tests: rivet-ndi, the live path, and NDI through the runtime (SKIPs without one).
	$(CARGO) test -p rivet-ndi --locked
	$(CARGO) test -p rivet-transcoder --features ndi,batch,server --lib --locked live
	$(CARGO) test -p rivet-transcoder --features ndi,batch,server --test ndi_loopback --locked

test-all: ## Every CI test group (.github/ci/tests.sh; slow, needs CI's tools).
	.github/ci/tests.sh

msrv: ## Check every crate and target at the minimum Rust version.
	.github/ci/msrv.sh

fmt: ## Format the workspace.
	$(CARGO) fmt --all

fmt-check: ## Check formatting, as CI does.
	$(CARGO) fmt --all --check

clippy: ## Clippy over the workspace with every hardware-free feature, -D warnings.
	$(CARGO) clippy --workspace --all-targets --locked \
		--features $(subst $(,),$(,)rivet-transcoder/,rivet-transcoder/$(ALL_FEATURES)) \
		-- -D warnings

lint: fmt-check clippy ## fmt-check and clippy.

check: lint test ## lint and the unit tests: the gate before a pull request.

doc: ## Build the API docs (every hardware-free feature).
	$(CARGO) doc --no-deps -p rivet-transcoder -p rivet-codec -p rivet-container \
		-p rivet-frame -p rivet-ndi --features rivet-transcoder/ndi,rivet-transcoder/image

# ── Clean ─────────────────────────────────────────────────────────────

clean: ## Remove build outputs.
	$(CARGO) clean

distclean: clean ## Also empty the submodule checkouts (make submodules restores them).
	git submodule deinit --all --force
