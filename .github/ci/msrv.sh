#!/usr/bin/env bash
# The MSRV check: every crate and target at the workspace's rust-version.
# A script (like tests.sh) so the codec repositories' cache warm-up runs the
# same command with the same settings.
set -euo pipefail
cd "$(dirname "$0")/../.."
export CARGO_TERM_COLOR=always
export CARGO_PROFILE_DEV_DEBUG=line-tables-only
cargo check --workspace --all-targets --locked --features rivet-transcoder/server,rivet-transcoder/batch,rivet-transcoder/image
