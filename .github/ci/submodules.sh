#!/usr/bin/env bash
# The codec submodules, all fetched at once (actions/checkout fetches them
# one after another: 18 round trips to GitHub, one to three minutes a job).
# Shallow, at the pinned commits. Public repositories: no credentials.
set -euo pipefail
cd "$(dirname "$0")/../.."
for attempt in 1 2 3; do
  if git -c protocol.version=2 submodule update --init --recursive --depth 1 --jobs 32; then exit 0; fi
  echo "submodule fetch failed (attempt $attempt); retrying" >&2
  sleep $((attempt * 5))
done
exit 1
