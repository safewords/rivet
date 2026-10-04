#!/usr/bin/env bash
# The black-box tools a test group checks rivet against, installed for that
# group's job only:
#
#   .github/ci/deps.sh <package>... [alacconvert]
#
# flac, mkvtoolnix: Xiph's reference FLAC tool and MKVToolNix, the lossless
# oracle tests compare against and mux with (they fail rather than skip
# without them: RIVET_REQUIRE_LOSSLESS_ORACLES). faad: faad2's AAC decoder,
# a black box the AAC encoder is checked against (AAC_REQUIRE_FAAD).
# libdca-utils: libdca's `dcadec`, the same for the DTS encoder
# (DTS_REQUIRE_DCADEC). mediainfo: MediaArea's MediaInfo, an independent
# reader of rivet's MP4s (RIVET_REQUIRE_MEDIAINFO). alacconvert: Apple's
# open-source ALAC release, pinned, built without reading it (exported as
# ALACCONVERT). No FFmpeg, in any form.
#
# MKVToolNix comes from its author's own repository: the distribution's (82)
# does not read FLAC in MP4, which the lossless oracle needs.
set -euo pipefail
cd "$(dirname "$0")/../.."

pkgs=()
alac=0
missing=()
for p in "$@"; do
  case "$p" in
    alacconvert) alac=1 ;;
    *) pkgs+=("$p") ;;
  esac
done

# Already on the runner (the safewords runner image carries them): nothing
# to install.
declare -A BIN=([flac]=flac [mkvtoolnix]=mkvmerge [faad]=faad [libdca-utils]=dcadec [mediainfo]=mediainfo)
missing=()
for p in "${pkgs[@]}"; do
  command -v "${BIN[$p]:-$p}" >/dev/null || missing+=("$p")
done
pkgs=("${missing[@]}")

if [ "${#pkgs[@]}" -gt 0 ]; then
  if printf '%s\n' "${pkgs[@]}" | grep -qx mkvtoolnix; then
    sudo install -d /etc/apt/keyrings
    sudo curl -fsSLo /etc/apt/keyrings/gpg-pub-moritzbunkus.gpg https://mkvtoolnix.download/gpg-pub-moritzbunkus.gpg
    echo "deb [signed-by=/etc/apt/keyrings/gpg-pub-moritzbunkus.gpg] https://mkvtoolnix.download/ubuntu/ $(lsb_release -cs) main" \
      | sudo tee /etc/apt/sources.list.d/mkvtoolnix.list >/dev/null
  fi
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${pkgs[@]}" >/dev/null
  if command -v mkvmerge >/dev/null; then mkvmerge --version; fi
fi

if [ "$alac" = 1 ]; then
  bin=$(crates/lossless/tools/build-alacconvert.sh "${RUNNER_TEMP:-/tmp}/alac")
  echo "ALACCONVERT=$bin" >> "${GITHUB_ENV:-/dev/null}"
  echo "alacconvert: $bin"
fi
