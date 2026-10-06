#!/usr/bin/env bash
# rivet's Linux CI, one group per job:
#
#   .github/ci/tests.sh <group>... [--no-run]
#   .github/ci/tests.sh --list
#   .github/ci/tests.sh --affected-by <crate>   the groups that build <crate>
#
# CI runs each group in its own job, in parallel (.github/workflows/ci.yml).
# `--no-run` builds every test binary a group would run and runs nothing: the
# codec repositories call it on a push to their develop, inside a rivet
# checkout at the same path as rivet's CI, so that the shared compile cache
# (sccache, safewords/helm-charts ci-cache) already holds the codec as rivet
# compiles it when rivet's submodule is bumped. That only works if both build
# with the same settings, so every setting the build depends on is HERE, not
# in the workflow: sccache's key includes the compiler arguments and every
# CARGO_* variable.
set -euo pipefail
cd "$(dirname "$0")/../.."

# --release in CI means release without the fat LTO and the single codegen
# unit of the shipped profile: those make every test binary a whole-program
# optimisation of every codec (minutes each, on one core), and they do not
# change what a test checks. The shipped profile is still built every push
# (the gpu-build job, the binaries the GPU jobs run).
export CARGO_TERM_COLOR=always
export CARGO_PROFILE_RELEASE_LTO=false
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
# Debug and test builds: line tables for backtraces, not full debug info
# (smaller objects to compile, link and cache).
export CARGO_PROFILE_DEV_DEBUG=line-tables-only

NO_RUN=0
SEL=()
AFFECTED_BY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --no-run) NO_RUN=1 ;;
    --list) SEL=(LIST) ;;
    --affected-by) AFFECTED_BY=$2; shift ;;
    *) SEL+=("$1") ;;
  esac
  shift
done

# `t <cargo test args> [-- <test filters>]`: runs the tests, or with --no-run
# builds them (filters dropped).
t() {
  local build=() filters=() past=0 a
  for a in "$@"; do
    if [ "$past" = 1 ]; then filters+=("$a"); elif [ "$a" = -- ]; then past=1; else build+=("$a"); fi
  done
  echo "::group::cargo test ${build[*]}${filters:+ -- ${filters[*]}}"
  if [ "$NO_RUN" = 1 ]; then
    cargo test "${build[@]}" --no-run
  elif [ "${#filters[@]}" -gt 0 ]; then
    cargo test "${build[@]}" -- "${filters[@]}"
  else
    cargo test "${build[@]}"
  fi
  echo "::endgroup::"
}

# ---- the groups ---------------------------------------------------------

# rivet-codec's and rivet-container's unit tests, and the workspace's debug build.
group_codec_lib() {
  echo "::group::cargo build --workspace"
  cargo build --workspace --locked
  echo "::endgroup::"
  t -p rivet-codec --lib --features serde --locked
  # The GPU backends' pure unit tests (parameter layouts, property
  # sequences): dlopen FFI, so they build and run with no card.
  t -p rivet-codec --lib --features serde,qsv,nvidia,amd --locked
  # The pipeline kernels (scaler, colour, tonemap, SDR->HDR, denoise,
  # resampler) once more with every SIMD switch off: the tests above
  # compare each vector level with the scalar reference explicitly;
  # this runs everything that dispatches on the scalar paths too.
  RIVET_PIPE_MAX_SIMD=none RIVET_DENOISE_MAX_SIMD=none RIVET_TONEMAP_SCALAR=1 \
    t -p rivet-codec --lib --release --locked -- simd colorspace tonemap filter audio::resample
  t -p rivet-container --locked
}

# rivet-codec's integration tests: the software encoders and decoders end to
# end (odd sizes, AV1 / H.26x round trips, the native codecs in their
# containers; the GPU ones build to nothing without their features), and
# FLAC / ALAC bit-exact both ways against the flac CLI and Apple's
# alacconvert through rivet's MP4 and Matroska code (mkvmerge) — required.
group_codec_tests() {
  RIVET_REQUIRE_LOSSLESS_ORACLES=1 t -p rivet-codec --release --features serde --locked --tests
}

# rivet-transcoder's unit tests. Still images included: every format
# through rivet's own codecs, AVIF through rivet's own AV1 encoder and
# decoder, stills from a clip made by rivet's own H.264 encoder.
group_transcoder_lib() {
  t -p rivet-transcoder --lib --features server,batch,ipc,thumbnail,image,ndi --locked
  # NDI: the FFI layouts and pixel conversions, and the loopback through the
  # runtime, which says SKIP on a runner without one.
  t -p rivet-ndi --locked
  t -p rivet-transcoder --test ndi_loopback --features ndi,batch,server --locked
}

# End-to-end jobs on synthetic sources. Inputs made by rivet's own encoders
# (tests/common/synth.rs): fitted geometry, DTS input, an HLS bitrate
# ladder, every audio codec in every file it goes in (read back and decoded
# by rivet); MediaInfo reading rivet's MP4s and MKVToolNix rivet's WebMs
# (required here); every video codec's output through the job engine,
# demuxed and decoded by rivet with PSNR against the source; the AV1 PSNR /
# SSIM gate on the software encoder. And the rest of the crate's integration
# tests that need no GPU: the AV1 encode / mux / decode chain on a pattern
# (e2e, fidelity_pattern, fidelity_sanity), the binary never writing over its
# input and writing each HLS file whole (output_overwrite), the HTTP API
# (server_api). Only cbr_rates is left out: it is the Intel GPU jobs' (QSV
# rate control) and skips without the card.
group_transcoder_e2e() {
  RIVET_REQUIRE_MEDIAINFO=1 RIVET_REQUIRE_MKVTOOLNIX=1 \
    t -p rivet-transcoder --release --locked --features h26x-fallback,av1-sw-fallback,image,server,batch \
      --test fit_e2e --test dts_audio --test hls_rates --test fidelity_mediainfo --test audio_codecs_e2e \
      --test new_codecs_e2e --test fidelity_psnr_ssim --test e2e --test fidelity_pattern \
      --test fidelity_sanity --test output_overwrite --test server_api
}

# The audio codecs. AAC (crates/aac): encoder, decoder, shared tables, and
# both against faad2's decoder (a black box) — required. AC-3 / E-AC-3
# (crates/ac3): unit tests and the bundled 5.1 vector. DTS (crates/dts): round
# trips, and the encoder's streams against libdca's decoder — required. Opus,
# MPEG audio / MP3 and Vorbis: committed vectors, round trips, spec tests.
# FLAC and ALAC (crates/lossless), both ways, against the flac CLI and
# Apple's alacconvert — required. The downloaded conformance suites run in
# each crate's own CI.
group_audio() {
  AAC_REQUIRE_FAAD=1 t -p rivet-aac --release --locked
  t -p rivet-ac3 --release --locked
  DTS_REQUIRE_DCADEC=1 t -p rivet-dts --release --locked
  t -p rivet-opus -p rivet-mp3 -p rivet-vorbis --release --locked
  RIVET_REQUIRE_LOSSLESS_ORACLES=1 t -p rivet-lossless --release --locked
}

# The clean-room video codecs but AV1: H.264 / H.265 (crates/h26x), ProRes,
# VP8, VP9, MPEG-2, MPEG-4 Part 2. Committed vectors, round trips, spec
# tests; the downloaded conformance suites run in each crate's own CI.
group_video() {
  t -p rivet-h26x --release --locked
  t -p rivet-prores -p rivet-vp8 -p rivet-vp9 -p rivet-mpeg2 -p rivet-mpeg4 --release --locked
}

# AV1 (crates/av1): the committed AOM vectors bit-exact, the encoder against
# its own reconstruction, the malformed-input property tests. The full AOM
# set and the Argon suite run in its own CI.
group_av1() {
  t -p rivet-av1 --release --locked
}

# The still-image codecs: WebP, PNG (PngSuite, DEFLATE round trips), JPEG
# (its committed corpus), and GIF, BMP and TIFF (crates/imagecodecs, a
# workspace of its own with its own lock file).
group_images() {
  t -p rivet-png -p rivet-jpeg -p rivet-webp --release --locked
  t --manifest-path crates/imagecodecs/Cargo.toml --workspace --release --locked
}

ALL=(codec-lib codec-tests transcoder-lib transcoder-e2e audio video av1 images)

# The packages each group builds, for --affected-by: `*` is every workspace
# member (codec-lib's `cargo build --workspace`), `@imagecodecs` the
# imagecodecs workspace.
declare -A PKGS=(
  [codec-lib]="*"
  [codec-tests]="rivet-codec"
  [transcoder-lib]="rivet-transcoder rivet-ndi"
  [transcoder-e2e]="rivet-transcoder"
  [audio]="rivet-aac rivet-ac3 rivet-dts rivet-opus rivet-mp3 rivet-vorbis rivet-lossless"
  [video]="rivet-h26x rivet-prores rivet-vp8 rivet-vp9 rivet-mpeg2 rivet-mpeg4"
  [av1]="rivet-av1"
  [images]="rivet-png rivet-jpeg rivet-webp @imagecodecs"
)
if [ -n "$AFFECTED_BY" ]; then
  for g in "${ALL[@]}"; do
    for p in ${PKGS[$g]}; do
      case "$p" in
        "*") hit=1 ;;
        @imagecodecs) cargo tree --manifest-path crates/imagecodecs/Cargo.toml --workspace --locked -e normal,build,dev -i "$AFFECTED_BY" >/dev/null 2>&1 && hit=1 || hit=0 ;;
        *) cargo tree --locked -e normal,build,dev -p "$p" -i "$AFFECTED_BY" >/dev/null 2>&1 && hit=1 || hit=0 ;;
      esac
      if [ "$hit" = 1 ]; then echo "$g"; break; fi
    done
  done
  exit 0
fi

if [ "${SEL[0]:-}" = LIST ]; then printf '%s\n' "${ALL[@]}"; exit 0; fi
[ "${#SEL[@]}" -gt 0 ] || { echo "usage: $0 <group>... [--no-run] | all | --list" >&2; exit 2; }
[ "${SEL[0]}" = all ] && SEL=("${ALL[@]}")
for g in "${SEL[@]}"; do
  fn="group_${g//-/_}"
  declare -F "$fn" >/dev/null || { echo "unknown group: $g (groups: ${ALL[*]})" >&2; exit 2; }
  echo "== $g"
  "$fn"
done
