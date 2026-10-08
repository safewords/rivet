# NDI and live jobs

rivet reads and writes [NDI®](https://ndi.video), the IP video protocol of
live production. An NDI source is an **input** and an NDI stream an
**output** — by URI, `ndi://NAME`, wherever a path goes — and a job with
either is the same job any file gets: the same settings, the same
[`OutputSpec`](output-spec.md), the same rungs, ladders, codecs, quality,
colour, filters, audio policy and encode plan, from every front end (CLI,
[batch manifest](batch.md), [HTTP API](api.md)). Only the run differs: the
job engine's **live path** runs it in real time.

```sh
cargo build --release --features ndi,nvidia         # NDI + an NVENC encoder

rivet ndi sources                                   # what is on the network
rivet probe "ndi://STUDIO (Camera 1)"               # what a source sends

# Record: any output a file job makes, from a live source.
rivet transcode "ndi://Camera 1" -o cam1.mp4 --codec h264 --duration 1h
rivet transcode "ndi://Camera 1" -o cam1/ --rung 1920x1080 --rung 1280x720 --codec h264
rivet transcode "ndi://Camera 1" --mode hls --ladder --codec h264 -o live/   # live HLS

# Play out: a file (or a live source) as an NDI stream.
rivet transcode programme.mkv -o ndi://Playout --loop
rivet transcode "ndi://Camera 1" -o ndi://Camera1-720 --rung 1280x720 --filter hflip
```

- [How it is built](#how-it-is-built) — the `rivet-ndi` crate, the runtime
- [`ndi://` URIs](#ndi-uris)
- [What a live job does with the spec](#what-a-live-job-does-with-the-spec)
- [Encode devices](#encode-devices) — and several sources at once
- [Keeping in step](#keeping-in-step)
- [Live HLS](#live-hls)
- [Ending a live job](#ending-a-live-job) — `duration`, Ctrl+C, the API's stop
- [Front ends](#front-ends): CLI, batch manifest, HTTP API, library
- [Testing](#testing)

## How it is built

The protocol is [`rivet-ndi`](../crates/ndi) (imported as `ndi`), a git
submodule of [safewords/rivet-ndi](https://github.com/safewords/rivet-ndi):
FFI written by hand against the NDI SDK's public C headers, loading the NDI
runtime the first time an NDI job runs. Like the GPU backends, nothing about
NDI is needed to **build** — no SDK, no bindgen, no link step — so the `ndi`
feature builds anywhere, and a host without the runtime is told how to get
it:

```text
error: no NDI runtime found: install the NDI runtime (https://ndi.video/tools/ — NDI Tools, or
the stand-alone runtime), or point RIVET_NDI_LIB at the library (Processing.NDI.Lib.x64.dll). …
```

Where the runtime is looked for, in order:

| | |
|---|---|
| `RIVET_NDI_LIB` | A full path to the library. |
| `NDI_RUNTIME_DIR_V6`, `_V5`, `_V4` | The directories the NDI installers set. |
| Windows defaults | `C:\Program Files\NDI\NDI 6 Runtime\v6` and the NDI Tools / NDI 5 runtime directories, for a shell started before the installer set the variable. |
| The platform loader | `Processing.NDI.Lib.x64.dll` on `PATH` (Windows); `libndi.so.6`, `.so.5`, `.so` (Linux). |

NDI 4 or later. The runtime is NDI's own and is not redistributed with rivet;
it is free to install, under NDI's licence.

The live path itself (`rivet::live`, `rivet::job::run_live_job`) is in every
build; only its NDI ends need the feature. A build without it refuses an
`ndi://` URI by name.

## `ndi://` URIs

```text
ndi://NAME[?groups=G&extra-ips=IPS&bandwidth=highest|lowest&high-bit-depth]
```

| | In (a source) | Out (a stream) |
|---|---|---|
| `NAME` | The source's full name (`STUDIO (Camera 1)`), or any part of it that names one source; a part two sources share is refused with both names. `%XX` escapes are decoded. | The stream name announced; receivers see `MACHINE (NAME)`. With several rungs, one stream each: `NAME (LABEL)`. |
| `groups` | NDI groups to look in. | NDI groups to announce in. |
| `extra-ips` | Machines to ask directly, where mDNS does not reach. | — |
| `bandwidth` | `highest` (default) or `lowest` (the sender's proxy stream). | — |
| `high-bit-depth` | Ask for the 16-bit stream (P216) when the source sends one; with `bit-depth=10bit` (or an HDR `color`) the bits are kept. | — |

The receive options are the endpoint's, not the job's, which is why they ride
on the URI: the same `TranscodeSettings` describe the output whatever the
input.

## What a live job does with the spec

Everything a file job does with it, where it means something live:

| Setting | Live |
|---|---|
| `mode` | `single` (one file per rung, or one NDI stream per rung) or `hls` (a package written as it records). `audio` is refused. |
| `rungs`, `ladder`, `fit`, `orientation`, `upscale`, `width` / `height`, `max-short-side` | As for a file: fitted to the source's first picture. |
| `codec`, `container`, `prores-profile` | As for a file (CMAF for HLS). |
| `crf`, `target`, `video-bitrate`, `rate-mode`, `video-buffer`, `video-speed`, `gop`, `encode-policy` | Per rung, as for a file. HLS: every segment opens on a keyframe. |
| `color`, `bit-depth`, `chroma-downsample`, `filter` | The decode pump's per-frame work, as for a file — NDI's 4:2:2 brought to 4:2:0, re-matrixed, tonemapped or mapped into HDR, filtered. |
| `max-fps` | The output frame rate; pictures beyond it are dropped. |
| `audio`, `audio-bitrate`, `audio-channels`, `audio-filter`, `audio-quality`, `audio-bit-depth`, `flac-compression` | NDI audio is 32-bit float, so it is always encoded: `auto` is Opus (MP3 into an `.mp3`, as for a file), and any codec the container takes. `drop` writes the video alone. |
| `audio-stereo-fallback` | HLS: a stereo rendition beside a surround one. |
| `encode` (`all`, `per-rung`, `single`, `gpu:N`, `family:VENDOR`) | Where the rungs encode — see [Encode devices](#encode-devices). |
| `decode` | Refused for a live source: it arrives decoded (the NDI runtime decodes NDI). A file played out decodes as a file does. |
| `trim-start` / `trim-end` | Refused: a live job runs for `duration`. |
| `metadata-keep`, `input-fps`, `subtitles` | Refused / nothing to carry: a live source has none. |
| hooks | Probe (on the first picture), decoded-frame and encoder-frame (every picture), artifact, completed and failed hooks run as on any job. |
| `duration`, `start-timeout`, `idle-timeout`, `loop` | A live job's own keys — see [Ending a live job](#ending-a-live-job). A file-to-file job refuses them. |

**Colour.** A frame carrying NDI 6's `<ndi_color_info transfer="…"
matrix="…" primaries="…"/>` is taken at its word (so an HLG or PQ source is
HDR); otherwise NDI's convention: studio-range BT.601 below 720 lines,
BT.709 from 720 up, RGB as full-range sRGB. A source that changes size
mid-stream is scaled to the size the job started at.

## Encode devices

The spec's encode plan places each rung's encoder, judged against the cards
this build can encode the output on — the same pool a file job's plan makes:

- `all` / `per-rung` (the default) / `family:VENDOR`: the rungs go round the
  capable cards in turn — and round them **across every live job in the
  process**, so several sources at once spread over the cards rather than
  all starting on the first.
- `single`, `gpu:N`: every rung on the one card.
- A host with no capable card for the codec, a bitrate rung (coded in
  software), or `TRANSCODE_ENCODER_BACKEND`: the serial target, as a file job.
- A plan that leaves nothing to encode on is refused by name before anything
  runs.

Measured on one machine (RTX 3090 + Ryzen iGPU), `--ladder --codec h264` from
a 720p30 source: 720p and 360p on NVENC, 480p and 240p on AMF, every rung
real time.

**Several sources at once.** Each source is its own job: several
`rivet transcode` processes, several entries in a [batch manifest](batch.md)
(live entries all start together), or several `POST /v1/transcode` requests
to one server. They share the cards as above.

## Keeping in step

A file has a fixed frame rate and a gapless audio track; a live source has
neither. A live job runs on the source's timestamps (the sender's clock,
which NDI carries with every frame; this machine's clock for a sender that
stamps none):

- Each picture is placed on the frame its timestamp falls on, counted from
  the first picture, at the output rate. A picture whose frame is already
  written is dropped; a gap — frames the source skipped, or pictures dropped
  because the encoders fell behind — is filled by repeating the last picture.
  So the output is as long as the time recorded.
- Audio is placed the same way: a gap of more than 40 ms is filled with
  silence, an overlap trimmed, smaller jitter absorbed. Audio that started
  after the first picture is padded at the front, audio from before it cut,
  and the track ends with the pictures.
- A jump of more than ten seconds either way (a sender restarting, a clock
  reset) re-anchors the timeline rather than filling ten seconds.

Pictures are received on a thread of their own and queued, about a second's
worth. When the encoders cannot keep up, pictures beyond that are dropped
(their frames repeated); audio is never dropped. A file played out (`-o
ndi://`) is not real time: it is read as fast as the destination takes it
(the NDI sender paces it at the file's frame rate), so nothing is dropped.

Every live job reports what it did (`JobOutput::live`, the summary line, the
API's `live` object): frames written, repeated, dropped early (a source faster
than the output rate), dropped while the encoders were behind, and why it
ended.

## Live HLS

`--mode hls` from a live source writes the package as it records: each
rendition's segments as they are encoded, and its media playlist rewritten
after every segment as an `EVENT` playlist (`#EXT-X-PLAYLIST-TYPE:EVENT`, no
`#EXT-X-ENDLIST`), so a player can start watching while it runs. The master
playlist appears once every rendition has its first segment. When the job
ends, the finished package — `VOD` playlists, measured bandwidths, the same
files a file job writes — is written over it.

## Ending a live job

| | |
|---|---|
| `duration` | Stop after this much output: `90`, `90s`, `15m`, `2h`, `1h30m`, `500ms`. Exact: the output is that many frames. |
| the source going away | The job ends; what was made is kept. |
| `idle-timeout` | End when no picture comes for this long (default `10s`; `0` waits for ever). |
| `start-timeout` | How long to wait for the source to appear and send its first picture (default `15s`). |
| stop | Ctrl+C (CLI, `rivet batch`), `POST /v1/jobs/{id}/stop` (API), or the caller's flag (library). A second Ctrl+C abandons the job. |
| `loop` | A file played out: start again at its end, until `duration` or stopped. |

In every case the output is finished properly: files are written whole
(spooled beside the target as `NAME.partial` and renamed into place, an MP4
straight from the muxer's spool files, never held in memory), and an HLS
package gets its final playlists.

## Front ends

### CLI

`rivet transcode` takes an `ndi://` input or `-o ndi://…` with every flag it
has, plus `--duration`, `--start-timeout`, `--idle-timeout` and `--loop`
([cli.md](cli.md#rivet-transcode)). Without `-o`, a live input is written
beside you, named after the source (`STUDIO_Camera_1.h264.mp4`,
`STUDIO_Camera_1.hls/`). `rivet probe ndi://…` describes a source;
`rivet ndi sources [--wait S] [--groups G] [--extra-ips IPS] [--json]` lists
them.

```text
$ rivet transcode "ndi://Camera 1" -o ladder/ --codec h264 --rung 1280x720 --rung 640x360 --duration 6s
ndi://Camera 1 (1280x720 @ 30.000 fps)
  audio: live pcm 2ch → opus [codecs=opus]
  720p   1280x720  180 frames  0.21 MiB  [ladder/720p.mp4]
  360p   640x360  180 frames  0.15 MiB  [ladder/360p.mp4]
  6.0 s at 30/1 fps: 0 repeated, 0 dropped (0 while the encoders were behind) — duration reached
```

### Batch manifest

`input:` and `output:` take `ndi://` URIs, and `duration`, `start_timeout`,
`idle_timeout` and `loop` are job keys ([batch.md](batch.md)). Live jobs start
together and run side by side while the file jobs go one by one beside them;
Ctrl+C ends them all, each writing what it made.

```yaml
defaults:
  codec: h264
  duration: 2h
jobs:
  - input: "ndi://STUDIO (Camera 1)"
    output: rec/cam1.mp4
  - input: "ndi://STUDIO (Camera 2)"
    output: rec/cam2/
    mode: hls
    ladder: true
  - input: promo.mp4
    output: ndi://Promo
    loop: true
```

### HTTP API

A JSON `POST /v1/transcode` whose `input.path` or `output.path` is `ndi://…`
is a live job ([api.md](api.md)). It returns `202` with the job id and its
stop URL at once; `GET /v1/jobs/{id}` reports it `running` with per-rung
progress, and, once ended, its `live` account; `POST /v1/jobs/{id}/stop` ends
it. A live input needs `output.path` (a server file or directory, or another
`ndi://`). `sync=true` with a live input needs `spec.duration`.

```sh
curl -X POST localhost:8080/v1/transcode -H 'Content-Type: application/json' \
  -d '{"input":{"path":"ndi://Camera 1"},"output":{"path":"/rec/cam1.mp4"},
       "spec":{"codec":"h264","rungs":["1280x720"],"duration":"1h"}}'
# {"job_id":"…","status":"queued","stop":"/v1/jobs/…/stop"}
curl -X POST localhost:8080/v1/jobs/<id>/stop
```

### Library

```rust
use std::time::Duration;

let wait = Duration::from_secs(15);
let source = rivet::live::open_source("ndi://STUDIO (Camera 1)", wait)?;
let (info, source) = rivet::live::probe_source(source, wait)?;
let mut settings = rivet::TranscodeSettings::default();
settings.apply_kv("codec", "h264")?;
settings.apply_kv("duration", "1h")?;
let spec = settings.into_spec_for(&info)?;
let out = rivet::run_live_job_blocking(
    source,
    &spec,
    rivet::LiveTarget::File("cam1.mp4".into()),
    std::sync::Arc::new(rivet::progress::NullSink),
    None, // or Some(stop flag)
)?;
println!("{:?}", out.live);
# Ok::<(), anyhow::Error>(())
```

`run_live_job` takes any `rivet::live::LiveSource` — one method,
`next_event`, yielding pictures and audio with timestamps — so another live
input (a capture card, a test pattern) runs through the same engine;
`rivet::live::FileSource` is a file read as one (what a file played out is).
The `ndi` crate (re-exported as `rivet::ndi_sys`) is usable on its own.

## Testing

- `cargo test -p rivet-ndi` — struct layouts against the SDK headers' offsets,
  every fourcc conversion both ways, source-name matching. No runtime.
- `cargo test -p rivet-transcoder --lib live` — the timeline (slots, drop /
  repeat / re-anchor, the `max-fps` rate), URIs, and live jobs on synthetic
  sources: a gap repeated and late audio padded, a ladder with `duration`, a
  live HLS package, a file through the live path with `loop`, and the
  refusals.
- `cargo test -p rivet-transcoder --features ndi,batch,server --test
  ndi_loopback` — through the real runtime: a source recorded through the
  spec (length, frame count, the tone throughout with no dropout), a file
  played out and received, two sources recorded at once from a manifest, and
  an API live job stopped by `POST /v1/jobs/{id}/stop`. Without a runtime each
  says SKIP; `RIVET_REQUIRE_NDI=1` makes that a failure.
- `make test-ndi` runs all three.

NDI® is a registered trademark of Vizrt NDI AB. rivet is not affiliated
with or endorsed by Vizrt.
