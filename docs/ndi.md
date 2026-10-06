# NDI

rivet reads and writes [NDI®](https://ndi.video), the IP video protocol of
live production: it **records an NDI source into a file**, encoding it live
with any video encoder rivet has (GPU first), and **plays a file out as an
NDI source**. Both are the opt-in `ndi` feature.

```sh
cargo build --release --features ndi,nvidia      # NDI + an NVENC encoder
rivet ndi sources                                # what is on the network
rivet ndi record "Camera 1" -o cam1.mp4 --codec h264 --duration 1h
rivet ndi send programme.mkv --name Playout --loop
```

- [How it is built](#how-it-is-built) — the `rivet-ndi` crate, the runtime
- [`rivet ndi sources`](#rivet-ndi-sources)
- [`rivet ndi record`](#rivet-ndi-record) — and how the recording stays in step
- [`rivet ndi send`](#rivet-ndi-send)
- [Library use](#library-use)
- [Testing](#testing)

## How it is built

The protocol is [`rivet-ndi`](../crates/ndi) (imported as `ndi`), a git
submodule of [safewords/rivet-ndi](https://github.com/safewords/rivet-ndi):
FFI written by hand against the NDI SDK's public C headers, loading the NDI
runtime the first time an NDI command runs. Like the GPU backends, nothing
about NDI is needed to **build** — no SDK, no bindgen, no link step — so the
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
| The platform loader | `Processing.NDI.Lib.x64.dll` on `PATH` (Windows); `libndi.so.6`, `.so.5`, `.so` (Linux: `/usr/lib`, `/usr/local/lib`, the `ld.so` cache). |

NDI 4 or later. The runtime is NDI's own and is not redistributed with rivet;
it is free to install, under NDI's licence.

## `rivet ndi sources`

```text
rivet ndi sources [--wait SECONDS] [--groups GROUPS] [--extra-ips IPS] [--json]
```

Listens for `--wait` seconds (3 by default) and lists every source seen, with
the address it was reached at. `--groups` looks in other NDI groups;
`--extra-ips` asks machines directly where mDNS does not reach (another
subnet). `--json` prints `[{"name": "...", "url": "..."}]`.

```text
$ rivet ndi sources
STUDIO-PC (Camera 1)  (10.0.1.20:5961)
STUDIO-PC (Camera 2)  (10.0.1.20:5962)
EDIT-BAY (Program)    (10.0.1.31:5961)
```

## `rivet ndi record`

```text
rivet ndi record SOURCE -o OUT [--duration D] [--frames N] [--codec C] ...
```

`SOURCE` is a name as listed, or any part of one that names only one source
(`"Camera 1"`, `Program`); a part two sources share is refused with both
names. The recording runs until `--duration` (`90`, `90s`, `15m`, `2h`,
`1h30m`), `--frames`, the source going away, no picture for
`--idle-timeout` seconds (10; `0` waits for ever), or Ctrl+C — and in every
case what was recorded is written. A second Ctrl+C abandons it.

| Flag | Default | |
|------|---------|---|
| `-o, --output` | — | `.mp4`, `.mov` or `.webm`. Written beside as `OUT.partial` while recording and renamed into place at the end, so a reader never sees half a file. |
| `--codec` | `av1` | `av1`, `h264`, `h265`, `vp9`, `vp8`, `mpeg2`, `mpeg4`, `prores[-PROFILE]`. |
| `--container` | from the extension | `mp4`, `mov`, `webm`. WebM takes VP8 / VP9 and Opus. |
| `--crf` / `--target` | the encoder's | Quality, as `rivet transcode` takes it. |
| `--gop` | `2` | Seconds between keyframes. |
| `--color` | `sdr` | `sdr` (an HDR source tonemapped to SDR BT.709), `passthrough`, `hdr10`, `hlg`. |
| `--pixel-format` | `auto` | `8bit`, `10bit`. |
| `--audio` | `opus` | `opus`, `aac`, `he-aac` (MP4 / MOV), `none`. |
| `--audio-bitrate` | the codec's | e.g. `160k`. |
| `--filter` | — | Video filters, as `rivet transcode --filter`. |
| `--high-bit-depth` | off | Ask for the source's 10-bit stream (P216) when it sends one; with `--pixel-format 10bit` (or an HDR `--color`) the bits are kept. |
| `--low-bandwidth` | off | Ask for the sender's proxy stream. |
| `--wait` | `15` | Seconds for the source to appear and send a picture. |
| `--encoder`, `--gpu` | the GPU-first chain | Force `nvenc`, `amf`, `qsv`, `h26x` or `av1`; pin a GPU. |
| `--groups`, `--extra-ips` | — | As for `sources`. |
| `--json` | off | Print the outcome as JSON. |

```text
$ rivet ndi record "Camera 1" -o cam1.mp4 --codec h264 --duration 10s
recording STUDIO-PC (Camera 1) → cam1.mp4 (Ctrl+C to stop)
      9.5 s      285 frames   30.0 fps  0 repeated  0 dropped
cam1.mp4: 1920x1080 @ 30000/1001 h264, audio opus 2ch 48000 Hz, 300 frames (10.0 s; 0 repeated, 0 dropped) — limit reached
```

### What a recording is made of

The pictures go through the same per-frame work as every rivet job (the
decode pump's `FrameNormalizer`): NDI's 4:2:2 UYVY (or P216 at 16 bits, or
RGBA) is brought to 4:2:0, re-matrixed, tonemapped or mapped into HDR as
`--color` says, set to the encoder's depth, and filtered. A source that
changes size mid-stream is scaled to the size the recording started at.

**Colour.** A frame carrying NDI 6's `<ndi_color_info transfer="…"
matrix="…" primaries="…"/>` metadata is taken at its word (so an HLG or PQ
source is HDR); otherwise NDI's convention holds: studio-range BT.601 below
720 lines, BT.709 from 720 up, and RGB as full-range sRGB.

**Audio** arrives as 32-bit float and is encoded as it comes, in Opus (or
AAC), up to 7.1; a source with more than eight channels is recorded as its
first two.

### Keeping in step

A file has a fixed frame rate and a gapless audio track; a live source has
neither. The recording runs on the source's timestamps (the sender's clock,
which NDI carries with every frame; this machine's clock for a sender that
stamps none):

- Each picture is placed on the frame its timestamp falls on, counted from
  the first picture, at the frame rate the source declares. A picture whose
  frame is already written is dropped; a gap — frames the source skipped, or
  pictures dropped because the encoder fell behind — is filled by repeating
  the last picture. So the file is as long as the time recorded, and
  `repeated` / `dropped` in the progress line say how it was kept so.
- Audio is placed the same way: a gap of more than 40 ms is filled with
  silence, an overlap trimmed, smaller jitter absorbed. Audio that started
  after the first picture is padded at the front; audio from before it is
  cut. At the end the audio track is cut to the pictures' length.
- A jump of more than ten seconds either way (a sender restarting, a clock
  reset) re-anchors the timeline rather than filling ten seconds.

Pictures are received on a thread of their own and queued, about a second's
worth. When the encoder cannot keep up, pictures beyond that are dropped
(and their frames repeated); audio is never dropped. A recording that
reports many `dropped` has an encoder slower than the source: a GPU encoder
(`--features nvidia|amd|qsv`), `--codec h264`, or a faster `--target` keeps
up.

## `rivet ndi send`

```text
rivet ndi send INPUT [--name NAME] [--loop] [--ten-bit] [--no-audio] [--groups GROUPS]
```

Announces `MACHINE (NAME)` (NAME defaults to the file's name) and plays the
file into it: demuxed and decoded as any rivet input is (hardware decoders
first), each picture normalised, and sent paced at the file's frame rate (the
NTSC rates as `30000/1001` and friends). Pictures go out as 8-bit I420, SDR
BT.709 (an HDR file tonemapped); `--ten-bit` sends 16-bit P216 in the file's
own colour. The audio is decoded alongside and each picture's audio is sent
just before it. `--loop` starts again at the end until Ctrl+C.

## Library use

```rust
use std::time::Duration;
use rivet::ndi::{NdiSource, RecordOptions, record};

let source = NdiSource::connect(
    "Camera 1",
    &ndi::FindOptions::default(),
    &ndi::ReceiverOptions::default(),
    Duration::from_secs(15),
)?;
let mut options = RecordOptions::new("cam1.mp4");
options.video_codec = rivet::VideoCodecPolicy::H264;
options.duration = Some(Duration::from_secs(60));
let outcome = record(source, &options, |p| eprintln!("{:.1} s", p.seconds))?;
# Ok::<(), anyhow::Error>(())
```

`record` takes any `rivet::ndi::LiveSource` — a trait of one method,
`next_event`, yielding pictures and audio with timestamps — so another live
input (a capture card, a test pattern) records through the same timeline,
normalisation and muxing. `rivet::ndi::send_file` is the send side;
`rivet::ndi::list_sources` the discovery. The `ndi` crate itself (re-exported
as `rivet::ndi_sys`) is usable on its own: `Ndi::load`, `finder`, `receiver`
(frames as planar `Picture`s), `sender`.

## Testing

- `cargo test -p rivet-ndi` — the struct layouts against the SDK headers'
  offsets, every fourcc conversion both ways, source-name matching. No
  runtime needed.
- `cargo test -p rivet-transcoder --features ndi --lib ndi` — the timeline
  (frame slots, drop / repeat / re-anchor), colour inference, cropping, and
  two recordings of a synthetic live source (a skipped frame repeated, late
  audio padded, limits, WebM).
- `cargo test -p rivet-transcoder --features ndi --test ndi_loopback` —
  through the real runtime: a sender on this machine sends pictures and a
  440 Hz tone, the recorder finds it by name and records three seconds, and
  the file's length, frame count and audio (the tone throughout, no dropout)
  are checked. Without a runtime it says SKIP; `RIVET_REQUIRE_NDI=1` makes
  that a failure.

NDI® is a registered trademark of Vizrt NDI AB. rivet is not affiliated
with or endorsed by Vizrt.
