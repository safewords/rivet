//! Subcommand implementations for the `rivet` CLI.

pub mod capabilities;
pub mod devices;
pub mod live;
pub mod pipe;
pub mod probe;
pub mod progress;
pub mod splice;
pub mod transcode;

#[cfg(feature = "batch")]
pub mod batch;
#[cfg(feature = "image")]
pub mod image;
#[cfg(feature = "ipc")]
pub mod ipc;
#[cfg(feature = "ndi")]
pub mod ndi;
#[cfg(feature = "server")]
pub mod serve;

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use rivet::progress::RungProgress;
use rivet::{RungArtifact, TranscodeSettings};

use crate::{ChromaArg, ColorArg, PixelArg, value_name};

/// How each rung's `WxH` box is filled — `--fit`, `--orientation`,
/// `--upscale` — for the commands that take a size (`transcode`, `pipe`).
/// Placed in the settings under the keys every surface uses.
#[derive(clap::Args, Debug, Default, Clone)]
pub(crate) struct FitArgs {
    /// How the source meets each rung's box, which is a maximum, not the
    /// output size: `contain` (default: inside the box, keeping the source's
    /// shape), `cover` (fill the box, centre-cropping the overflow), `pad`
    /// (contain, then black bars to exactly the box) or `stretch` (exactly
    /// the box, distorting the picture). A rung's own `:FIT` wins
    /// (`--rung 1080x1920:cover:fixed`).
    #[arg(long, value_name = "FIT")]
    pub fit: Option<String>,
    /// `auto` (default): a box turns to the source's orientation, so a
    /// 1920x1080 rung on a portrait source is 1080x1920. `fixed`: boxes are
    /// used as written.
    #[arg(long, value_name = "auto|fixed")]
    pub orientation: Option<String>,
    /// Let a rung be larger than the source. Off by default: a source smaller
    /// than a box comes out at its own size, and rungs that collapse onto the
    /// same size are merged.
    #[arg(long)]
    pub upscale: bool,
}

impl FitArgs {
    pub(crate) fn apply(&self, settings: &mut TranscodeSettings) -> Result<()> {
        if let Some(f) = &self.fit {
            settings.apply_kv("fit", f).context("parsing --fit")?;
        }
        if let Some(o) = &self.orientation {
            settings
                .apply_kv("orientation", o)
                .context("parsing --orientation")?;
        }
        settings.upscale = self.upscale;
        Ok(())
    }
}

/// A live job's flags — `--duration`, `--start-timeout`, `--idle-timeout`,
/// `--loop` — for a job whose input or output is `ndi://…`. Placed in the
/// settings under the keys every surface uses (`duration`, `start-timeout`,
/// `idle-timeout`, `loop`); a file-to-file job refuses them.
#[derive(clap::Args, Debug, Default, Clone)]
pub(crate) struct LiveArgs {
    /// A live job: stop after this long — `90`, `90s`, `15m`, `2h`, `1h30m`.
    /// Without it the job runs until the source goes away or Ctrl+C.
    #[arg(long, value_name = "DURATION")]
    pub duration: Option<String>,
    /// A live input: how long to wait for the source to appear and send a
    /// picture (default 15s).
    #[arg(long = "start-timeout", value_name = "DURATION")]
    pub start_timeout: Option<String>,
    /// A live input: end when no picture arrives for this long (default
    /// 10s; `0` waits for ever).
    #[arg(long = "idle-timeout", value_name = "DURATION")]
    pub idle_timeout: Option<String>,
    /// A file played out live (`-o ndi://NAME`): start again at its end,
    /// until Ctrl+C or `--duration`.
    #[arg(long = "loop")]
    pub repeat: bool,
}

impl LiveArgs {
    pub(crate) fn apply(&self, settings: &mut TranscodeSettings) -> Result<()> {
        for (key, value) in [
            ("duration", &self.duration),
            ("start-timeout", &self.start_timeout),
            ("idle-timeout", &self.idle_timeout),
        ] {
            if let Some(v) = value {
                settings
                    .apply_kv(key, v)
                    .with_context(|| format!("parsing --{key}"))?;
            }
        }
        if self.repeat {
            settings.apply_kv("loop", "true")?;
        }
        Ok(())
    }
}

/// A flag Ctrl+C sets, for the jobs that run until stopped (a live input).
/// A second Ctrl+C exits at once.
pub(crate) fn ctrl_c_flag() -> Arc<std::sync::atomic::AtomicBool> {
    use std::sync::atomic::{AtomicBool, Ordering};
    let flag = Arc::new(AtomicBool::new(false));
    let set = Arc::clone(&flag);
    std::thread::spawn(move || {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        else {
            return;
        };
        rt.block_on(async {
            if tokio::signal::ctrl_c().await.is_ok() {
                set.store(true, Ordering::Relaxed);
                eprintln!(
                    "
stopping: finishing the output (Ctrl+C again to abandon it)…"
                );
                if tokio::signal::ctrl_c().await.is_ok() {
                    std::process::exit(130);
                }
            }
        });
    });
    flag
}

/// The file a single-file output is — `--container`, `--prores-profile` —
/// for `transcode` and `splice`. Placed in the settings under the keys every
/// surface uses (`container`, `prores-profile`).
#[derive(clap::Args, Debug, Default, Clone)]
pub(crate) struct FileArgs {
    /// The file a single-file output is: `mp4`, `mov` (a QuickTime movie) or
    /// `webm`. Default: the codec's own — `mov` for ProRes, `webm` for VP8 /
    /// VP9, `mp4` otherwise.
    #[arg(long, value_name = "mp4|mov|webm")]
    pub container: Option<String>,
    /// The ProRes profile with `--codec prores`: `proxy`, `lt`, `422`
    /// (default), `hq`, `4444`, `4444xq`. `--codec prores-hq` says the same.
    #[arg(long = "prores-profile", value_name = "PROFILE")]
    pub prores_profile: Option<String>,
}

impl FileArgs {
    pub(crate) fn apply(&self, settings: &mut TranscodeSettings) -> Result<()> {
        if let Some(c) = &self.container {
            settings
                .apply_kv("container", c)
                .context("parsing --container")?;
        }
        if let Some(p) = &self.prores_profile {
            settings
                .apply_kv("prores-profile", p)
                .context("parsing --prores-profile")?;
        }
        Ok(())
    }
}

/// The output-shaping flags `rivet transcode` and `rivet splice` share, as
/// clap parsed them: what the output looks like (colour, depth, chroma
/// filter, video filters, quality, GOP) and how transcoded audio is made.
/// [`OutputShaping::apply`] places them in [`TranscodeSettings`] the way every
/// surface does — typed values directly, worded ones through `apply_kv` under
/// the keys the IPC socket, the HTTP API and the batch manifest use — so both
/// commands reach the same spec builder and the same validation.
#[derive(clap::Args)]
pub(crate) struct OutputShaping {
    /// Perceptual quality target: `visually_lossless`, `high`, `standard`
    /// (default), `low`, or `vmaf=N` — see `rivet transcode --help`.
    #[arg(long, value_parser = rivet::settings::parse_quality_target)]
    pub target: Option<rivet::codec::encode::tuning::QualityTarget>,
    /// GOP length: frames (`48`) or seconds of output (`2s`, `1.5s`);
    /// default two seconds at the output frame rate, which `2s` states.
    #[arg(
        long,
        visible_alias = "keyframe-interval",
        value_name = "FRAMES|SECONDSs"
    )]
    pub gop: Option<String>,
    /// Video bitrate, e.g. `3M`: code every rung without its own
    /// (`--rung WxH@RATE`) to a rate rather than to `--target`. An average
    /// rate (the default `--rate-mode`) is coded by the native software
    /// H.264 / H.265 encoder, and a job whose encode pool is GPUs is refused
    /// before a frame is decoded; a constant one (`--rate-mode cbr`) is coded
    /// by the GPU encoders and the software H.264 / H.265 encoder.
    #[arg(long = "video-bitrate", value_name = "BPS")]
    pub video_bitrate: Option<String>,
    /// Coded picture buffer for every bitrate rung, e.g. `500ms` (`0` for
    /// none; one second when not given): the stream declares it and keeps to
    /// it, which is what bounds its peaks (and an HLS rendition's BANDWIDTH).
    #[arg(long = "video-buffer", value_name = "DURATION")]
    pub video_buffer: Option<String>,
    /// Rate mode for every bitrate rung: `average` (default; `abr`) or `cbr`
    /// (`constant`) — a constant rate, the rate also the maximum within the
    /// declared buffer (`--video-buffer`, one second by default), coded by
    /// the GPU encoders (QSV, NVENC, AMF; AV1 included) and the software
    /// H.264 / H.265 encoder (not the software AV1 encoder). A `cbr` rung with
    /// no rate of its own takes `--video-bitrate`, else a default by codec,
    /// size and frame rate (H.264 1080p30 5 Mb/s, 720p 3M, 480p 1.2M, 360p
    /// 0.8M, 2160p 16M; H.265 0.65x, AV1 0.5x; more above 30 fps).
    #[arg(long = "rate-mode", value_name = "MODE")]
    pub rate_mode: Option<String>,
    /// Encoder effort for every rung: `draft`, `standard` (default) or
    /// `archive`. Each encoder maps it to its own presets (NVENC P5 / P6 /
    /// P7; VP9: a fixed partition at `standard`, ~10 frames/s CIF in
    /// software, a searched one at `archive`, ~1.7).
    #[arg(long = "video-speed", value_name = "TIER")]
    pub video_speed: Option<String>,
    /// Target bitrate for transcoded audio, e.g. `240k` (Opus; MP3 takes
    /// 32k..320k on the MPEG-1 ladder), or `standard` (the default for the
    /// codec and layout). Ignored for passthrough tracks.
    #[arg(long = "audio-bitrate", value_name = "BPS")]
    pub audio_bitrate: Option<String>,
    /// Output channel layout: `source` (default), `mono`, `stereo`, `5.1`,
    /// `7.1` — see `rivet transcode --help`.
    #[arg(long = "audio-channels", value_name = "LAYOUT")]
    pub audio_channels: Option<String>,
    /// Audio filter chain applied before the Opus encoder, e.g.
    /// `channelmap=FL-FL|FR-FR:stereo` — see `rivet transcode --help`.
    #[arg(long = "audio-filter", value_name = "CHAIN")]
    pub audio_filter: Option<String>,
    /// Output color / tonemap policy. The output follows the first clip:
    /// `passthrough` keeps its colour and depth, and later clips are
    /// mapped into it.
    #[arg(long, value_enum, default_value = "sdr")]
    pub color: ColorArg,
    /// 4:4:4 → 4:2:0 chroma filter for 4:4:4 clips (`box` default).
    #[arg(long = "chroma-downsample", value_enum, default_value = "box")]
    pub chroma_downsample: ChromaArg,
    /// Output luma bit depth: `auto` follows the color policy and the first
    /// clip; `8bit` encodes a 10-bit clip at 8 bits.
    #[arg(long, value_enum, default_value = "auto")]
    pub pixel_format: PixelArg,
    /// Video filter chain applied to every clip before scaling, e.g.
    /// `crop=1280:720,hflip` — see `rivet transcode --help`.
    #[arg(long)]
    pub filter: Option<String>,
}

impl OutputShaping {
    pub(crate) fn apply(&self, settings: &mut TranscodeSettings) -> Result<()> {
        settings.filters = match self.filter.as_deref() {
            Some(s) => codec::filter::parse_chain(s).context("parsing --filter")?,
            None => Vec::new(),
        };
        settings.audio_filters = match self.audio_filter.as_deref() {
            Some(s) => codec::audio::filter::parse_chain(s).context("parsing --audio-filter")?,
            None => Vec::new(),
        };
        settings.audio_bitrate = None;
        if let Some(b) = &self.audio_bitrate {
            settings
                .apply_kv("audio-bitrate", b)
                .context("parsing --audio-bitrate")?;
        }
        if let Some(c) = &self.audio_channels {
            settings
                .apply_kv("audio-channels", c)
                .context("parsing --audio-channels")?;
        }
        settings.target = self.target;
        settings.gop = None;
        settings.gop_seconds = None;
        if let Some(g) = &self.gop {
            settings.apply_kv("gop", g).context("parsing --gop")?;
        }
        if let Some(v) = &self.video_bitrate {
            settings
                .apply_kv("video-bitrate", v)
                .context("parsing --video-bitrate")?;
        }
        if let Some(v) = &self.video_buffer {
            settings
                .apply_kv("video-buffer", v)
                .context("parsing --video-buffer")?;
        }
        if let Some(v) = &self.rate_mode {
            settings
                .apply_kv("rate-mode", v)
                .context("parsing --rate-mode")?;
        }
        if let Some(v) = &self.video_speed {
            settings
                .apply_kv("video-speed", v)
                .context("parsing --video-speed")?;
        }
        settings.apply_kv("color", &value_name(self.color))?;
        settings.apply_kv("chroma-downsample", &value_name(self.chroma_downsample))?;
        settings.apply_kv("bit-depth", &value_name(self.pixel_format))?;
        Ok(())
    }
}

/// Convert a [`rivet::progress::RungStatus`] to a short display label.
pub(crate) fn status_str(s: rivet::progress::RungStatus) -> &'static str {
    match s {
        rivet::progress::RungStatus::Pending => "pend",
        rivet::progress::RungStatus::Running => "run",
        rivet::progress::RungStatus::Finalizing => "final",
        rivet::progress::RungStatus::Completed => "done",
        rivet::progress::RungStatus::Failed => "FAIL",
    }
}

/// JSON-escape a bare string value (no surrounding quotes).
pub(crate) fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Transcode `input` honouring `settings`; returns `(mp4_bytes, frame_count, audio_label)`.
///
/// All-default settings take the fast [`rivet::transcode_bytes`] path; any set field
/// routes through [`rivet::TranscodeSettings::into_spec`] + the full `run_job` engine.
pub(crate) fn stream_transcode(
    input: &[u8],
    settings: &TranscodeSettings,
) -> Result<(Vec<u8>, u64, String)> {
    if settings.is_empty() {
        let out = rivet::transcode_bytes(input).context("transcoding")?;
        return Ok((
            out.output_bytes,
            out.frames_processed,
            out.audio_handling.label(),
        ));
    }
    let probed = rivet::probe_bytes(input).context("probing input")?;
    let spec = settings
        .clone()
        .into_spec_for(&probed)
        .context("invalid settings")?;
    if matches!(spec.mode, rivet::OutputMode::Hls { .. }) {
        bail!(
            "HLS/segmented output isn't supported over pipe/ipc (a single stream) — \
             use `rivet transcode -o <dir>` or the HTTP API"
        );
    }
    let sink = Arc::new(rivet::fn_sink(|_p: RungProgress| {}));
    let out = rivet::run_job_blocking(input, &spec, None, sink).context("transcoding")?;
    let audio = out.audio_handling.clone();
    for r in out.rungs {
        let frames = r.frames;
        if let RungArtifact::File(bytes) = r.artifact {
            return Ok((bytes, frames, audio));
        }
    }
    bail!("no single-file output produced")
}
