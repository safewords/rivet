//! Implementation of `rivet ndi` (the `ndi` feature): `sources`, `record`,
//! `send`.

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rivet::codec::audio::AudioCodec;
use rivet::codec::encode::EncoderBackend;
use rivet::ndi::{EndReason, RecordAudio, RecordOptions, SendOptions};

use super::esc;
use crate::{ColorArg, PixelArg, value_name};

/// `rivet ndi`'s subcommands. Parsed once per run, so the variants' sizes
/// differing costs nothing.
#[allow(clippy::large_enum_variant)]
#[derive(clap::Subcommand)]
pub(crate) enum NdiCommand {
    /// List the NDI sources on the network.
    Sources {
        /// Seconds to listen for sources before listing them.
        #[arg(long, default_value_t = 3.0, value_name = "SECONDS")]
        wait: f64,
        #[command(flatten)]
        find: FindArgs,
        /// Print JSON instead of a list.
        #[arg(long)]
        json: bool,
    },
    /// Record an NDI source into an MP4, QuickTime or WebM file, until
    /// `--duration`, `--frames`, the source going away, or Ctrl+C.
    Record(RecordArgs),
    /// Play a media file out as an NDI source.
    Send(SendArgs),
}

/// Where discovery looks.
#[derive(clap::Args, Clone)]
pub(crate) struct FindArgs {
    /// NDI groups to look in, comma-separated (default: the runtime's).
    #[arg(long, value_name = "GROUPS")]
    pub groups: Option<String>,
    /// Further machines to ask directly, comma-separated IP addresses (for
    /// networks mDNS does not cross).
    #[arg(long = "extra-ips", value_name = "IPS")]
    pub extra_ips: Option<String>,
}

impl FindArgs {
    fn options(&self) -> ndi::FindOptions {
        ndi::FindOptions {
            groups: self.groups.clone(),
            extra_ips: self.extra_ips.clone(),
            ..Default::default()
        }
    }
}

#[derive(clap::Args)]
pub(crate) struct RecordArgs {
    /// The source: its full name (`MACHINE (Stream)`) or any part of it
    /// that names only one source.
    pub source: String,
    /// The file to write (`.mp4`, `.mov` or `.webm`).
    #[arg(short, long)]
    pub output: PathBuf,
    /// Stop after this long: `90`, `90s`, `15m`, `2h`, `1h30m`.
    #[arg(long, value_name = "DURATION")]
    pub duration: Option<String>,
    /// Stop after this many frames.
    #[arg(long, value_name = "N")]
    pub frames: Option<u64>,
    /// Output video codec: `av1` (default), `h264`, `h265`, `vp9`, `vp8`,
    /// `mpeg2`, `mpeg4` or `prores[-PROFILE]`.
    #[arg(long)]
    pub codec: Option<String>,
    /// `mp4`, `mov` or `webm` (default: the output's extension, else the
    /// codec's own).
    #[arg(long, value_name = "mp4|mov|webm")]
    pub container: Option<String>,
    /// Constant rate factor (lower = better); wins over `--target`.
    #[arg(long)]
    pub crf: Option<u8>,
    /// Perceptual quality target: `visually_lossless`, `high`, `standard`
    /// (default), `low`, or `vmaf=N`.
    #[arg(long, value_parser = rivet::settings::parse_quality_target)]
    pub target: Option<rivet::codec::encode::tuning::QualityTarget>,
    /// Seconds between keyframes.
    #[arg(long, default_value_t = 2.0, value_name = "SECONDS")]
    pub gop: f64,
    /// Output colour policy (`sdr` tonemaps an HDR source).
    #[arg(long, value_enum, default_value = "sdr")]
    pub color: ColorArg,
    /// Output bit depth.
    #[arg(long = "pixel-format", value_enum, default_value = "auto")]
    pub pixel_format: PixelArg,
    /// Audio: `opus` (default), `aac`, `he-aac`, or `none`.
    #[arg(long, default_value = "opus", value_name = "CODEC")]
    pub audio: String,
    /// Audio bitrate, e.g. `160k` (default: the codec's for the layout).
    #[arg(long = "audio-bitrate", value_name = "BPS")]
    pub audio_bitrate: Option<String>,
    /// Video filter chain, e.g. `crop=1280:720` — see `rivet transcode --help`.
    #[arg(long)]
    pub filter: Option<String>,
    /// Ask the source for its 10-bit stream when it sends one (P216);
    /// otherwise 8-bit. Pair with `--pixel-format 10bit` to keep the bits.
    #[arg(long = "high-bit-depth")]
    pub high_bit_depth: bool,
    /// Ask for the sender's low-bandwidth proxy stream.
    #[arg(long = "low-bandwidth")]
    pub low_bandwidth: bool,
    /// Seconds to wait for the source to appear and send a picture.
    #[arg(long, default_value_t = 15.0, value_name = "SECONDS")]
    pub wait: f64,
    /// End the recording when no picture arrives for this many seconds (0:
    /// wait for ever).
    #[arg(long = "idle-timeout", default_value_t = 10.0, value_name = "SECONDS")]
    pub idle_timeout: f64,
    /// Force an encoder: `nvenc`, `amf`, `qsv`, `h26x` or `av1`.
    #[arg(long, value_name = "BACKEND")]
    pub encoder: Option<String>,
    /// The GPU (global index, see `rivet devices`) the encoder runs on.
    #[arg(long, value_name = "N")]
    pub gpu: Option<u32>,
    #[command(flatten)]
    pub find: FindArgs,
    /// Print the outcome as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(clap::Args)]
pub(crate) struct SendArgs {
    /// The media file to send.
    pub input: PathBuf,
    /// The stream name (receivers see `MACHINE (NAME)`); default: the
    /// file's name.
    #[arg(long)]
    pub name: Option<String>,
    /// Start again at the end, until Ctrl+C.
    #[arg(long = "loop")]
    pub repeat: bool,
    /// Send 10-bit P216 in the file's own colour (default 8-bit, SDR
    /// BT.709, an HDR file tonemapped).
    #[arg(long = "ten-bit")]
    pub ten_bit: bool,
    /// Send no audio.
    #[arg(long = "no-audio")]
    pub no_audio: bool,
    /// NDI groups to announce in, comma-separated.
    #[arg(long, value_name = "GROUPS")]
    pub groups: Option<String>,
}

pub(crate) fn run(command: NdiCommand) -> Result<()> {
    match command {
        NdiCommand::Sources { wait, find, json } => sources(wait, &find, json),
        NdiCommand::Record(args) => record(args),
        NdiCommand::Send(args) => send(args),
    }
}

fn sources(wait: f64, find: &FindArgs, json: bool) -> Result<()> {
    let wait = seconds(wait, "--wait")?;
    let found = rivet::ndi::list_sources(&find.options(), wait)?;
    if json {
        let items: Vec<String> = found
            .iter()
            .map(|s| {
                format!(
                    "{{\"name\":\"{}\",\"url\":{}}}",
                    esc(&s.name),
                    s.url
                        .as_deref()
                        .map_or("null".to_string(), |u| format!("\"{}\"", esc(u)))
                )
            })
            .collect();
        println!("[{}]", items.join(","));
    } else if found.is_empty() {
        eprintln!("no NDI sources seen in {:.1} s", wait.as_secs_f64());
    } else {
        for s in &found {
            match &s.url {
                Some(u) => println!("{}  ({u})", s.name),
                None => println!("{}", s.name),
            }
        }
    }
    Ok(())
}

fn record(args: RecordArgs) -> Result<()> {
    let mut options = RecordOptions::new(&args.output);
    if let Some(c) = &args.codec {
        options.video_codec = rivet::settings::parse_video_codec(c).context("parsing --codec")?;
    }
    if let Some(c) = &args.container {
        options.container =
            Some(rivet::settings::parse_container(c).context("parsing --container")?);
    }
    options.crf = args.crf;
    options.target = args.target;
    anyhow::ensure!(
        args.gop.is_finite() && args.gop > 0.0,
        "--gop must be a positive number of seconds"
    );
    options.gop_seconds = args.gop;
    options.color = rivet::settings::parse_color(&value_name(args.color))?;
    options.bit_depth = rivet::settings::parse_bit_depth(&value_name(args.pixel_format))?;
    options.audio = match args.audio.to_ascii_lowercase().as_str() {
        "opus" | "auto" => RecordAudio::Auto,
        "aac" => RecordAudio::Codec(AudioCodec::Aac),
        "he-aac" | "heaac" => RecordAudio::Codec(AudioCodec::HeAac),
        "none" | "drop" => RecordAudio::Drop,
        o => bail!("--audio must be opus|aac|he-aac|none, got '{o}'"),
    };
    options.audio_bitrate = args
        .audio_bitrate
        .as_deref()
        .map(rivet::settings::parse_bitrate)
        .transpose()
        .context("parsing --audio-bitrate")?;
    if let Some(f) = &args.filter {
        options.filters = rivet::codec::filter::parse_chain(f).context("parsing --filter")?;
    }
    options.duration = args
        .duration
        .as_deref()
        .map(parse_duration)
        .transpose()
        .context("parsing --duration")?;
    options.max_frames = args.frames;
    options.start_timeout = seconds(args.wait, "--wait")?;
    options.idle_timeout = seconds(args.idle_timeout, "--idle-timeout")?;
    options.encoder_backend = args
        .encoder
        .as_deref()
        .map(|e| match e.to_ascii_lowercase().as_str() {
            "nvenc" => Ok(EncoderBackend::Nvenc),
            "amf" => Ok(EncoderBackend::Amf),
            "qsv" => Ok(EncoderBackend::Qsv),
            "h26x" => Ok(EncoderBackend::H26x),
            "av1" => Ok(EncoderBackend::Av1),
            o => Err(anyhow::anyhow!(
                "--encoder must be nvenc|amf|qsv|h26x|av1, got '{o}'"
            )),
        })
        .transpose()?;
    options.gpu_index = args.gpu;
    let stop = ctrl_c_flag();
    options.stop = Some(Arc::clone(&stop));

    let receiver = ndi::ReceiverOptions {
        color_format: if args.high_bit_depth {
            ndi::ColorFormat::Best
        } else {
            ndi::ColorFormat::Fastest
        },
        bandwidth: if args.low_bandwidth {
            ndi::Bandwidth::Lowest
        } else {
            ndi::Bandwidth::Highest
        },
        name: Some("rivet".into()),
        ..Default::default()
    };
    let source = rivet::ndi::NdiSource::connect(
        &args.source,
        &args.find.options(),
        &receiver,
        options.start_timeout,
    )?;
    eprintln!(
        "recording {} → {} (Ctrl+C to stop)",
        rivet::ndi::LiveSource::name(&source),
        args.output.display()
    );
    let outcome = rivet::ndi::record(source, &options, |p| {
        eprint!(
            "\r  {:>8.1} s  {:>7} frames  {:>5.1} fps  {} repeated  {} dropped   ",
            p.seconds, p.frames, p.fps, p.repeated, p.dropped
        );
        let _ = std::io::stderr().flush();
    })?;
    eprintln!();
    let ended = match outcome.ended {
        EndReason::Limit => "limit reached",
        EndReason::Stopped => "stopped",
        EndReason::Idle => "the source stopped sending",
        EndReason::SourceLost => "the source went away",
        EndReason::SourceEnded => "the source ended",
    };
    let p = &outcome.progress;
    if args.json {
        println!(
            "{{\"output\":\"{}\",\"source\":\"{}\",\"width\":{},\"height\":{},\"frame_rate\":\"{}/{}\",\"codec\":\"{}\",\"audio\":{},\"frames\":{},\"seconds\":{:.3},\"repeated\":{},\"dropped\":{},\"dropped_behind\":{},\"ended\":\"{}\"}}",
            esc(&outcome.output.display().to_string()),
            esc(&outcome.source),
            outcome.width,
            outcome.height,
            outcome.frame_rate.0,
            outcome.frame_rate.1,
            outcome.video_codec.label(),
            outcome
                .audio
                .as_deref()
                .map_or("null".into(), |a| format!("\"{}\"", esc(a))),
            p.frames,
            p.seconds,
            p.repeated,
            p.dropped,
            p.dropped_behind,
            ended
        );
    } else {
        println!(
            "{}: {}x{} @ {}/{} {}, audio {}, {} frames ({:.1} s; {} repeated, {} dropped) — {ended}",
            outcome.output.display(),
            outcome.width,
            outcome.height,
            outcome.frame_rate.0,
            outcome.frame_rate.1,
            outcome.video_codec.label(),
            outcome.audio.as_deref().unwrap_or("none"),
            p.frames,
            p.seconds,
            p.repeated,
            p.dropped
        );
        if p.dropped_behind > 0 {
            eprintln!(
                "note: {} pictures arrived while the encoder was behind and were replaced by repeats; \
                 a GPU encoder (or --codec h264 / a faster --target) keeps up with a live source",
                p.dropped_behind
            );
        }
    }
    Ok(())
}

fn send(args: SendArgs) -> Result<()> {
    let name = args.name.clone().unwrap_or_else(|| {
        args.input
            .file_stem()
            .map_or("rivet".into(), |s| s.to_string_lossy().into_owned())
    });
    let mut options = SendOptions::new(&args.input, name);
    options.groups = args.groups.clone();
    options.repeat = args.repeat;
    options.ten_bit = args.ten_bit;
    options.audio = !args.no_audio;
    let stop = ctrl_c_flag();
    options.stop = Some(Arc::clone(&stop));
    eprintln!(
        "sending {} as NDI source `{}` (Ctrl+C to stop)",
        args.input.display(),
        options.name
    );
    let outcome = rivet::ndi::send_file(&options, |frames| {
        eprint!("\r  {frames} frames sent   ");
        let _ = std::io::stderr().flush();
    })?;
    eprintln!();
    println!(
        "sent {} frames of {}x{} @ {}/{}{} in {} pass(es), {:.1} s{}",
        outcome.frames,
        outcome.width,
        outcome.height,
        outcome.frame_rate.0,
        outcome.frame_rate.1,
        if outcome.audio { " with audio" } else { "" },
        outcome.passes,
        outcome.elapsed.as_secs_f64(),
        if outcome.stopped { " (stopped)" } else { "" }
    );
    Ok(())
}

/// A flag Ctrl+C sets. A second Ctrl+C exits at once.
fn ctrl_c_flag() -> Arc<AtomicBool> {
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
                eprintln!("\nstopping (Ctrl+C again to abandon)…");
                if tokio::signal::ctrl_c().await.is_ok() {
                    std::process::exit(130);
                }
            }
        });
    });
    flag
}

fn seconds(s: f64, flag: &str) -> Result<Duration> {
    anyhow::ensure!(
        s.is_finite() && s >= 0.0,
        "{flag} must be a number of seconds"
    );
    Ok(Duration::from_secs_f64(s))
}

/// `90`, `90s`, `1.5m`, `2h`, `1h30m`, `1h2m3s`.
pub(crate) fn parse_duration(s: &str) -> Result<Duration> {
    let s = s.trim();
    if let Ok(secs) = s.parse::<f64>() {
        return seconds(secs, "a duration");
    }
    let (mut total, mut number) = (0.0f64, String::new());
    for c in s.chars() {
        if c.is_ascii_digit() || c == '.' {
            number.push(c);
            continue;
        }
        let unit = match c {
            'h' => 3600.0,
            'm' => 60.0,
            's' => 1.0,
            _ => bail!("'{s}' is not a duration (90, 90s, 15m, 2h, 1h30m)"),
        };
        let n: f64 = number
            .parse()
            .with_context(|| format!("'{s}' is not a duration"))?;
        total += n * unit;
        number.clear();
    }
    anyhow::ensure!(number.is_empty(), "'{s}' ends without a unit");
    seconds(total, "a duration")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_read_as_people_write_them() {
        let d = |s| parse_duration(s).unwrap().as_secs_f64();
        assert_eq!(d("90"), 90.0);
        assert_eq!(d("90s"), 90.0);
        assert_eq!(d("1.5m"), 90.0);
        assert_eq!(d("1h30m"), 5400.0);
        assert_eq!(d("2h"), 7200.0);
        assert!(parse_duration("10x").is_err());
        assert!(parse_duration("1h30").is_err());
    }
}
