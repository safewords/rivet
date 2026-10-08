//! The job engine's live path: a job whose input is a live source (an NDI
//! source) or whose output is live (a file played out to NDI), run on the
//! same [`OutputSpec`] every other job is.
//!
//! ```text
//! capture thread: source.next_event → queue (pictures bounded; audio not)
//! engine thread:  queue → timeline → hooks → FrameNormalizer ─┬→ rung 0 worker → scale → encode → MP4 / CMAF / NDI
//!                                   └→ audio: place → EncodeState(s) → muxers   └→ rung 1 worker → …
//! ```
//!
//! What the spec says is done as a file job does it: the rungs are fitted to
//! the source ([`crate::fit`]), the rung policy and constant rates resolved,
//! the source's colour and depth checked against the output, each rung's
//! encoder configured by its quality (CRF, target, bitrate, GOP, speed), the
//! encode plan's cards chosen ([`placement`]), the decode pump's per-frame
//! work (colour, tonemap, depth, filters) done by its `FrameNormalizer`, the
//! audio policy (codec, bitrate, channels, filters) by the audio path's
//! `EncodeState`, and every rung muxed into the file it asks for — a single
//! file each, or an HLS package written **as it records** (`EVENT`
//! playlists, rewritten as the final `VOD` package at the end).
//!
//! **The timeline.** A file has a fixed frame rate and a gapless audio
//! track; a live source has neither. Each picture is placed on the frame its
//! timestamp falls on, at the output rate (the source's, or the spec's
//! `max-fps` cap), counted from the first picture: a picture whose frame is
//! written already is dropped, a gap is filled by repeating the last
//! picture, and a jump of more than ten seconds re-anchors the timeline.
//! Audio is placed the same way — gaps over 40 ms filled with silence,
//! overlaps trimmed — so it stays in step however the encoders keep up.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use codec::audio::AudioFrame;
use codec::encode::{self, EncoderConfig};
use codec::frame::{StreamInfo, VideoFrame};
use codec::gpu::GpuVendor;
use container::AudioInfo;
use container::cmaf::{CmafAudioMuxer, CmafTrackManifest, CmafVideoMuxer, CmafVideoMuxerOptions};
use container::hls::{AudioVariantSpec, VideoVariantSpec};
use container::streaming::DemuxHeader;

use crate::decode_pump::{DecodePumpConfig, FrameNormalizer};
use crate::live::TICKS_PER_SECOND;
use crate::live::{LiveAudio, LiveEvent, LiveSource, LiveVideo, NdiEndpoint, SourceLost};
use crate::multigpu::{self, RungManifest};
use crate::progress::{JobEvent, ProgressSink, RungStatus};
use crate::spec::{AudioChannels, AudioCodecPolicy, Container, OutputMode, OutputSpec, Rung};

use super::audio::{AudioRequest, EncodeState, audio_codec_string};
use super::file_mux::FileMuxer;
use super::run::{encoder_backend_override, report, serial_threads_per_rung};
use super::{
    FRAME_CHANNEL_CAPACITY, JobOutput, RungArtifact, RungOutput, fit_to, remap_rung_indices,
    report_rung_error,
};

/// Where a live job's output goes.
#[derive(Debug, Clone)]
pub enum LiveTarget {
    /// One file: a single-file job with one rung.
    File(PathBuf),
    /// A directory: `<label>.<ext>` per rung of a single-file job, or the
    /// HLS package's root.
    Dir(PathBuf),
    /// NDI: each rung announced as a source — under the endpoint's name for
    /// one rung, `NAME (LABEL)` for each of several. Needs the `ndi` feature.
    Ndi(NdiEndpoint),
}

/// Why a live job ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveEnd {
    /// The spec's `duration` was reached.
    Duration,
    /// The caller's stop flag was set.
    Stopped,
    /// No picture came for the spec's `idle-timeout`.
    Idle,
    /// The source went away.
    SourceLost,
    /// The source ended (a file).
    SourceEnded,
}

impl LiveEnd {
    /// A few words for a summary line.
    pub fn as_str(self) -> &'static str {
        match self {
            LiveEnd::Duration => "duration reached",
            LiveEnd::Stopped => "stopped",
            LiveEnd::Idle => "the source stopped sending",
            LiveEnd::SourceLost => "the source went away",
            LiveEnd::SourceEnded => "the source ended",
        }
    }
}

/// How a live job went: what the timeline did to keep the output in step.
#[derive(Debug, Clone)]
pub struct LiveStats {
    /// The source's name.
    pub source: String,
    /// Frames written (repeats included), at `frame_rate`.
    pub frames: u64,
    /// The output frame rate, `(numerator, denominator)`.
    pub frame_rate: (u32, u32),
    /// Frames written as a repeat of the one before, to fill a gap.
    pub repeated: u64,
    /// Pictures dropped because their frame was written already (a source
    /// faster than the output rate, or jitter).
    pub dropped_early: u64,
    /// Pictures dropped because the encoders were behind (a live source
    /// only); their frames were filled by repeats.
    pub dropped_behind: u64,
    /// Samples (per channel) of audio placed on the timeline.
    pub audio_samples: u64,
    pub ended: LiveEnd,
}

impl LiveStats {
    /// Seconds of output.
    pub fn seconds(&self) -> f64 {
        self.frames as f64 * f64::from(self.frame_rate.1) / f64::from(self.frame_rate.0.max(1))
    }
}

/// Run a live job. Async — call from within a Tokio runtime. `stop`, when
/// set, ends it (Ctrl+C, an API call); whatever was made is kept.
///
/// The spec's [hooks](crate::hooks) run as on any job: probe hooks once the
/// first picture says what the source is, decoded-frame and encoder-frame
/// hooks on every picture, artifact hooks for each output, then completed
/// or failed.
pub async fn run_live_job<S: LiveSource + 'static>(
    source: S,
    spec: &OutputSpec,
    target: LiveTarget,
    sink: Arc<dyn ProgressSink>,
    stop: Option<Arc<AtomicBool>>,
) -> Result<JobOutput> {
    let _slot = crate::thread_budget::enter_job();
    let spec = if spec.hooks.is_empty() {
        spec.clone()
    } else {
        let hooks = spec.hooks.ensure_session(crate::hooks::JobKind::Transcode);
        spec.clone().with_hooks(hooks)
    };
    let hooks = spec.hooks.clone();
    let run = async move {
        tokio::task::spawn_blocking(move || run_live_inner(source, &spec, target, sink, stop))
            .await
            .context("the live job panicked")?
    };
    if hooks.is_empty() {
        return run.await;
    }
    let mut out = hooks.run(super::artifact_events, run).await?;
    out.hooks = hooks.report();
    Ok(out)
}

/// [`run_live_job`] on a runtime of its own.
pub fn run_live_job_blocking<S: LiveSource + 'static>(
    source: S,
    spec: &OutputSpec,
    target: LiveTarget,
    sink: Arc<dyn ProgressSink>,
    stop: Option<Arc<AtomicBool>>,
) -> Result<JobOutput> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building Tokio runtime")?;
    rt.block_on(run_live_job(source, spec, target, sink, stop))
}

/// The settings a live job cannot honour, refused by name before anything
/// runs, and the target's fit to the spec's shape.
pub fn check_live_spec(spec: &OutputSpec, target: &LiveTarget) -> Result<()> {
    let refuse = |what: &str, why: &str| -> Result<()> { bail!("{what} {why}") };
    if spec.mode == OutputMode::AudioOnly {
        refuse(
            "mode=audio",
            "is not available for a live job: record the video too (audio=… still chooses the sound)",
        )?;
    }
    if spec.trim_start.is_some() || spec.trim_end.is_some() {
        refuse(
            "trim-start / trim-end",
            "cut a file; a live job runs for `duration` (duration=90s, 1h30m) or until stopped",
        )?;
    }
    if !spec.metadata_keep.is_empty() {
        refuse(
            "metadata-keep",
            "carries a source file's metadata, and a live source has none",
        )?;
    }
    if spec.input_frame_rate.is_some() {
        refuse(
            "input-fps",
            "times a raw elementary stream; a live source times its own pictures",
        )?;
    }
    match target {
        LiveTarget::Ndi(_) => {
            if matches!(spec.mode, OutputMode::Hls { .. }) {
                refuse(
                    "mode=hls",
                    "writes a package of files; an NDI output is a live stream (one source per rung)",
                )?;
            }
        }
        LiveTarget::File(path) => {
            if matches!(spec.mode, OutputMode::Hls { .. }) {
                bail!(
                    "an HLS package is a directory of files; {} names one file",
                    path.display()
                );
            }
            if spec.rungs.len() > 1 {
                bail!(
                    "{} rungs make {} files; give a directory, not {}",
                    spec.rungs.len(),
                    spec.rungs.len(),
                    path.display()
                );
            }
        }
        LiveTarget::Dir(_) => {}
    }
    if let Some(d) = spec.live.duration
        && !(d.is_finite() && d > 0.0)
    {
        bail!("duration must be a positive length of time, got {d}");
    }
    Ok(())
}

/// A live job's decode plan is the source's: refused when it names one,
/// since there is nothing for rivet to decode.
fn check_decode_plan(spec: &OutputSpec, realtime: bool) -> Result<()> {
    if realtime && spec.decode_policy != crate::spec::DecodePolicy::Auto {
        bail!(
            "decode={:?} chooses rivet's decoder for a file; a live source arrives decoded \
             (the NDI runtime decodes NDI), so there is no decode to plan",
            spec.decode_policy
        );
    }
    Ok(())
}

// ─── The capture thread ───────────────────────────────────────────

/// Pictures the queue holds: about a second at 60 fps. A live source's
/// picture beyond it is dropped (its frame repeated); a file's waits.
const QUEUE_PICTURES: usize = 60;

#[derive(Default)]
struct Queue {
    pictures: AtomicUsize,
    behind: AtomicU64,
}

enum Message {
    Event(LiveEvent),
    Lost,
    Failed(anyhow::Error),
}

fn capture_loop<S: LiveSource>(
    mut source: S,
    tx: Sender<Message>,
    halt: &AtomicBool,
    queue: &Queue,
    realtime: bool,
) {
    while !halt.load(Ordering::Relaxed) {
        let msg = match source.next_event(Duration::from_millis(100)) {
            Ok(None) => continue,
            Ok(Some(LiveEvent::Video(v))) => {
                if realtime {
                    if queue.pictures.load(Ordering::Acquire) >= QUEUE_PICTURES {
                        queue.behind.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                } else {
                    // A file waits for room rather than losing a picture.
                    while queue.pictures.load(Ordering::Acquire) >= QUEUE_PICTURES {
                        if halt.load(Ordering::Relaxed) {
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(2));
                    }
                }
                queue.pictures.fetch_add(1, Ordering::AcqRel);
                Message::Event(LiveEvent::Video(v))
            }
            Ok(Some(e @ LiveEvent::End)) => {
                let _ = tx.send(Message::Event(e));
                return;
            }
            Ok(Some(e)) => Message::Event(e),
            Err(e) if e.is::<SourceLost>() => {
                let _ = tx.send(Message::Lost);
                return;
            }
            Err(e) => {
                let _ = tx.send(Message::Failed(e));
                return;
            }
        };
        if tx.send(msg).is_err() {
            return;
        }
    }
}

fn run_live_inner<S: LiveSource + 'static>(
    source: S,
    spec: &OutputSpec,
    target: LiveTarget,
    sink: Arc<dyn ProgressSink>,
    stop: Option<Arc<AtomicBool>>,
) -> Result<JobOutput> {
    let started = Instant::now();
    spec.validate().context("invalid OutputSpec")?;
    check_live_spec(spec, &target)?;
    check_decode_plan(spec, source.is_realtime())?;
    let name = source.name();
    let kind = source.kind();
    let realtime = source.is_realtime();

    let halt = Arc::new(AtomicBool::new(false));
    let queue = Arc::new(Queue::default());
    let (tx, rx) = mpsc::channel::<Message>();
    let capture = {
        let (halt, queue) = (Arc::clone(&halt), Arc::clone(&queue));
        std::thread::Builder::new()
            .name("live-capture".into())
            .spawn(move || capture_loop(source, tx, &halt, &queue, realtime))
            .context("starting the capture thread")?
    };
    let result = run_engine(
        &rx, spec, target, sink, &queue, stop, started, &name, kind, realtime,
    );
    halt.store(true, Ordering::Relaxed);
    drop(rx);
    let _ = capture.join();
    result
}

#[allow(clippy::too_many_arguments)]
fn run_engine(
    rx: &Receiver<Message>,
    spec: &OutputSpec,
    target: LiveTarget,
    sink: Arc<dyn ProgressSink>,
    queue: &Queue,
    stop: Option<Arc<AtomicBool>>,
    started: Instant,
    name: &str,
    kind: &'static str,
    realtime: bool,
) -> Result<JobOutput> {
    let stopped = || stop.as_ref().is_some_and(|s| s.load(Ordering::Relaxed));
    let start_timeout = Duration::from_secs_f64(spec.live.start_timeout.max(0.0));
    let mut pending_audio: Vec<LiveAudio> = Vec::new();
    // The first picture says what the source is.
    let first = loop {
        if stopped() {
            bail!("stopped before {name} sent a picture");
        }
        let left = start_timeout.saturating_sub(started.elapsed());
        if left.is_zero() {
            bail!(
                "no picture from {name} within {:.0} s{}",
                start_timeout.as_secs_f64(),
                if pending_audio.is_empty() {
                    ""
                } else {
                    " (it sends audio only? a video job needs pictures)"
                }
            );
        }
        match rx.recv_timeout(left.min(Duration::from_millis(100))) {
            Ok(Message::Event(LiveEvent::Video(v))) => {
                queue.pictures.fetch_sub(1, Ordering::AcqRel);
                break v;
            }
            Ok(Message::Event(LiveEvent::Audio(a))) => {
                pending_audio.push(a);
                if pending_audio.len() > 400 {
                    pending_audio.remove(0);
                }
            }
            Ok(Message::Event(LiveEvent::End)) => bail!("{name} ended before sending a picture"),
            Ok(Message::Lost) => bail!("{name} went away before sending a picture"),
            Ok(Message::Failed(e)) => return Err(e.context(format!("receiving from {name}"))),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => bail!("the capture thread ended"),
        }
    };

    let mut engine = Engine::start(&first, spec, target, sink, name, kind, realtime)?;
    engine.video(first)?;
    for a in pending_audio {
        engine.audio(a)?;
    }

    let idle = Duration::from_secs_f64(spec.live.idle_timeout.max(0.0));
    let mut last_picture = Instant::now();
    let ended = loop {
        engine.dropped_behind = queue.behind.load(Ordering::Relaxed);
        if engine.limit_reached() {
            break LiveEnd::Duration;
        }
        if stopped() {
            break LiveEnd::Stopped;
        }
        if realtime && !idle.is_zero() && last_picture.elapsed() >= idle {
            tracing::warn!(
                seconds = idle.as_secs_f64(),
                "no picture from the source; ending the job"
            );
            break LiveEnd::Idle;
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Message::Event(LiveEvent::Video(v))) => {
                queue.pictures.fetch_sub(1, Ordering::AcqRel);
                last_picture = Instant::now();
                engine.video(v)?;
            }
            Ok(Message::Event(LiveEvent::Audio(a))) => engine.audio(a)?,
            Ok(Message::Event(LiveEvent::End)) => break LiveEnd::SourceEnded,
            Ok(Message::Lost) => {
                tracing::warn!("the source went away; keeping what was made");
                break LiveEnd::SourceLost;
            }
            Ok(Message::Failed(e)) => return Err(e.context(format!("receiving from {name}"))),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break LiveEnd::SourceEnded,
        }
    };
    engine.dropped_behind = queue.behind.load(Ordering::Relaxed);
    engine.finish(ended, started)
}

// ─── The timeline ─────────────────────────────────────────────────

/// The slot `time` falls on, counted from `t0`, at `rate_num / rate_den` per
/// second, rounded to the nearest.
fn slot(time: i64, t0: i64, rate_num: u64, rate_den: u64) -> i64 {
    let num = (time - t0) as i128 * rate_num as i128;
    let den = TICKS_PER_SECOND as i128 * rate_den as i128;
    ((2 * num + num.signum() * den) / (2 * den)) as i64
}

/// Samples at `rate` in `frames` frames at `frame_rate`.
fn samples_in(frames: u64, frame_rate: (u32, u32), rate: u32) -> u64 {
    (u128::from(frames) * u128::from(rate) * u128::from(frame_rate.1)
        / u128::from(frame_rate.0.max(1))) as u64
}

#[derive(Debug, PartialEq, Eq)]
enum Place {
    /// Write it as the next frame, after repeating the last this many times.
    Write { repeats: u64 },
    /// Its frame is written already.
    Drop,
    /// Too far from the timeline: re-anchor so it is the next frame.
    Reanchor,
}

fn place(k: i64, n: u64, max_gap: u64) -> Place {
    let ahead = k - n as i64;
    if ahead < -(max_gap as i64) || ahead > max_gap as i64 {
        Place::Reanchor
    } else if ahead < 0 {
        Place::Drop
    } else {
        Place::Write {
            repeats: ahead as u64,
        }
    }
}

/// The output frame rate: the source's, under the spec's `max-fps` cap.
fn output_rate(source: (u32, u32), cap: Option<f64>) -> (u32, u32) {
    let source = match source {
        (n, d) if n > 0 && d > 0 => source,
        _ => {
            tracing::warn!("the source declares no frame rate; 30 fps");
            (30, 1)
        }
    };
    match cap {
        Some(c) if c > 0.0 && c < f64::from(source.0) / f64::from(source.1) => {
            crate::live::rational_frame_rate(c)
        }
        _ => source,
    }
}

// ─── Encode placement ─────────────────────────────────────────────

/// Rungs of every live job so far, so concurrent jobs take turns over the
/// cards rather than all starting on the first.
static LIVE_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Where each rung encodes, `(gpu_index, gpu_vendor)`, under the spec's
/// encode plan. A plan that leaves nothing to encode on is refused by name
/// here, before anything is encoded (as a file job's is). A plan that
/// spreads (`all`, `per-rung`, `family:…`) over several capable cards puts
/// each rung on the next card in turn, across every live job in the process;
/// `single` and `gpu:N` put every rung on the one card; a software host, or a
/// bitrate rung (coded in software), takes the serial target.
fn placement(
    spec: &OutputSpec,
    output_pixel_format: codec::frame::PixelFormat,
) -> Result<Vec<(Option<u32>, Option<GpuVendor>)>> {
    let rungs = spec.rungs.len();
    let serial_pool = multigpu::gpu_pool_for_serial_job(spec, output_pixel_format)?;
    multigpu::check_rate_pool(
        spec,
        &serial_pool,
        output_pixel_format,
        encoder_backend_override(),
    )?;
    let serial = multigpu::serial_target(spec.encode_policy, &serial_pool);
    let rate_rungs = spec
        .rungs
        .iter()
        .any(|r| r.quality.overrides.bitrate.is_some());
    if !spec.encode_policy.spreads() || encoder_backend_override().is_some() || rate_rungs {
        return Ok(vec![serial; rungs]);
    }
    let pool = match multigpu::gpu_pool_for_job(spec, output_pixel_format) {
        Ok(p) if !p.is_software() && p.capacity() > 1 => p,
        _ => return Ok(vec![serial; rungs]),
    };
    let slots = pool.snapshot_leases();
    let start = LIVE_CURSOR.fetch_add(rungs, Ordering::Relaxed);
    Ok((0..rungs)
        .map(|i| {
            let s = &slots[(start + i) % slots.len()];
            (Some(s.index), Some(s.vendor))
        })
        .collect())
}

// ─── Rung workers ─────────────────────────────────────────────────

enum RungMsg {
    /// A normalised picture at the source's size, the next frame.
    Frame(VideoFrame),
    /// The audio track every rung's file carries.
    AudioTrack(AudioInfo),
    AudioPacket(Arc<Vec<u8>>, u32),
    AudioEdit(container::edit::TrackEdit),
    /// Raw audio, for an NDI output.
    #[cfg_attr(not(feature = "ndi"), allow(dead_code))]
    Pcm(Arc<AudioFrame>),
}

struct WorkerDone {
    output: RungOutput,
    /// An HLS rendition's segments.
    hls: Option<RungManifest>,
}

/// The live HLS package's shared state: each rendition's first segment, so
/// the master playlist is written as soon as every one has something to
/// play.
struct HlsLive {
    root: PathBuf,
    frame_rate: f64,
    video: Vec<Option<VideoVariantSpec>>,
    audio: Vec<Option<AudioVariantSpec>>,
    master_written: bool,
}

impl HlsLive {
    fn update(&mut self) {
        if self.video.iter().any(Option::is_none) || self.video.is_empty() {
            return;
        }
        let video: Vec<VideoVariantSpec> = self.video.iter().flatten().cloned().collect();
        let audio: Vec<AudioVariantSpec> = self.audio.iter().flatten().cloned().collect();
        let master = self.root.join("master.m3u8");
        match container::hls::write_live_master_playlist(&master, &video, &audio) {
            Ok(()) if !self.master_written => {
                self.master_written = true;
                tracing::info!(path = %master.display(), "live HLS: the master playlist is up");
            }
            Ok(()) => {}
            Err(e) => tracing::warn!("live HLS: writing the master playlist: {e:#}"),
        }
    }
}

struct RungWork {
    index: usize,
    rung: Rung,
    cfg: EncoderConfig,
    frame_rate: (u32, u32),
    frames_total: Option<u64>,
    sink: Arc<dyn ProgressSink>,
}

/// A single-file rung: encode, mux, and at the end write the file in place.
fn file_worker(
    w: RungWork,
    rx: Receiver<RungMsg>,
    container: Container,
    path: PathBuf,
) -> Result<WorkerDone> {
    let fps = rate_f64(w.frame_rate);
    let codec = w.cfg.codec;
    let color = w.cfg.color_metadata;
    let mut encoder = encode::select_encoder(w.cfg, encoder_backend_override())
        .with_context(|| format!("creating encoder for rung {}", w.rung.label))?;
    let mut muxer = FileMuxer::new(container, w.rung.width, w.rung.height, fps, codec, false)?;
    muxer.set_color_metadata(color);
    let (mut frames, mut bytes) = (0u64, 0u64);
    report(
        w.sink.as_ref(),
        w.index,
        &w.rung,
        RungStatus::Running,
        0,
        w.frames_total,
        0,
        0,
    );
    let mut audio_ok = true;
    for msg in rx {
        match msg {
            RungMsg::Frame(frame) => {
                let mut scaled = w.rung.scale(&frame).context("scaling to the rung")?;
                scaled.pts = frames;
                encoder.send_frame(&scaled).context("send_frame")?;
                while let Some(pkt) = encoder.receive_packet().context("receive_packet")? {
                    bytes += pkt.data.len() as u64;
                    muxer.add_packet(pkt).context("add_packet")?;
                }
                frames += 1;
                if frames.is_multiple_of(30) {
                    report(
                        w.sink.as_ref(),
                        w.index,
                        &w.rung,
                        RungStatus::Running,
                        frames,
                        w.frames_total,
                        0,
                        bytes,
                    );
                }
            }
            RungMsg::AudioTrack(info) => {
                if let Err(e) = muxer.with_audio(info) {
                    tracing::warn!(rung = %w.rung.label, "the file refuses the audio track ({e:#}); video only");
                    audio_ok = false;
                }
            }
            RungMsg::AudioPacket(data, dur) if audio_ok => muxer.add_audio_sample(&data, dur)?,
            RungMsg::AudioEdit(edit) if audio_ok => muxer.set_audio_edit(edit),
            _ => {}
        }
    }
    encoder.flush().context("encoder flush")?;
    while let Some(pkt) = encoder.receive_packet().context("receive_packet drain")? {
        muxer.add_packet(pkt).context("add_packet drain")?;
    }
    report(
        w.sink.as_ref(),
        w.index,
        &w.rung,
        RungStatus::Finalizing,
        frames,
        w.frames_total,
        0,
        bytes,
    );
    let written = muxer.finalize_to(&path)?;
    report(
        w.sink.as_ref(),
        w.index,
        &w.rung,
        RungStatus::Completed,
        frames,
        w.frames_total,
        0,
        written,
    );
    Ok(WorkerDone {
        output: RungOutput {
            label: w.rung.label.clone(),
            width: w.rung.width,
            height: w.rung.height,
            frames,
            bytes: written,
            artifact: RungArtifact::Written(path),
        },
        hls: None,
    })
}

/// The HLS timing of a rendition: timescale, ticks per frame, frames per
/// segment, ticks per segment.
#[derive(Clone, Copy)]
struct HlsTiming {
    timescale: u32,
    per_frame_ticks: u32,
    segment_ticks: u64,
    target_duration: u32,
}

impl HlsTiming {
    fn new(segment_seconds: f32, fps: f64) -> Self {
        let timescale = (fps * 1000.0).round().max(1.0) as u32;
        let per_frame_ticks = (f64::from(timescale) / fps.max(1.0)).round().max(1.0) as u32;
        let kf = crate::cmaf_util::keyframe_interval_for_segment(f64::from(segment_seconds), fps);
        Self {
            timescale,
            per_frame_ticks,
            segment_ticks: u64::from(kf) * u64::from(per_frame_ticks),
            target_duration: segment_seconds.ceil().max(1.0) as u32,
        }
    }
}

/// An HLS rendition: encode, segment as it goes, keep its playlist current.
fn hls_worker(
    w: RungWork,
    rx: Receiver<RungMsg>,
    timing: HlsTiming,
    live: Arc<Mutex<HlsLive>>,
) -> Result<WorkerDone> {
    let fps = rate_f64(w.frame_rate);
    let codec = w.cfg.codec;
    let color = w.cfg.color_metadata;
    let declared = codec::encode::tuning::ConstantRate::from_overrides(&w.rung.quality.overrides)
        .map(|c| c.bps);
    let mut encoder = encode::select_encoder(w.cfg.clone(), encoder_backend_override())
        .with_context(|| format!("creating encoder for rung {}", w.rung.label))?;
    let relative_dir = format!("video/{}", w.rung.label);
    let root = live.lock().unwrap().root.clone();
    let dir = root.join(&relative_dir);
    let mut muxer = CmafVideoMuxer::new_with_codec_options(
        &dir,
        w.rung.width,
        w.rung.height,
        timing.timescale,
        color,
        codec,
        CmafVideoMuxerOptions::default(),
    )?;
    let playlist = dir.join("playlist.m3u8");
    let manifest_of = |m: &CmafVideoMuxer| CmafTrackManifest {
        init_path: dir.join("init.mp4"),
        segments: m.segments().to_vec(),
        timescale: timing.timescale,
    };
    let rung_manifest = |m: &CmafVideoMuxer| RungManifest {
        rung_index: w.index,
        width: w.rung.width,
        height: w.rung.height,
        label: w.rung.label.clone(),
        relative_dir: relative_dir.clone(),
        manifest: manifest_of(m),
    };
    let flushed = |m: &mut CmafVideoMuxer| -> Result<()> {
        m.flush_segment()?;
        container::hls::write_live_media_playlist(
            &playlist,
            &manifest_of(m),
            timing.target_duration,
        )?;
        let mut l = live.lock().unwrap();
        if l.video[w.index].is_none() {
            let rm = rung_manifest(m);
            let bytes = rm.manifest.byte_size();
            l.video[w.index] = Some(super::pump::build_video_variant_spec(
                &rm, fps, bytes, declared,
            ));
            l.update();
        }
        Ok(())
    };
    let (mut frames, mut bytes) = (0u64, 0u64);
    report(
        w.sink.as_ref(),
        w.index,
        &w.rung,
        RungStatus::Running,
        0,
        w.frames_total,
        0,
        0,
    );
    let add = |m: &mut CmafVideoMuxer,
               pkt: codec::encode::EncodedPacket,
               bytes: &mut u64|
     -> Result<()> {
        if pkt.is_keyframe && m.pending_duration_ticks() >= timing.segment_ticks {
            flushed(m)?;
        }
        *bytes += pkt.data.len() as u64;
        m.add_packet(
            pkt.data.to_vec(),
            timing.per_frame_ticks,
            pkt.is_keyframe,
            pkt.pts,
        )
    };
    for msg in rx {
        if let RungMsg::Frame(frame) = msg {
            let mut scaled = w.rung.scale(&frame).context("scaling to the rung")?;
            scaled.pts = frames;
            encoder.send_frame(&scaled).context("send_frame")?;
            while let Some(pkt) = encoder.receive_packet().context("receive_packet")? {
                add(&mut muxer, pkt, &mut bytes)?;
            }
            frames += 1;
            if frames.is_multiple_of(30) {
                let segs = muxer.segments().len() as u32;
                report(
                    w.sink.as_ref(),
                    w.index,
                    &w.rung,
                    RungStatus::Running,
                    frames,
                    w.frames_total,
                    segs,
                    bytes,
                );
            }
        }
    }
    encoder.flush().context("encoder flush")?;
    while let Some(pkt) = encoder.receive_packet().context("receive_packet drain")? {
        add(&mut muxer, pkt, &mut bytes)?;
    }
    if muxer.pending_duration_ticks() > 0 {
        flushed(&mut muxer)?;
    }
    let rm = RungManifest {
        manifest: muxer.finalize().context("finalizing the rendition")?,
        ..rung_manifest_empty(&w, &relative_dir, timing)
    };
    let segs = rm.manifest.segments.len() as u32;
    let size = rm.manifest.byte_size();
    report(
        w.sink.as_ref(),
        w.index,
        &w.rung,
        RungStatus::Completed,
        frames,
        w.frames_total,
        segs,
        size,
    );
    Ok(WorkerDone {
        output: RungOutput {
            label: w.rung.label.clone(),
            width: w.rung.width,
            height: w.rung.height,
            frames,
            bytes: size,
            artifact: RungArtifact::HlsRendition { dir, relative_dir },
        },
        hls: Some(rm),
    })
}

fn rung_manifest_empty(w: &RungWork, relative_dir: &str, timing: HlsTiming) -> RungManifest {
    RungManifest {
        rung_index: w.index,
        width: w.rung.width,
        height: w.rung.height,
        label: w.rung.label.clone(),
        relative_dir: relative_dir.to_string(),
        manifest: CmafTrackManifest {
            init_path: PathBuf::new(),
            segments: Vec::new(),
            timescale: timing.timescale,
        },
    }
}

/// An NDI output rung: scale, and send as a source.
#[cfg(feature = "ndi")]
fn ndi_worker(
    w: RungWork,
    rx: Receiver<RungMsg>,
    endpoint: NdiEndpoint,
    stream_name: String,
    clock_video: bool,
) -> Result<WorkerDone> {
    let mut out = crate::ndi::NdiSink::new(&endpoint, &stream_name, clock_video)?;
    let mut frames = 0u64;
    report(
        w.sink.as_ref(),
        w.index,
        &w.rung,
        RungStatus::Running,
        0,
        w.frames_total,
        0,
        0,
    );
    for msg in rx {
        match msg {
            RungMsg::Frame(frame) => {
                let scaled = w.rung.scale(&frame).context("scaling to the rung")?;
                out.send_frame(&scaled, w.frame_rate)?;
                frames += 1;
                if frames.is_multiple_of(30) {
                    report(
                        w.sink.as_ref(),
                        w.index,
                        &w.rung,
                        RungStatus::Running,
                        frames,
                        w.frames_total,
                        0,
                        0,
                    );
                }
            }
            RungMsg::Pcm(frame) => out.send_audio(&frame)?,
            _ => {}
        }
    }
    report(
        w.sink.as_ref(),
        w.index,
        &w.rung,
        RungStatus::Completed,
        frames,
        w.frames_total,
        0,
        0,
    );
    Ok(WorkerDone {
        output: RungOutput {
            label: w.rung.label.clone(),
            width: w.rung.width,
            height: w.rung.height,
            frames,
            bytes: 0,
            artifact: RungArtifact::Ndi {
                source: stream_name,
            },
        },
        hls: None,
    })
}

fn rate_f64(r: (u32, u32)) -> f64 {
    f64::from(r.0) / f64::from(r.1.max(1))
}

// ─── Audio ────────────────────────────────────────────────────────

/// One encoded audio rendition: its encoder, and where its packets go.
struct AudioRendition<'a> {
    state: Option<EncodeState<'a>>,
    announced: bool,
    /// The track, once described.
    info: Option<AudioInfo>,
    /// HLS: this rendition's segmenter.
    cmaf: Option<CmafAudioMuxer>,
    relative_dir: &'static str,
    name: &'static str,
    index: usize,
}

struct AudioOut<'a> {
    renditions: Vec<AudioRendition<'a>>,
    /// The rate and channels the first frame set; later frames must match.
    rate: u32,
    channels: u8,
    /// Samples (per channel) placed on the timeline.
    written: u64,
    warned: bool,
}

// ─── The engine ───────────────────────────────────────────────────

enum Output {
    Encode,
    Ndi,
}

struct Engine<'a> {
    spec: &'a OutputSpec,
    name: String,
    frame_rate: (u32, u32),
    t0: i64,
    max_gap_frames: u64,
    limit: Option<u64>,
    header: DemuxHeader,
    normalizer: FrameNormalizer,
    normalizer_for: (codec::frame::PixelFormat, codec::frame::ColorMetadata),
    filters: Arc<codec::filter::FilterChain>,
    workers: Vec<(
        SyncSender<RungMsg>,
        std::thread::JoinHandle<Result<WorkerDone>>,
    )>,
    rungs: Vec<Rung>,
    last: Option<VideoFrame>,
    frames: u64,
    repeated: u64,
    dropped_early: u64,
    dropped_behind: u64,
    output: Output,
    audio: Option<AudioOut<'a>>,
    hls: Option<(Arc<Mutex<HlsLive>>, HlsTiming)>,
    sink: Arc<dyn ProgressSink>,
    renditions: Vec<crate::fit::FittedRung>,
    kind: &'static str,
    source_dims: (u32, u32),
    source_fps: f64,
    /// The size the first picture came out of the normaliser at.
    normalized_dims: Option<(u32, u32)>,
}

impl<'a> Engine<'a> {
    fn start(
        first: &LiveVideo,
        spec: &'a OutputSpec,
        target: LiveTarget,
        sink: Arc<dyn ProgressSink>,
        name: &str,
        kind: &'static str,
        realtime: bool,
    ) -> Result<Engine<'a>> {
        let source_rate = first.frame_rate;
        let source_fps = if source_rate.0 > 0 && source_rate.1 > 0 {
            rate_f64(source_rate)
        } else {
            30.0
        };
        let header = DemuxHeader {
            codec: kind.into(),
            info: StreamInfo {
                codec: kind.into(),
                width: first.frame.width,
                height: first.frame.height,
                frame_rate: source_fps,
                duration: 0.0,
                pixel_format: first.frame.format,
                color_space: first.frame.color_space,
                total_frames: 0,
                bitrate: 0,
                color_metadata: first.color,
            },
            timescale: 90_000,
            rotation_degrees: 0,
            sample_aspect: (1, 1),
        };
        // Every rung's box made its size for this source, the rung policy
        // resolved on those sizes, and the source's colour and depth judged
        // against the output — as `run_job` does, before anything runs.
        let (fitted, renditions) = fit_to(spec, &header);
        let resolved = fitted.with_rung_policy_resolved();
        let sink = remap_rung_indices(sink, &renditions);
        resolved.check_source_colour(&header.info.color_metadata)?;
        resolved
            .check_source(header.info.color_metadata, header.info.pixel_format)
            .context("invalid OutputSpec")?;
        let frame_rate = output_rate(source_rate, resolved.max_frame_rate);
        let fps = rate_f64(frame_rate);
        let resolved = resolved.with_constant_rates_resolved(fps);
        // The engine keeps the spec it was handed for the audio path's
        // borrow; the rungs and per-rung knobs come from the resolved one.
        let rungs = resolved.rungs.clone();
        resolved.hooks.emit_probe(
            0,
            crate::hooks::MediaSummary::of_header(kind, &header, None),
        )?;
        sink.on_event(JobEvent::Started { rungs: rungs.len() });
        sink.on_event(JobEvent::Probed {
            codec: kind.into(),
            width: first.frame.width,
            height: first.frame.height,
            frame_rate: source_fps,
            audio_codec: None,
        });

        let filters = Arc::new(
            codec::filter::FilterChain::prepare(&resolved.filters)
                .context("preparing video filters")?,
        );
        let normalizer = normalizer_for(&resolved, &filters, first, &header)?;
        let limit = spec
            .live
            .duration
            .map(|d| (d * fps).round().max(1.0) as u64);
        let frames_total = limit;

        let (output_color, output_pixel_format) =
            resolved.resolve_output(header.info.color_metadata, header.info.pixel_format);
        let mut workers = Vec::with_capacity(rungs.len());
        let mut hls = None;
        let output = match &target {
            LiveTarget::Ndi(_) => Output::Ndi,
            _ => Output::Encode,
        };
        let threads = serial_threads_per_rung(rungs.len());
        let places = match output {
            Output::Encode => placement(&resolved, output_pixel_format)?,
            Output::Ndi => vec![(None, None); rungs.len()],
        };
        if let LiveTarget::Dir(root) = &target {
            std::fs::create_dir_all(root)
                .with_context(|| format!("creating {}", root.display()))?;
        }
        if let (LiveTarget::Dir(root), OutputMode::Hls { segment_seconds }) =
            (&target, &resolved.mode)
        {
            let timing = HlsTiming::new(*segment_seconds, fps);
            hls = Some((
                Arc::new(Mutex::new(HlsLive {
                    root: root.clone(),
                    frame_rate: fps,
                    video: vec![None; rungs.len()],
                    audio: Vec::new(),
                    master_written: false,
                })),
                timing,
            ));
        }
        for (index, rung) in rungs.iter().enumerate() {
            let (gpu_index, gpu_vendor) = places[index];
            let mut cfg = EncoderConfig {
                frame_rate: fps,
                pixel_format: output_pixel_format,
                color_metadata: output_color,
                gpu_index,
                gpu_vendor,
                codec: resolved.video_codec.codec(),
                threads,
                width: rung.width,
                height: rung.height,
                ..EncoderConfig::default()
            };
            rung.quality.apply(&mut cfg, fps);
            if let Some((_, timing)) = &hls {
                // Every segment opens on a keyframe.
                cfg.keyframe_interval =
                    (timing.segment_ticks / u64::from(timing.per_frame_ticks)).max(1) as u32;
            }
            tracing::info!(
                rung = %rung.label,
                width = rung.width,
                height = rung.height,
                codec = cfg.codec.label(),
                gpu = ?gpu_index,
                vendor = ?gpu_vendor,
                "live rung"
            );
            let work = RungWork {
                index,
                rung: rung.clone(),
                cfg,
                frame_rate,
                frames_total,
                sink: Arc::clone(&sink),
            };
            let (tx, rx) = mpsc::sync_channel::<RungMsg>(FRAME_CHANNEL_CAPACITY);
            let handle = match &target {
                LiveTarget::File(path) => {
                    let (container, path) = (resolved.container, path.clone());
                    spawn_worker(&rung.label, move || file_worker(work, rx, container, path))?
                }
                LiveTarget::Dir(root) => match &hls {
                    Some((live, timing)) => {
                        let (live, timing) = (Arc::clone(live), *timing);
                        spawn_worker(&rung.label, move || hls_worker(work, rx, timing, live))?
                    }
                    None => {
                        let container = resolved.container;
                        let path =
                            root.join(format!("{}.{}", rung.label, resolved.file_extension()));
                        spawn_worker(&rung.label, move || file_worker(work, rx, container, path))?
                    }
                },
                LiveTarget::Ndi(endpoint) => {
                    let stream = if rungs.len() == 1 {
                        endpoint.name.clone()
                    } else {
                        format!("{} ({})", endpoint.name, rung.label)
                    };
                    spawn_ndi(work, rx, endpoint.clone(), stream, !realtime, &rung.label)?
                }
            };
            workers.push((tx, handle));
        }

        Ok(Engine {
            spec,
            name: name.to_string(),
            frame_rate,
            t0: first.time,
            max_gap_frames: (fps * 10.0).ceil() as u64,
            limit,
            header,
            normalizer,
            normalizer_for: (first.frame.format, first.color),
            filters,
            workers,
            rungs,
            last: None,
            frames: 0,
            repeated: 0,
            dropped_early: 0,
            dropped_behind: 0,
            output,
            audio: None,
            hls,
            sink,
            renditions,
            kind,
            source_dims: (first.frame.width, first.frame.height),
            source_fps,
            normalized_dims: None,
        })
    }

    fn limit_reached(&self) -> bool {
        self.limit.is_some_and(|m| self.frames >= m)
    }

    fn room(&self) -> u64 {
        self.limit
            .map_or(u64::MAX, |m| m.saturating_sub(self.frames))
    }

    fn video(&mut self, v: LiveVideo) -> Result<()> {
        let (n, d) = self.frame_rate;
        let k = slot(v.time, self.t0, u64::from(n), u64::from(d));
        match place(k, self.frames, self.max_gap_frames) {
            Place::Drop => {
                self.dropped_early += 1;
                return Ok(());
            }
            Place::Reanchor => {
                tracing::warn!(
                    jump_frames = k - self.frames as i64,
                    "the source's clock jumped; re-anchoring the timeline"
                );
                self.t0 = v.time
                    - (i128::from(self.frames) * i128::from(TICKS_PER_SECOND) * i128::from(d)
                        / i128::from(n)) as i64;
            }
            Place::Write { repeats } => {
                if let Some(last) = self.last.clone() {
                    for _ in 0..repeats.min(self.room()) {
                        self.send_frame(last.clone())?;
                        self.repeated += 1;
                    }
                }
            }
        }
        if self.room() == 0 {
            return Ok(());
        }
        let fps = rate_f64(self.frame_rate);
        self.spec
            .hooks
            .emit_decoded_frame(0, self.frames, fps, &v.frame)?;
        if (v.frame.format, v.color) != self.normalizer_for {
            tracing::info!(format = ?v.frame.format, "the source's picture format changed");
            self.normalizer = normalizer_for(self.spec, &self.filters, &v, &self.header)?;
            self.normalizer_for = (v.frame.format, v.color);
        }
        let mut frame = self.normalizer.normalize(v.frame)?;
        // A source that changed size mid-stream is scaled to the size its
        // first picture came out at, which the rungs were fitted to.
        match self.normalized_dims {
            None => self.normalized_dims = Some((frame.width, frame.height)),
            Some((w, h)) if (frame.width, frame.height) != (w, h) => {
                frame = codec::colorspace::scale_frame(&frame, w, h)
                    .context("scaling a resized source picture to the job's size")?;
            }
            Some(_) => {}
        }
        self.spec
            .hooks
            .emit_encoder_frame(0, self.frames, fps, &frame)?;
        self.send_frame(frame)
    }

    fn send_frame(&mut self, frame: VideoFrame) -> Result<()> {
        self.frames += 1;
        self.last = Some(frame.clone());
        self.broadcast(|| RungMsg::Frame(frame.clone()))
    }

    /// Send to every rung still running; a rung whose worker failed is
    /// reported when the workers are joined.
    fn broadcast(&mut self, msg: impl Fn() -> RungMsg) -> Result<()> {
        let mut alive = 0;
        for (tx, _) in &self.workers {
            if tx.send(msg()).is_ok() {
                alive += 1;
            }
        }
        if alive == 0 && !self.workers.is_empty() {
            bail!("every rung failed");
        }
        Ok(())
    }

    fn audio(&mut self, a: LiveAudio) -> Result<()> {
        if self.spec.audio == AudioCodecPolicy::Drop || a.channels == 0 || a.sample_rate == 0 {
            return Ok(());
        }
        if self.audio.is_none() {
            self.audio = Some(self.open_audio(&a)?);
        }
        let out = self.audio.as_mut().expect("opened above");
        if a.sample_rate != out.rate || a.channels != out.channels {
            if !out.warned {
                tracing::warn!(
                    from = %format!("{} Hz {}ch", out.rate, out.channels),
                    to = %format!("{} Hz {}ch", a.sample_rate, a.channels),
                    "the source's audio changed shape; its audio is left out from here"
                );
                out.warned = true;
            }
            return Ok(());
        }
        let ch = usize::from(a.channels);
        let mut samples = a.samples;
        let s = slot(a.time, self.t0, u64::from(out.rate), 1);
        let drift = s - out.written as i64;
        let tolerance = i64::from(out.rate) / 25; // 40 ms
        let max_gap = 10 * i64::from(out.rate);
        let first = out.written == 0;
        // Never past the pictures' end when the job has a length.
        let cap = self.limit.map(|m| samples_in(m, self.frame_rate, out.rate));
        if (first && drift > 0 && drift <= max_gap) || (drift > tolerance && drift <= max_gap) {
            let fill = cap.map_or(drift as u64, |c| {
                (drift as u64).min(c.saturating_sub(out.written))
            });
            let silence = vec![0.0f32; fill as usize * ch];
            self.audio_out(silence)?;
        } else if drift < 0 && (first || drift < -tolerance) {
            let cut = ((-drift) as usize * ch).min(samples.len());
            samples.drain(..cut);
        }
        let out = self.audio.as_ref().expect("opened above");
        if let Some(c) = cap {
            let room = c.saturating_sub(out.written) as usize * ch;
            samples.truncate(room);
        }
        if !samples.is_empty() {
            self.audio_out(samples)?;
        }
        Ok(())
    }

    fn open_audio(&mut self, a: &LiveAudio) -> Result<AudioOut<'a>> {
        let codec = self.spec.audio_encode_codec();
        let mut renditions = Vec::new();
        if matches!(self.output, Output::Encode) {
            let main = AudioRequest::of(self.spec);
            let hls = self.hls.is_some();
            // HLS with a stereo fallback beside a surround source: the
            // stereo downmix first (the group's default), the surround beside.
            if hls && self.spec.audio_stereo_fallback && a.channels > 2 {
                let mut stereo = main;
                stereo.channels = AudioChannels::Stereo;
                renditions.push(AudioRendition {
                    state: Some(EncodeState::new(stereo, codec)),
                    announced: false,
                    info: None,
                    cmaf: None,
                    relative_dir: "audio-stereo",
                    name: "Stereo",
                    index: 0,
                });
                renditions.push(AudioRendition {
                    state: Some(EncodeState::new(main, codec)),
                    announced: false,
                    info: None,
                    cmaf: None,
                    relative_dir: "audio",
                    name: "Surround",
                    index: 1,
                });
            } else {
                renditions.push(AudioRendition {
                    state: Some(EncodeState::new(main, codec)),
                    announced: false,
                    info: None,
                    cmaf: None,
                    relative_dir: "audio",
                    name: "Audio",
                    index: 0,
                });
            }
            if let Some((live, _)) = &self.hls {
                live.lock().unwrap().audio = vec![None; renditions.len()];
            }
        }
        tracing::info!(
            ?codec,
            channels = a.channels,
            rate = a.sample_rate,
            "live audio"
        );
        Ok(AudioOut {
            renditions,
            rate: a.sample_rate,
            channels: a.channels,
            written: 0,
            warned: false,
        })
    }

    /// `samples` (interleaved, the next on the timeline) to every audio
    /// output.
    fn audio_out(&mut self, samples: Vec<f32>) -> Result<()> {
        let out = self.audio.as_mut().expect("opened");
        let ch = usize::from(out.channels);
        let n = (samples.len() / ch) as u64;
        let frame = AudioFrame {
            samples,
            sample_rate: out.rate,
            channels: out.channels,
            pts: (i128::from(out.written) * 1_000_000 / i128::from(out.rate)) as i64,
        };
        out.written += n;
        if matches!(self.output, Output::Ndi) {
            let filtered = codec::audio::filter::apply_chain(&frame, &self.spec.audio_filters)
                .context("audio filter chain")?;
            let pcm = Arc::new(filtered);
            return self.broadcast(|| RungMsg::Pcm(Arc::clone(&pcm)));
        }
        let mut sends: Vec<RungMsg> = Vec::new();
        let hls = self.hls.clone();
        for r in &mut out.renditions {
            let state = r.state.as_mut().expect("present until finish");
            let mut packets = Vec::new();
            state.encode(&frame, None, &mut packets)?;
            if !r.announced {
                let Some(info) = state.describe(packets.first().map(|(p, _)| p.as_slice()))? else {
                    // Nothing to describe the track by yet (AC-3 waits for
                    // its first packet): hold on to these.
                    if !packets.is_empty() {
                        bail!("the audio encoder wrote packets it cannot describe");
                    }
                    continue;
                };
                r.announced = true;
                r.info = Some(info.clone());
                match &hls {
                    Some((live, _)) => {
                        let root = live.lock().unwrap().root.clone();
                        let mut m = CmafAudioMuxer::new(root.join(r.relative_dir), info)?;
                        let (pre_skip, _) = state.timing().unwrap_or((0, 48_000));
                        m.set_edit(container::edit::TrackEdit {
                            delay: 0,
                            media_time: u64::from(pre_skip),
                            duration: None,
                        })?;
                        r.cmaf = Some(m);
                    }
                    None => sends.push(RungMsg::AudioTrack(info)),
                }
            }
            match (&hls, r.cmaf.as_mut()) {
                (Some((live, timing)), Some(m)) => {
                    for (data, dur) in packets {
                        m.add_packet(data, dur)?;
                    }
                    let target = u64::from(timing.target_duration) * u64::from(m.timescale());
                    if m.pending_duration_ticks() >= target {
                        let info = r.info.as_ref().expect("described above");
                        flush_audio(
                            m,
                            info,
                            r.relative_dir,
                            r.name,
                            r.index,
                            live,
                            timing.target_duration,
                        )?;
                    }
                }
                _ => {
                    for (data, dur) in packets {
                        sends.push(RungMsg::AudioPacket(Arc::new(data), dur));
                    }
                }
            }
        }
        for msg in sends {
            match msg {
                RungMsg::AudioTrack(info) => {
                    self.broadcast(|| RungMsg::AudioTrack(info.clone()))?
                }
                RungMsg::AudioPacket(data, dur) => {
                    self.broadcast(|| RungMsg::AudioPacket(Arc::clone(&data), dur))?
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn finish(mut self, ended: LiveEnd, started: Instant) -> Result<JobOutput> {
        // The audio: flushed, its priming hidden and its end on the
        // pictures' (or the last real sample's, when that is sooner).
        let video_samples = |rate: u32, frames: u64, fr: (u32, u32)| samples_in(frames, fr, rate);
        let mut audio_handling = "no audio".to_string();
        let mut audio_codecs = None;
        let mut audio_specs: Vec<AudioVariantSpec> = Vec::new();
        let mut audio_samples = 0;
        if let Some(mut out) = self.audio.take() {
            audio_samples = out.written;
            let in_rate = out.rate;
            let hls = self.hls.clone();
            let mut sends = Vec::new();
            for r in &mut out.renditions {
                let mut tail = Vec::new();
                let Some(encoded) = r.state.take().expect("present").finish(&mut tail)? else {
                    continue;
                };
                let presented = video_samples(in_rate, self.frames, self.frame_rate)
                    .min(encoded.encoded_samples);
                let edit = container::edit::TrackEdit {
                    delay: 0,
                    media_time: u64::from(encoded.pre_skip),
                    duration: Some(container::edit::rescale_round(
                        presented,
                        encoded.out_rate,
                        encoded.in_rate,
                    )),
                };
                audio_handling = format!(
                    "live {} {}ch → {}",
                    "pcm",
                    out.channels,
                    audio_codec_string(&encoded.info)
                );
                audio_codecs.get_or_insert_with(|| audio_codec_string(&encoded.info));
                match (&hls, r.cmaf.take()) {
                    (Some(_), Some(mut m)) => {
                        for (data, dur) in tail {
                            m.add_packet(data, dur)?;
                        }
                        m.flush_segment()?;
                        let manifest = m.finalize()?;
                        let spec = AudioVariantSpec {
                            codec_string: audio_codec_string(&encoded.info),
                            channels: encoded.info.channels,
                            sample_rate: encoded.info.sample_rate,
                            relative_dir: r.relative_dir.to_string(),
                            language: "und".into(),
                            name: r.name.to_string(),
                            manifest,
                        };
                        audio_specs.push(spec);
                    }
                    _ => {
                        if !r.announced {
                            sends.push(RungMsg::AudioTrack(encoded.info.clone()));
                        }
                        for (data, dur) in tail {
                            sends.push(RungMsg::AudioPacket(Arc::new(data), dur));
                        }
                        sends.push(RungMsg::AudioEdit(edit));
                    }
                }
            }
            for msg in sends {
                match msg {
                    RungMsg::AudioTrack(info) => {
                        self.broadcast(|| RungMsg::AudioTrack(info.clone()))?
                    }
                    RungMsg::AudioPacket(data, dur) => {
                        self.broadcast(|| RungMsg::AudioPacket(Arc::clone(&data), dur))?
                    }
                    RungMsg::AudioEdit(edit) => self.broadcast(|| RungMsg::AudioEdit(edit))?,
                    _ => {}
                }
            }
            if matches!(self.output, Output::Ndi) {
                audio_handling = format!("live pcm {}ch → NDI", out.channels);
            }
        }

        // The rungs: their channels closed, each worker finishes its output.
        let mut outputs = Vec::new();
        let mut manifests = Vec::new();
        let workers = std::mem::take(&mut self.workers);
        for (i, (tx, handle)) in workers.into_iter().enumerate() {
            drop(tx);
            let rung = &self.rungs[i];
            match handle.join() {
                Ok(Ok(done)) => {
                    outputs.push(done.output);
                    if let Some(m) = done.hls {
                        manifests.push(m);
                    }
                }
                Ok(Err(e)) => report_rung_error(self.sink.as_ref(), i, rung, &e),
                Err(_) => report_rung_error(
                    self.sink.as_ref(),
                    i,
                    rung,
                    &anyhow::anyhow!("the rung's worker panicked"),
                ),
            }
        }
        if outputs.is_empty() {
            bail!("all {} rung(s) failed", self.rungs.len());
        }

        // HLS: the finished package over the live playlists.
        let (mut hls_root, mut master_playlist) = (None, None);
        if let Some((live, timing)) = &self.hls {
            let root = live.lock().unwrap().root.clone();
            let fps = live.lock().unwrap().frame_rate;
            let mut video_specs = Vec::new();
            for rm in &manifests {
                super::pump::settle_sample_entry(rm);
                let declared = self.rungs.get(rm.rung_index).and_then(|r| {
                    codec::encode::tuning::ConstantRate::from_overrides(&r.quality.overrides)
                        .map(|c| c.bps)
                });
                let bytes = rm.manifest.byte_size();
                video_specs.push(super::pump::build_video_variant_spec(
                    rm, fps, bytes, declared,
                ));
            }
            super::pump::add_rendition_rates(&mut video_specs, &audio_specs, &[]);
            let paths = container::hls::write_hls_package(
                &root,
                &video_specs,
                &audio_specs,
                &[],
                timing.target_duration,
            )
            .context("writing the HLS package")?;
            hls_root = Some(root);
            master_playlist = Some(paths.master_path);
        }

        let completed = outputs.len();
        self.sink.on_event(JobEvent::Finished {
            rungs_completed: completed,
            rungs_failed: self.rungs.len().saturating_sub(completed),
        });
        let stats = LiveStats {
            source: self.name.clone(),
            frames: self.frames,
            frame_rate: self.frame_rate,
            repeated: self.repeated,
            dropped_early: self.dropped_early,
            dropped_behind: self.dropped_behind,
            audio_samples,
            ended,
        };
        tracing::info!(
            frames = stats.frames,
            seconds = stats.seconds(),
            repeated = stats.repeated,
            dropped_early = stats.dropped_early,
            dropped_behind = stats.dropped_behind,
            ended = ended.as_str(),
            "live job done"
        );
        Ok(JobOutput {
            rungs: outputs,
            hls_root,
            master_playlist,
            source_codec: self.kind.to_string(),
            source_dims: self.source_dims,
            source_frame_rate: self.source_fps,
            audio_handling,
            audio_codecs,
            renditions: self.renditions,
            elapsed: started.elapsed(),
            hooks: crate::hooks::HookReport::default(),
            live: Some(stats),
        })
    }
}

/// Flush an HLS audio rendition's segment and keep its playlist (and, the
/// first time, the master) current.
fn flush_audio(
    m: &mut CmafAudioMuxer,
    info: &AudioInfo,
    relative_dir: &str,
    name: &str,
    index: usize,
    live: &Arc<Mutex<HlsLive>>,
    target_duration: u32,
) -> Result<()> {
    if m.flush_segment()?.is_none() {
        return Ok(());
    }
    let manifest = CmafTrackManifest {
        init_path: m.init_path().to_path_buf(),
        segments: m.segments().to_vec(),
        timescale: m.timescale(),
    };
    let mut l = live.lock().unwrap();
    let playlist = l.root.join(relative_dir).join("audio.m3u8");
    container::hls::write_live_media_playlist(&playlist, &manifest, target_duration)?;
    if l.audio.get(index).is_some_and(Option::is_none) {
        l.audio[index] = Some(AudioVariantSpec {
            codec_string: audio_codec_string(info),
            channels: info.channels,
            sample_rate: info.sample_rate,
            relative_dir: relative_dir.to_string(),
            language: "und".into(),
            name: name.to_string(),
            manifest,
        });
        l.update();
    }
    Ok(())
}

fn spawn_worker(
    label: &str,
    f: impl FnOnce() -> Result<WorkerDone> + Send + 'static,
) -> Result<std::thread::JoinHandle<Result<WorkerDone>>> {
    std::thread::Builder::new()
        .name(format!("live-rung-{label}"))
        .spawn(f)
        .context("starting a rung worker")
}

#[cfg(feature = "ndi")]
fn spawn_ndi(
    work: RungWork,
    rx: Receiver<RungMsg>,
    endpoint: NdiEndpoint,
    stream: String,
    clock_video: bool,
    label: &str,
) -> Result<std::thread::JoinHandle<Result<WorkerDone>>> {
    spawn_worker(label, move || {
        ndi_worker(work, rx, endpoint, stream, clock_video)
    })
}

#[cfg(not(feature = "ndi"))]
fn spawn_ndi(
    _work: RungWork,
    _rx: Receiver<RungMsg>,
    endpoint: NdiEndpoint,
    _stream: String,
    _clock_video: bool,
    _label: &str,
) -> Result<std::thread::JoinHandle<Result<WorkerDone>>> {
    bail!(
        "ndi://{} is an NDI output, and this build has no NDI: rebuild with `--features ndi`",
        endpoint.name
    )
}

/// The decode pump's per-frame work for pictures like `v`.
fn normalizer_for(
    spec: &OutputSpec,
    filters: &Arc<codec::filter::FilterChain>,
    v: &LiveVideo,
    header: &DemuxHeader,
) -> Result<FrameNormalizer> {
    let mut header = header.clone();
    header.info.width = v.frame.width;
    header.info.height = v.frame.height;
    header.info.pixel_format = v.frame.format;
    header.info.color_space = v.frame.color_space;
    header.info.color_metadata = v.color;
    let cfg = DecodePumpConfig::for_source(&header, spec, Arc::clone(filters), None);
    FrameNormalizer::new(&cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_rounds_to_the_nearest_frame_both_ways() {
        let t = |s: f64| (s * TICKS_PER_SECOND as f64) as i64;
        assert_eq!(slot(t(0.0), 0, 30000, 1001), 0);
        assert_eq!(slot(t(0.0333), 0, 30000, 1001), 1);
        assert_eq!(slot(t(0.0160), 0, 30000, 1001), 0);
        assert_eq!(slot(t(0.0170), 0, 30000, 1001), 1);
        assert_eq!(slot(t(10.01), 0, 30000, 1001), 300);
        assert_eq!(slot(-t(0.0334), 0, 30000, 1001), -1);
        assert_eq!(slot(t(1.0), 0, 48_000, 1), 48_000);
    }

    #[test]
    fn a_picture_is_written_dropped_or_reanchored_by_where_it_falls() {
        assert_eq!(place(5, 5, 300), Place::Write { repeats: 0 });
        assert_eq!(place(8, 5, 300), Place::Write { repeats: 3 });
        assert_eq!(place(4, 5, 300), Place::Drop);
        assert_eq!(place(400, 5, 300), Place::Reanchor);
        assert_eq!(place(-400, 5, 300), Place::Reanchor);
    }

    #[test]
    fn a_max_fps_cap_below_the_source_is_the_output_rate() {
        assert_eq!(output_rate((60000, 1001), Some(30.0)), (30, 1));
        assert_eq!(output_rate((25, 1), Some(30.0)), (25, 1));
        assert_eq!(output_rate((0, 1), None), (30, 1));
        assert_eq!(output_rate((50, 1), None), (50, 1));
    }
}
