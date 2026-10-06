//! Record a live source into a file: an NDI source (or any [`LiveSource`])
//! → the job engine's per-frame normalisation → a video encoder and an
//! audio encoder → an MP4, QuickTime or WebM file.
//!
//! ```text
//! capture thread: source.next_event → bounded queue (video dropped when full)
//! this thread:    queue → timeline → normalise (+ scale) → encoder → muxer
//!                                  → audio encoder ────────────────→ muxer
//! ```
//!
//! **The timeline.** A file has a fixed frame rate (the source's declared
//! one) and a gapless audio track; a live source has neither. Each picture
//! is placed at the frame its timestamp falls on, counted from the first
//! picture: a picture that falls on a frame already written is dropped, and
//! a gap (frames the source skipped, or the queue dropped while the encoder
//! was behind) is filled by repeating the last picture. Audio is placed the
//! same way: a gap of more than 40 ms is filled with silence and an overlap
//! trimmed, smaller jitter absorbed. So the audio stays in step with the
//! picture however the encoder keeps up, and the file is as long as the
//! time recorded. A jump of more than ten seconds either way (a sender
//! restarting, its clock reset) re-anchors the timeline instead of filling it.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use codec::audio::{AudioCodec, AudioEncoder, AudioEncoderConfig, AudioFrame};
use codec::encode::{self, EncoderBackend, EncoderConfig};
use codec::frame::{ColorMetadata, StreamInfo, VideoCodec, VideoFrame};
use container::AudioInfo;
use container::mux::Av1Mp4Muxer;
use container::webm::WebmMuxer;

use super::live::{LiveAudio, LiveEvent, LiveSource, LiveVideo, SourceLost, TICKS_PER_SECOND};
use crate::decode_pump::{DecodePumpConfig, FrameNormalizer};
use crate::spec::{BitDepth, ColorPolicy, Container, OutputSpec, Rung, VideoCodecPolicy};

/// What the recording's audio is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecordAudio {
    /// Opus (the default; the codec every container here takes).
    #[default]
    Auto,
    /// This codec: Opus anywhere, AAC (`aac`, `he-aac`) into MP4 / QuickTime.
    Codec(AudioCodec),
    /// No audio track.
    Drop,
}

/// How a recording is made.
#[derive(Debug, Clone)]
pub struct RecordOptions {
    /// The file to write. It appears when the recording ends; until then
    /// the data is spooled beside it.
    pub output: PathBuf,
    /// `None`: from the output's extension (`.mov`, `.webm`), else the
    /// codec's default container.
    pub container: Option<Container>,
    pub video_codec: VideoCodecPolicy,
    /// Perceptual quality target (`None`: the encoders' default).
    pub target: Option<codec::encode::tuning::QualityTarget>,
    /// Constant rate factor; wins over `target`.
    pub crf: Option<u8>,
    /// Keyframe interval in seconds.
    pub gop_seconds: f64,
    /// SDR (tonemapping an HDR source, the default), passthrough or HDR.
    pub color: ColorPolicy,
    pub bit_depth: BitDepth,
    pub audio: RecordAudio,
    /// Audio bitrate in bits per second; `None` for the codec's default.
    pub audio_bitrate: Option<u32>,
    /// Stop after this much has been recorded.
    pub duration: Option<Duration>,
    /// Stop after this many frames.
    pub max_frames: Option<u64>,
    /// How long to wait for the first picture.
    pub start_timeout: Duration,
    /// End the recording when no picture arrives for this long (zero: wait
    /// for ever).
    pub idle_timeout: Duration,
    /// Set to end the recording (Ctrl+C); what was recorded is kept.
    pub stop: Option<Arc<AtomicBool>>,
    /// Force an encoder backend (`None`: the GPU-first chain).
    pub encoder_backend: Option<EncoderBackend>,
    /// The GPU the encoder runs on.
    pub gpu_index: Option<u32>,
    /// Video filters, as `--filter` takes them.
    pub filters: Vec<codec::filter::VideoFilter>,
}

impl RecordOptions {
    pub fn new(output: impl Into<PathBuf>) -> Self {
        Self {
            output: output.into(),
            container: None,
            video_codec: VideoCodecPolicy::default(),
            target: None,
            crf: None,
            gop_seconds: 2.0,
            color: ColorPolicy::default(),
            bit_depth: BitDepth::default(),
            audio: RecordAudio::Auto,
            audio_bitrate: None,
            duration: None,
            max_frames: None,
            start_timeout: Duration::from_secs(15),
            idle_timeout: Duration::from_secs(10),
            stop: None,
            encoder_backend: None,
            gpu_index: None,
            filters: Vec::new(),
        }
    }

    fn container(&self) -> Container {
        if let Some(c) = self.container {
            return c;
        }
        match self
            .output
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("mov") => Container::Mov,
            Some("webm") => Container::WebM,
            Some("mp4" | "m4v") => Container::Mp4,
            _ => self.video_codec.default_container(),
        }
    }
}

/// Why a recording ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    /// `duration` or `max_frames` reached.
    Limit,
    /// The stop flag was set.
    Stopped,
    /// No picture for `idle_timeout`.
    Idle,
    /// The source went away.
    SourceLost,
    /// A finite source ended.
    SourceEnded,
}

/// A running recording's counters, reported about once a second.
#[derive(Debug, Clone, Default)]
pub struct RecordProgress {
    /// Frames written (repeats included).
    pub frames: u64,
    /// Seconds of output.
    pub seconds: f64,
    /// Frames written as a repeat of the one before, to fill a gap.
    pub repeated: u64,
    /// Pictures dropped: early ones, and ones the queue had no room for.
    pub dropped: u64,
    /// Of those, the ones dropped because the encoder was behind.
    pub dropped_behind: u64,
    /// Output frames per wall-clock second since the first picture.
    pub fps: f64,
}

/// What a recording made.
#[derive(Debug, Clone)]
pub struct RecordOutcome {
    pub output: PathBuf,
    pub source: String,
    pub width: u32,
    pub height: u32,
    pub frame_rate: (u32, u32),
    pub video_codec: VideoCodec,
    /// `"opus 2ch 48000 Hz"`, or `None` for no audio track.
    pub audio: Option<String>,
    pub progress: RecordProgress,
    pub ended: EndReason,
    pub elapsed: Duration,
}

/// Record `source` as `options` say. `progress` is called about once a
/// second while recording.
pub fn record<S: LiveSource + 'static>(
    source: S,
    options: &RecordOptions,
    mut progress: impl FnMut(&RecordProgress),
) -> Result<RecordOutcome> {
    let started = Instant::now();
    let name = source.name();
    let container = options.container();
    check_combination(container, options)?;
    if let Some(dir) = options
        .output
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
    {
        anyhow::ensure!(
            dir.is_dir(),
            "the output directory {} does not exist",
            dir.display()
        );
    }

    let halt = Arc::new(AtomicBool::new(false));
    let queue = Arc::new(Queue::default());
    let (tx, rx) = channel::<Message>();
    let capture = {
        let (halt, queue) = (Arc::clone(&halt), Arc::clone(&queue));
        std::thread::Builder::new()
            .name("ndi-capture".into())
            .spawn(move || capture_loop(source, tx, &halt, &queue))
            .context("starting the capture thread")?
    };

    let result = run(&rx, options, container, &queue, &mut progress);
    halt.store(true, Ordering::Relaxed);
    drop(rx);
    let _ = capture.join();
    let (recorder, ended) = result?;
    let (outcome_progress, audio, (width, height), frame_rate) =
        recorder.finish(&options.output)?;
    Ok(RecordOutcome {
        output: options.output.clone(),
        source: name,
        width,
        height,
        frame_rate,
        video_codec: options.video_codec.codec(),
        audio,
        progress: outcome_progress,
        ended,
        elapsed: started.elapsed(),
    })
}

/// Pictures the queue holds before it drops them: about a second at 60 fps.
const QUEUE_PICTURES: usize = 60;

/// The capture thread's bookkeeping. Pictures are bounded — one the encoder
/// has no room for is dropped and its slot filled by a repeat — and audio is
/// not: it is small, cheap to encode, and must not be lost, nor wait behind
/// pictures (a shared bound let audio fill the queue and starve the video).
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
) {
    while !halt.load(Ordering::Relaxed) {
        let msg = match source.next_event(Duration::from_millis(100)) {
            Ok(None) => continue,
            Ok(Some(LiveEvent::Video(v))) => {
                if queue.pictures.load(Ordering::Acquire) >= QUEUE_PICTURES {
                    queue.behind.fetch_add(1, Ordering::Relaxed);
                    continue;
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

fn check_combination(container: Container, options: &RecordOptions) -> Result<()> {
    let codec = options.video_codec.codec();
    match container {
        Container::Mp4 | Container::Mov => {}
        Container::WebM => {
            if !matches!(codec, VideoCodec::Vp8 | VideoCodec::Vp9) {
                bail!(
                    "a WebM file carries VP8 or VP9 video, not {}: use --codec vp9, or an .mp4 output",
                    codec.label()
                );
            }
            if let RecordAudio::Codec(c) = options.audio
                && c != AudioCodec::Opus
            {
                bail!("a WebM file carries Opus audio here, not {c:?}");
            }
        }
        other => bail!("an NDI recording is an MP4, QuickTime or WebM file, not {other:?}"),
    }
    if let RecordAudio::Codec(c) = options.audio
        && !matches!(c, AudioCodec::Opus | AudioCodec::Aac | AudioCodec::HeAac)
    {
        bail!("an NDI recording's audio is Opus or AAC (aac, he-aac), not {c:?}");
    }
    Ok(())
}

fn run(
    rx: &Receiver<Message>,
    options: &RecordOptions,
    container: Container,
    queue: &Queue,
    progress: &mut dyn FnMut(&RecordProgress),
) -> Result<(Recorder, EndReason)> {
    let stopped = || {
        options
            .stop
            .as_ref()
            .is_some_and(|s| s.load(Ordering::Relaxed))
    };
    let wait_from = Instant::now();
    let mut pending_audio: Vec<LiveAudio> = Vec::new();
    // Until the first picture: keep the audio that arrives with it (it is
    // placed against the picture's time), give up after `start_timeout`.
    let mut recorder = loop {
        if stopped() {
            bail!("stopped before the source sent a picture");
        }
        let left = options.start_timeout.saturating_sub(wait_from.elapsed());
        if left.is_zero() {
            bail!(
                "no picture from the source within {:.0} s{}",
                options.start_timeout.as_secs_f64(),
                if pending_audio.is_empty() {
                    ""
                } else {
                    " (it sends audio only? a recording needs video)"
                }
            );
        }
        match rx.recv_timeout(left.min(Duration::from_millis(100))) {
            Ok(Message::Event(LiveEvent::Video(v))) => {
                queue.pictures.fetch_sub(1, Ordering::AcqRel);
                let mut r = Recorder::start(&v, options, container)?;
                r.video(v)?;
                for a in pending_audio.drain(..) {
                    r.audio(a)?;
                }
                break r;
            }
            Ok(Message::Event(LiveEvent::Audio(a))) => {
                pending_audio.push(a);
                // Keep a couple of seconds at most.
                if pending_audio.len() > 200 {
                    pending_audio.remove(0);
                }
            }
            Ok(Message::Event(LiveEvent::End)) => {
                bail!("the source ended before sending a picture")
            }
            Ok(Message::Lost) => bail!("the source went away before sending a picture"),
            Ok(Message::Failed(e)) => return Err(e.context("receiving from the source")),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => bail!("the capture thread ended"),
        }
    };

    let mut last_picture = Instant::now();
    let mut last_report = Instant::now();
    let ended = loop {
        recorder.dropped_behind = queue.behind.load(Ordering::Relaxed);
        if last_report.elapsed() >= Duration::from_secs(1) {
            progress(&recorder.progress());
            last_report = Instant::now();
        }
        if recorder.limit_reached(options) {
            break EndReason::Limit;
        }
        if stopped() {
            break EndReason::Stopped;
        }
        if !options.idle_timeout.is_zero() && last_picture.elapsed() >= options.idle_timeout {
            tracing::warn!(
                seconds = options.idle_timeout.as_secs_f64(),
                "no picture from the source; ending the recording"
            );
            break EndReason::Idle;
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Message::Event(LiveEvent::Video(v))) => {
                queue.pictures.fetch_sub(1, Ordering::AcqRel);
                last_picture = Instant::now();
                recorder.video(v)?;
            }
            Ok(Message::Event(LiveEvent::Audio(a))) => recorder.audio(a)?,
            Ok(Message::Event(LiveEvent::End)) => break EndReason::SourceEnded,
            Ok(Message::Lost) => {
                tracing::warn!("the source went away; keeping what was recorded");
                break EndReason::SourceLost;
            }
            Ok(Message::Failed(e)) => return Err(e.context("receiving from the source")),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break EndReason::SourceEnded,
        }
    };
    recorder.dropped_behind = queue.behind.load(Ordering::Relaxed);
    Ok((recorder, ended))
}

/// The output file, by container.
enum LiveMuxer {
    Mp4(Box<Av1Mp4Muxer>),
    WebM(Box<WebmMuxer>),
}

impl LiveMuxer {
    fn add_packet(&mut self, p: codec::encode::EncodedPacket) -> Result<()> {
        match self {
            Self::Mp4(m) => m.add_packet(p),
            Self::WebM(m) => m.add_packet(p),
        }
    }

    fn with_audio(&mut self, info: AudioInfo) -> Result<()> {
        match self {
            Self::Mp4(m) => m.with_audio(info).map(|_| ()),
            Self::WebM(m) => m.with_audio(info).map(|_| ()),
        }
    }

    fn add_audio(&mut self, data: &[u8], duration: u32) -> Result<()> {
        match self {
            Self::Mp4(m) => m.add_audio_sample(data, 0, duration),
            Self::WebM(m) => m.add_audio_sample(data, duration),
        }
    }

    fn set_audio_edit(&mut self, edit: container::edit::TrackEdit) {
        match self {
            Self::Mp4(m) => {
                m.set_audio_edit(edit);
            }
            Self::WebM(m) => {
                m.set_audio_edit(edit);
            }
        }
    }

    /// Write the file: to a spool file beside `output`, then renamed over
    /// it, so a reader never sees half a file.
    fn finalize(self, output: &Path) -> Result<()> {
        let mut spool = output.as_os_str().to_owned();
        spool.push(".partial");
        let spool = PathBuf::from(spool);
        let written = match self {
            Self::Mp4(m) => m.finalize_to_file(&spool),
            Self::WebM(m) => m
                .finalize()
                .and_then(|bytes| std::fs::write(&spool, bytes).map_err(Into::into)),
        };
        if let Err(e) = written {
            let _ = std::fs::remove_file(&spool);
            return Err(e.context("writing the recording"));
        }
        std::fs::rename(&spool, output)
            .with_context(|| format!("moving the recording to {}", output.display()))
    }
}

/// The timeline's slot for a time: the frame (or sample) `time` falls on,
/// counted from `t0`, at `rate_num / rate_den` per second.
fn slot(time: i64, t0: i64, rate_num: u64, rate_den: u64) -> i64 {
    let dt = (time - t0) as i128;
    let num = dt * rate_num as i128;
    let den = TICKS_PER_SECOND as i128 * rate_den as i128;
    // Round half away from zero.
    let q = (2 * num + den.signum() * num.signum() * den) / (2 * den);
    q as i64
}

/// Samples at `rate` in `frames` frames at `frame_rate`.
fn slot_count(frames: u64, frame_rate: (u32, u32), rate: u32) -> u64 {
    (u128::from(frames) * u128::from(rate) * u128::from(frame_rate.1)
        / u128::from(frame_rate.0.max(1))) as u64
}

/// What to do with a picture that falls on frame `k` when `n` frames have
/// been written.
#[derive(Debug, PartialEq, Eq)]
enum Place {
    /// Write it as frame `n`, after repeating the last frame this many times.
    Write { repeats: u64 },
    /// Its frame is written already.
    Drop,
    /// Too far from the timeline: re-anchor it so this is frame `n`.
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

struct AudioOut {
    encoder: Box<dyn AudioEncoder>,
    codec: AudioCodec,
    rate: u32,
    channels: u8,
    /// Channels kept from each source frame (the first `channels`).
    source_channels: u8,
    /// Samples (per channel) placed on the timeline so far.
    written: u64,
    warned_rate: bool,
}

struct Recorder {
    width: u32,
    height: u32,
    frame_rate: (u32, u32),
    fps: f64,
    t0: i64,
    max_gap_frames: u64,
    /// The frame count `max_frames` / `duration` stop at.
    limit: Option<u64>,
    normalizer: FrameNormalizer,
    /// The source layout the normaliser was built for.
    normalizer_for: (codec::frame::PixelFormat, ColorMetadata),
    spec: OutputSpec,
    filters: Arc<codec::filter::FilterChain>,
    encoder: Box<dyn encode::Encoder>,
    muxer: LiveMuxer,
    last: Option<VideoFrame>,
    frames: u64,
    repeated: u64,
    dropped_early: u64,
    dropped_behind: u64,
    packets: u64,
    audio: Option<AudioOut>,
    audio_policy: RecordAudio,
    audio_bitrate: Option<u32>,
    started: Instant,
}

impl Recorder {
    fn start(first: &LiveVideo, options: &RecordOptions, container: Container) -> Result<Self> {
        let (width, height) = (first.frame.width, first.frame.height);
        let frame_rate = match first.frame_rate {
            (n, d) if n > 0 && d > 0 => (n, d),
            _ => {
                tracing::warn!("the source declares no frame rate; recording at 30 fps");
                (30, 1)
            }
        };
        let fps = f64::from(frame_rate.0) / f64::from(frame_rate.1);
        let mut spec = OutputSpec::single_file(vec![Rung::new(width, height)]);
        spec.video_codec = options.video_codec;
        spec.color = options.color;
        spec.bit_depth = options.bit_depth;
        spec.filters = options.filters.clone();
        spec.check_source(first.color, first.frame.format)
            .context("the recording's colour and depth")?;
        let (output_color, output_pixel_format) =
            spec.resolve_output(first.color, first.frame.format);
        let filters = Arc::new(
            codec::filter::FilterChain::prepare(&spec.filters)
                .context("preparing video filters")?,
        );
        let normalizer = normalizer_for(&spec, &filters, first, fps)?;

        let codec = options.video_codec.codec();
        let config = EncoderConfig {
            width,
            height,
            frame_rate: fps,
            keyframe_interval: ((fps * options.gop_seconds).round() as u32).max(1),
            pixel_format: output_pixel_format,
            color_metadata: output_color,
            codec,
            target: options.target.unwrap_or_default(),
            quality: options.crf.unwrap_or(encode::AUTO_FROM_TARGET),
            gpu_index: options.gpu_index,
            ..EncoderConfig::default()
        };
        let encoder = encode::select_encoder(config, options.encoder_backend)
            .with_context(|| format!("no {} encoder for the recording", codec.label()))?;
        let mut muxer = match container {
            Container::WebM => {
                LiveMuxer::WebM(Box::new(WebmMuxer::new(width, height, fps, codec)?))
            }
            _ => {
                let mut m = Av1Mp4Muxer::new_with_codec(width, height, fps, codec)?;
                m.set_quicktime(container == Container::Mov);
                LiveMuxer::Mp4(Box::new(m))
            }
        };
        match &mut muxer {
            LiveMuxer::Mp4(m) => {
                m.set_color_metadata(output_color);
            }
            LiveMuxer::WebM(m) => {
                m.set_color_metadata(output_color);
            }
        }
        tracing::info!(
            width,
            height,
            fps,
            codec = codec.label(),
            ?container,
            "recording"
        );
        Ok(Self {
            width,
            height,
            frame_rate,
            fps,
            t0: first.time,
            max_gap_frames: (fps * 10.0).ceil() as u64,
            limit: [
                options.max_frames,
                options
                    .duration
                    .map(|d| (d.as_secs_f64() * fps).round() as u64),
            ]
            .into_iter()
            .flatten()
            .min(),
            normalizer,
            normalizer_for: (first.frame.format, first.color),
            spec,
            filters,
            encoder,
            muxer,
            last: None,
            frames: 0,
            repeated: 0,
            dropped_early: 0,
            dropped_behind: 0,
            packets: 0,
            audio: None,
            audio_policy: options.audio,
            audio_bitrate: options.audio_bitrate,
            started: Instant::now(),
        })
    }

    fn limit_reached(&self, _options: &RecordOptions) -> bool {
        self.limit.is_some_and(|m| self.frames >= m)
    }

    /// Frames that may still be written before the limit.
    fn room(&self) -> u64 {
        self.limit
            .map_or(u64::MAX, |m| m.saturating_sub(self.frames))
    }

    fn progress(&self) -> RecordProgress {
        let wall = self.started.elapsed().as_secs_f64();
        RecordProgress {
            frames: self.frames,
            seconds: self.frames as f64 / self.fps,
            repeated: self.repeated,
            dropped: self.dropped_early + self.dropped_behind,
            dropped_behind: self.dropped_behind,
            fps: if wall > 0.0 {
                self.frames as f64 / wall
            } else {
                0.0
            },
        }
    }

    fn video(&mut self, v: LiveVideo) -> Result<()> {
        let k = slot(
            v.time,
            self.t0,
            u64::from(self.frame_rate.0),
            u64::from(self.frame_rate.1),
        );
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
                    - (self.frames as i128 * TICKS_PER_SECOND as i128 * self.frame_rate.1 as i128
                        / self.frame_rate.0 as i128) as i64;
            }
            Place::Write { repeats } => {
                if let Some(last) = self.last.clone() {
                    // A gap fills up to the limit, and no further.
                    for _ in 0..repeats.min(self.room()) {
                        self.encode(last.clone())?;
                        self.repeated += 1;
                    }
                }
            }
        }
        if self.room() == 0 {
            return Ok(());
        }
        let frame = self.normalize(v)?;
        self.encode(frame)
    }

    fn normalize(&mut self, v: LiveVideo) -> Result<VideoFrame> {
        if (v.frame.format, v.color) != self.normalizer_for {
            tracing::info!(format = ?v.frame.format, "the source's picture format changed");
            self.normalizer = normalizer_for(&self.spec, &self.filters, &v, self.fps)?;
            self.normalizer_for = (v.frame.format, v.color);
        }
        let frame = self.normalizer.normalize(v.frame)?;
        if (frame.width, frame.height) == (self.width, self.height) {
            Ok(frame)
        } else {
            // A source that changed size mid-stream is scaled to the size
            // the recording started at.
            codec::colorspace::scale_frame(&frame, self.width, self.height)
                .context("scaling a resized source picture to the recording's size")
        }
    }

    fn encode(&mut self, mut frame: VideoFrame) -> Result<()> {
        frame.pts = self.frames;
        self.encoder
            .send_frame(&frame)
            .context("encoder.send_frame")?;
        self.frames += 1;
        self.last = Some(frame);
        while let Some(p) = self.encoder.receive_packet().context("receive_packet")? {
            self.muxer.add_packet(p)?;
            self.packets += 1;
        }
        Ok(())
    }

    fn audio(&mut self, a: LiveAudio) -> Result<()> {
        if self.audio_policy == RecordAudio::Drop || a.channels == 0 || a.sample_rate == 0 {
            return Ok(());
        }
        if self.audio.is_none() {
            self.audio = Some(self.open_audio(&a)?);
        }
        let out = self.audio.as_mut().expect("opened above");
        if a.sample_rate != out.rate {
            if !out.warned_rate {
                tracing::warn!(
                    from = out.rate,
                    to = a.sample_rate,
                    "the source's audio rate changed; its audio is dropped from here"
                );
                out.warned_rate = true;
            }
            return Ok(());
        }
        // Keep the first `channels` of each frame (all of them, usually).
        let src_ch = usize::from(a.channels);
        let keep = usize::from(out.channels).min(src_ch);
        let mut samples: Vec<f32> = if keep == src_ch && out.channels == a.channels {
            a.samples
        } else {
            let mut v = Vec::with_capacity(a.samples.len() / src_ch * usize::from(out.channels));
            for frame in a.samples.chunks_exact(src_ch) {
                v.extend_from_slice(&frame[..keep]);
                v.extend(std::iter::repeat_n(0.0, usize::from(out.channels) - keep));
            }
            v
        };
        let ch = usize::from(out.channels);
        let s = slot(a.time, self.t0, u64::from(out.rate), 1);
        let drift = s - out.written as i64;
        let tolerance = i64::from(out.rate) / 25; // 40 ms
        let max_gap = 10 * i64::from(out.rate);
        let first = out.written == 0;
        if (first && drift > 0 && drift <= max_gap) || (drift > tolerance && drift <= max_gap) {
            // A gap: silence up to where this frame belongs.
            let silence = vec![0.0f32; drift as usize * ch];
            encode_audio(out, &mut self.muxer, silence)?;
        } else if drift < 0 && (first || drift < -tolerance) {
            // An overlap (or, first, audio from before the picture): trim.
            let cut = ((-drift) as usize * ch).min(samples.len());
            samples.drain(..cut);
        }
        if !samples.is_empty() {
            encode_audio(out, &mut self.muxer, samples)?;
        }
        Ok(())
    }

    fn open_audio(&mut self, a: &LiveAudio) -> Result<AudioOut> {
        let codec = match self.audio_policy {
            RecordAudio::Codec(c) => c,
            _ => AudioCodec::Opus,
        };
        // Opus and AAC code up to 7.1 here; a wider source keeps its first
        // two channels (NDI's convention puts the main pair first).
        let channels = if a.channels > 8 {
            tracing::warn!(
                channels = a.channels,
                "more channels than the audio encoder takes; recording the first two"
            );
            2
        } else {
            a.channels
        };
        let mut config = AudioEncoderConfig::new(
            codec,
            a.sample_rate,
            channels,
            self.audio_bitrate.unwrap_or(0),
        );
        config.threads = 1;
        let encoder = codec::audio::create_encoder(config)
            .with_context(|| format!("the {codec:?} audio encoder"))?;
        let info = match codec {
            AudioCodec::Opus => {
                AudioInfo::opus(a.sample_rate, u16::from(channels), encoder.extra_data())
            }
            _ => AudioInfo::aac_lc(
                encoder.sample_rate(),
                u16::from(channels),
                encoder.extra_data(),
            ),
        };
        self.muxer
            .with_audio(info)
            .context("the recording's audio track")?;
        tracing::info!(?codec, channels, rate = a.sample_rate, "audio");
        Ok(AudioOut {
            encoder,
            codec,
            rate: a.sample_rate,
            channels,
            source_channels: a.channels,
            written: 0,
            warned_rate: false,
        })
    }

    /// Flush both encoders and write the file.
    fn finish(
        mut self,
        output: &Path,
    ) -> Result<(RecordProgress, Option<String>, (u32, u32), (u32, u32))> {
        self.encoder.flush().context("encoder.flush")?;
        while let Some(p) = self.encoder.receive_packet().context("receive_packet")? {
            self.muxer.add_packet(p)?;
            self.packets += 1;
        }
        anyhow::ensure!(self.packets > 0, "the encoder produced no video");
        let mut audio_label = None;
        if let Some(mut out) = self.audio.take() {
            for p in out.encoder.flush().context("audio encoder flush")? {
                self.muxer.add_audio(&p.data, p.duration as u32)?;
            }
            // Hide the encoder's priming, and end the track on the last
            // real sample, as every other audio path here does.
            let coded_rate = out.encoder.sample_rate();
            self.muxer.set_audio_edit(container::edit::TrackEdit {
                delay: 0,
                media_time: u64::from(out.encoder.pre_skip()),
                // The audio ends with the pictures: what arrived after the
                // last frame (the limit reached mid-run) is not presented.
                duration: Some(container::edit::rescale_round(
                    out.written
                        .min(slot_count(self.frames, self.frame_rate, out.rate)),
                    coded_rate,
                    out.rate,
                )),
            });
            audio_label = Some(format!(
                "{} {}ch {} Hz{}",
                match out.codec {
                    AudioCodec::Opus => "opus",
                    AudioCodec::HeAac => "he-aac",
                    _ => "aac",
                },
                out.channels,
                out.rate,
                if out.source_channels != out.channels {
                    format!(" (of {})", out.source_channels)
                } else {
                    String::new()
                }
            ));
        }
        let progress = self.progress();
        self.muxer.finalize(output)?;
        Ok((
            progress,
            audio_label,
            (self.width, self.height),
            self.frame_rate,
        ))
    }
}

fn encode_audio(out: &mut AudioOut, muxer: &mut LiveMuxer, samples: Vec<f32>) -> Result<()> {
    let n = (samples.len() / usize::from(out.channels)) as u64;
    let frame = AudioFrame {
        samples,
        sample_rate: out.rate,
        channels: out.channels,
        pts: (out.written as i128 * 1_000_000 / i128::from(out.rate)) as i64,
    };
    out.written += n;
    for p in out.encoder.encode(&frame).context("audio encode")? {
        muxer.add_audio(&p.data, p.duration as u32)?;
    }
    Ok(())
}

/// A normaliser for pictures like `v`: the job engine's per-frame work for
/// a source of `v`'s layout and colour.
fn normalizer_for(
    spec: &OutputSpec,
    filters: &Arc<codec::filter::FilterChain>,
    v: &LiveVideo,
    fps: f64,
) -> Result<FrameNormalizer> {
    let header = container::streaming::DemuxHeader {
        codec: "ndi".into(),
        info: StreamInfo {
            codec: "ndi".into(),
            width: v.frame.width,
            height: v.frame.height,
            frame_rate: fps,
            duration: 0.0,
            pixel_format: v.frame.format,
            color_space: v.frame.color_space,
            total_frames: 0,
            bitrate: 0,
            color_metadata: v.color,
        },
        timescale: 90_000,
        rotation_degrees: 0,
        sample_aspect: (1, 1),
    };
    let cfg = DecodePumpConfig::for_source(&header, spec, Arc::clone(filters), None);
    FrameNormalizer::new(&cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_rounds_to_the_nearest_frame_both_ways() {
        let t = |s: f64| (s * TICKS_PER_SECOND as f64) as i64;
        // 29.97: a frame every 33.3667 ms.
        assert_eq!(slot(t(0.0), 0, 30000, 1001), 0);
        assert_eq!(slot(t(0.0333), 0, 30000, 1001), 1);
        assert_eq!(slot(t(0.0160), 0, 30000, 1001), 0);
        assert_eq!(slot(t(0.0170), 0, 30000, 1001), 1);
        assert_eq!(slot(t(10.01), 0, 30000, 1001), 300);
        assert_eq!(slot(-t(0.0334), 0, 30000, 1001), -1);
        // Samples at 48 kHz.
        assert_eq!(slot(t(1.0), 0, 48_000, 1), 48_000);
    }

    /// A finite source on a 25 fps clock: `frames` pictures of 4:2:2 grey,
    /// one missing (`skip`), and 48 kHz stereo audio in 40 ms runs that
    /// starts 100 ms after the first picture.
    struct Synthetic {
        next: u64,
        frames: u64,
        skip: u64,
        audio_next: bool,
    }

    impl LiveSource for Synthetic {
        fn next_event(&mut self, _timeout: Duration) -> Result<Option<LiveEvent>> {
            let tick = TICKS_PER_SECOND / 25;
            if self.next >= self.frames {
                return Ok(Some(LiveEvent::End));
            }
            let i = self.next;
            if self.audio_next {
                self.audio_next = false;
                self.next += 1;
                if i < 3 {
                    return Ok(None);
                }
                let samples: Vec<f32> = (0..1920 * 2)
                    .map(|n| ((n / 2) as f32 * 0.05).sin() * 0.25)
                    .collect();
                return Ok(Some(LiveEvent::Audio(LiveAudio {
                    samples,
                    sample_rate: 48_000,
                    channels: 2,
                    time: 1_000 + i as i64 * tick,
                })));
            }
            self.audio_next = true;
            if i == self.skip {
                return Ok(None);
            }
            let (w, h) = (64u32, 48u32);
            let mut data = vec![100u8; (w * h) as usize];
            data.extend(vec![128u8; (w * h) as usize]);
            Ok(Some(LiveEvent::Video(LiveVideo {
                frame: VideoFrame::new(
                    bytes::Bytes::from(data),
                    w,
                    h,
                    codec::frame::PixelFormat::Yuv422p,
                    codec::frame::ColorSpace::Bt601,
                    0,
                ),
                color: ColorMetadata {
                    matrix_coefficients: 6,
                    colour_primaries: 6,
                    ..ColorMetadata::default()
                },
                frame_rate: (25, 1),
                time: 1_000 + i as i64 * tick,
            })))
        }

        fn name(&self) -> String {
            "SYNTH (test)".into()
        }
    }

    /// A recording of a live source is as long as the time it covered: the
    /// skipped picture is repeated, not lost, and the audio, which started
    /// late, is padded at the front so it stays in step.
    #[test]
    fn a_recording_fills_gaps_and_keeps_audio_in_step() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("rec.mp4");
        let mut options = RecordOptions::new(&out);
        options.video_codec = VideoCodecPolicy::H264;
        options.encoder_backend = Some(EncoderBackend::H26x);
        options.idle_timeout = Duration::ZERO;
        let source = Synthetic {
            next: 0,
            frames: 50,
            skip: 20,
            audio_next: false,
        };
        let outcome = record(source, &options, |_| {}).expect("record");
        assert_eq!(outcome.ended, EndReason::SourceEnded);
        assert_eq!((outcome.width, outcome.height), (64, 48));
        assert_eq!(outcome.progress.frames, 50, "{:?}", outcome.progress);
        assert_eq!(outcome.progress.repeated, 1);
        assert_eq!(outcome.audio.as_deref(), Some("opus 2ch 48000 Hz"));

        let info = crate::probe_file(&out).expect("probe");
        assert_eq!(info.video_codec, "h264");
        assert!((info.duration - 2.0).abs() < 0.05, "{}", info.duration);
        let audio = info.audio.expect("an audio track");
        assert_eq!((audio.codec.as_str(), audio.channels), ("opus", 2));
    }

    #[test]
    fn a_limit_ends_the_recording_and_webm_takes_only_vp8_or_vp9() {
        let dir = tempfile::tempdir().unwrap();
        let mut options = RecordOptions::new(dir.path().join("rec.webm"));
        let source = || Synthetic {
            next: 0,
            frames: 1000,
            skip: u64::MAX,
            audio_next: false,
        };
        let err = record(source(), &options, |_| {}).unwrap_err().to_string();
        assert!(err.contains("WebM file carries VP8 or VP9"), "{err}");

        options.video_codec = VideoCodecPolicy::Vp9;
        options.max_frames = Some(10);
        options.audio = RecordAudio::Drop;
        let outcome = record(source(), &options, |_| {}).expect("record");
        assert_eq!(outcome.ended, EndReason::Limit);
        assert_eq!(outcome.progress.frames, 10);
        assert_eq!(outcome.audio, None);
        let info = crate::probe_file(dir.path().join("rec.webm")).expect("probe");
        assert_eq!(info.video_codec, "vp9");
        assert!(info.audio.is_none());
    }

    #[test]
    fn a_picture_is_written_dropped_or_reanchored_by_where_it_falls() {
        assert_eq!(place(5, 5, 300), Place::Write { repeats: 0 });
        assert_eq!(place(8, 5, 300), Place::Write { repeats: 3 });
        assert_eq!(place(4, 5, 300), Place::Drop);
        assert_eq!(place(400, 5, 300), Place::Reanchor);
        assert_eq!(place(-400, 5, 300), Place::Reanchor);
    }
}
