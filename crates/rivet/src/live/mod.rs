//! Live ends of a job: a source that runs in real time (an NDI source), or
//! a destination that does (a file played out to NDI).
//!
//! A live job is still an ordinary job: the same [`OutputSpec`] built by the
//! same [`TranscodeSettings`](crate::TranscodeSettings) from the same keys
//! (CLI flags, the batch manifest, the HTTP API), run by the job engine's
//! live path ([`crate::job::run_live_job`]). What differs is only where the
//! pictures come from and go to, which a URI in place of a path says:
//!
//! ```text
//! rivet transcode "ndi://STUDIO (Camera 1)" -o cam1.mp4 --codec h264 --duration 1h
//! rivet transcode "ndi://Camera 1?bandwidth=lowest" --mode hls --ladder -o live/
//! rivet transcode programme.mkv -o ndi://Playout --loop
//! ```
//!
//! This module holds what is not the engine: the [`LiveSource`] trait the
//! engine pulls from, the URI vocabulary ([`LiveUri`]), opening a source by
//! URI ([`open_source`]), probing one into a [`MediaInfo`] the spec builder
//! reads ([`probe_source`]), and a file read at its own pace as a source
//! ([`FileSource`], what a file played out to NDI is).
//!
//! [`OutputSpec`]: crate::spec::OutputSpec
//! [`MediaInfo`]: crate::probe::MediaInfo

mod file_source;
mod uri;

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use codec::frame::{ColorMetadata, VideoFrame};

pub use file_source::{FileSource, rational_frame_rate};
pub use uri::{LiveUri, NdiBandwidth, NdiEndpoint, is_live_uri};

/// 100 ns ticks per second: the unit of every live timestamp (NDI's).
pub const TICKS_PER_SECOND: i64 = 10_000_000;

/// One picture from a live source.
#[derive(Debug, Clone)]
pub struct LiveVideo {
    /// The picture, in any layout the colorspace layer takes (4:2:0, 4:2:2,
    /// NV12, RGBA; 8 or 10 bits). Its `pts` is ignored.
    pub frame: VideoFrame,
    /// The colour the source declares for it.
    pub color: ColorMetadata,
    /// `(numerator, denominator)`; `(0, _)` when the source does not say.
    pub frame_rate: (u32, u32),
    /// When it was taken, in 100 ns ticks on a clock every event of the
    /// source shares (NDI: the sender's, since the Unix epoch).
    pub time: i64,
}

/// A run of sound from a live source.
#[derive(Debug, Clone)]
pub struct LiveAudio {
    /// Interleaved samples, nominal level ±1.0.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u8,
    /// When its first sample was taken, on the same clock as the video.
    pub time: i64,
}

/// What a source yields.
#[derive(Debug, Clone)]
pub enum LiveEvent {
    Video(LiveVideo),
    Audio(LiveAudio),
    /// The source has ended (a file; NDI never says this).
    End,
}

/// A live source the engine pulls from on a thread of its own.
pub trait LiveSource: Send {
    /// The next event, waiting up to `timeout`; `None` when nothing came.
    /// An error ends the job: [`SourceLost`] as the source going away (what
    /// was made is kept), anything else as a failure.
    fn next_event(&mut self, timeout: Duration) -> Result<Option<LiveEvent>>;

    /// The source's name, for messages and default output names.
    fn name(&self) -> String;

    /// What kind of source it is, as a probe names a container (`ndi`).
    fn kind(&self) -> &'static str {
        "live"
    }

    /// Whether the source runs on its own clock (a camera, NDI), so that a
    /// picture the engine has no room for is gone and its frame is filled by
    /// a repeat; or yields as fast as it is read (a file), so the engine
    /// waits for room instead.
    fn is_realtime(&self) -> bool {
        true
    }
}

impl<S: LiveSource + ?Sized> LiveSource for Box<S> {
    fn next_event(&mut self, timeout: Duration) -> Result<Option<LiveEvent>> {
        (**self).next_event(timeout)
    }
    fn name(&self) -> String {
        (**self).name()
    }
    fn kind(&self) -> &'static str {
        (**self).kind()
    }
    fn is_realtime(&self) -> bool {
        (**self).is_realtime()
    }
}

/// The error a [`LiveSource`] returns when its source went away.
#[derive(Debug, Clone, Copy)]
pub struct SourceLost;

impl std::fmt::Display for SourceLost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the source went away")
    }
}

impl std::error::Error for SourceLost {}

/// Open the live source `uri` names, waiting up to `wait` for it to appear.
/// The `ndi` feature opens `ndi://` sources; a build without it says so.
pub fn open_source(uri: &str, wait: Duration) -> Result<Box<dyn LiveSource>> {
    match LiveUri::parse(uri)? {
        LiveUri::Ndi(endpoint) => open_ndi(&endpoint, wait),
    }
}

#[cfg(feature = "ndi")]
fn open_ndi(endpoint: &NdiEndpoint, wait: Duration) -> Result<Box<dyn LiveSource>> {
    Ok(Box::new(crate::ndi::NdiSource::open(endpoint, wait)?))
}

#[cfg(not(feature = "ndi"))]
fn open_ndi(endpoint: &NdiEndpoint, _wait: Duration) -> Result<Box<dyn LiveSource>> {
    bail!(
        "ndi://{} is an NDI source, and this build has no NDI: rebuild with `--features ndi`",
        endpoint.name
    )
}

/// A source that has been probed: the events the probe read are handed out
/// first, so the job loses nothing to it.
pub struct ProbedSource<S> {
    source: S,
    pending: VecDeque<LiveEvent>,
}

impl<S: LiveSource> LiveSource for ProbedSource<S> {
    fn next_event(&mut self, timeout: Duration) -> Result<Option<LiveEvent>> {
        match self.pending.pop_front() {
            Some(e) => Ok(Some(e)),
            None => self.source.next_event(timeout),
        }
    }
    fn name(&self) -> String {
        self.source.name()
    }
    fn kind(&self) -> &'static str {
        self.source.kind()
    }
    fn is_realtime(&self) -> bool {
        self.source.is_realtime()
    }
}

/// How long the probe keeps listening for sound after the first picture.
/// NDI sends audio and video separately, and a sender's first audio frame
/// can trail its first picture by a frame or two.
const AUDIO_GRACE: Duration = Duration::from_millis(500);

/// Wait up to `wait` for `source`'s first picture and describe the source as
/// a probe describes a file — size, rate, pixel format, colour, its audio —
/// for the spec builder ([`TranscodeSettings::into_spec_for`](crate::TranscodeSettings::into_spec_for)).
/// The source comes back with every event the probe read still to come.
pub fn probe_source<S: LiveSource>(
    mut source: S,
    wait: Duration,
) -> Result<(crate::probe::MediaInfo, ProbedSource<S>)> {
    let started = Instant::now();
    let mut pending = VecDeque::new();
    let mut first: Option<LiveVideo> = None;
    let mut audio: Option<(u32, u8)> = None;
    let mut seen_picture_at = None;
    loop {
        let limit = match seen_picture_at {
            Some(t) => AUDIO_GRACE.saturating_sub(Instant::now().duration_since(t)),
            None => wait.saturating_sub(started.elapsed()),
        };
        if limit.is_zero() || (first.is_some() && audio.is_some()) {
            break;
        }
        match source.next_event(limit.min(Duration::from_millis(100)))? {
            Some(LiveEvent::Video(v)) => {
                if first.is_none() {
                    first = Some(v.clone());
                    seen_picture_at = Some(Instant::now());
                }
                pending.push_back(LiveEvent::Video(v));
            }
            Some(LiveEvent::Audio(a)) => {
                audio.get_or_insert((a.sample_rate, a.channels));
                pending.push_back(LiveEvent::Audio(a));
            }
            Some(LiveEvent::End) => {
                pending.push_back(LiveEvent::End);
                break;
            }
            None => {}
        }
    }
    let Some(first) = first else {
        bail!(
            "no picture from {} within {:.0} s{}",
            source.name(),
            wait.as_secs_f64(),
            if audio.is_some() {
                " (it sends audio only? a video job needs pictures)"
            } else {
                ""
            }
        );
    };
    let info = media_info(&source, &first, audio);
    Ok((info, ProbedSource { source, pending }))
}

/// A live source's first picture (and audio, when it sends any) as a
/// [`MediaInfo`](crate::probe::MediaInfo).
fn media_info<S: LiveSource>(
    source: &S,
    first: &LiveVideo,
    audio: Option<(u32, u8)>,
) -> crate::probe::MediaInfo {
    let (n, d) = first.frame_rate;
    let frame_rate = if n > 0 && d > 0 {
        f64::from(n) / f64::from(d)
    } else {
        0.0
    };
    let (w, h) = (first.frame.width, first.frame.height);
    crate::probe::MediaInfo {
        container: source.kind().to_string(),
        video_codec: format!("{} (uncompressed)", source.kind()),
        width: w,
        height: h,
        stored_width: w,
        stored_height: h,
        rotation_degrees: 0,
        sample_aspect: (1, 1),
        frame_rate,
        duration: 0.0,
        pixel_format: format!("{:?}", first.frame.format),
        audio: audio.map(|(sample_rate, channels)| crate::probe::AudioStreamInfo {
            codec: "pcm_f32le".into(),
            sample_rate,
            channels: u16::from(channels),
        }),
        subtitles: Vec::new(),
    }
}

/// The media type of a file a live job wrote, by its extension.
pub fn media_type_of_path(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("mov") => "video/quicktime",
        Some("webm") => "video/webm",
        _ => "video/mp4",
    }
}
