//! Play a file out as an NDI source: demux → decode (hardware first, as
//! everywhere) → the job engine's per-frame normalisation → NDI, with the
//! audio decoded alongside and sent ahead of each picture it accompanies.
//! The NDI runtime paces the pictures at the file's frame rate.
//!
//! Pictures go out as I420 (8-bit, the default: SDR BT.709, an HDR source
//! tonemapped) or, with `ten_bit`, P216 carrying the source's own colour
//! and depth.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use codec::audio::AudioDecoder;
use codec::decode;
use codec::frame::{PixelFormat, VideoFrame};
use container::streaming;

use crate::decode_pump::{DecodePumpConfig, FrameNormalizer};
use crate::spec::{BitDepth, ColorPolicy, OutputSpec, Rung};

/// How a file is sent.
#[derive(Debug, Clone)]
pub struct SendOptions {
    pub input: PathBuf,
    /// The stream name; receivers see `MACHINE (name)`.
    pub name: String,
    /// NDI groups to announce in, comma-separated.
    pub groups: Option<String>,
    /// Start again from the beginning at the end, until stopped.
    pub repeat: bool,
    /// Send 10-bit P216 in the source's own colour (default: 8-bit I420,
    /// SDR BT.709).
    pub ten_bit: bool,
    /// Send the file's audio (default `true`).
    pub audio: bool,
    /// Set to stop sending.
    pub stop: Option<Arc<AtomicBool>>,
}

impl SendOptions {
    pub fn new(input: impl Into<PathBuf>, name: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            name: name.into(),
            groups: None,
            repeat: false,
            ten_bit: false,
            audio: true,
            stop: None,
        }
    }
}

/// What a send did.
#[derive(Debug, Clone)]
pub struct SendOutcome {
    /// The source name receivers see.
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub frame_rate: (u32, u32),
    /// Pictures sent, over every pass.
    pub frames: u64,
    /// Passes through the file (more than one with `repeat`).
    pub passes: u64,
    /// Whether audio was sent.
    pub audio: bool,
    pub stopped: bool,
    pub elapsed: Duration,
}

/// Send `options.input` as an NDI source. `progress` is called with the
/// pictures sent so far about once a second.
pub fn send_file(options: &SendOptions, mut progress: impl FnMut(u64)) -> Result<SendOutcome> {
    let started = Instant::now();
    let bytes = std::fs::read(&options.input)
        .with_context(|| format!("reading {}", options.input.display()))?;
    let runtime = ndi::Ndi::load()?;
    let mut sender_options = ndi::SenderOptions::new(options.name.clone());
    sender_options.groups = options.groups.clone();
    let mut sender = runtime.sender(&sender_options)?;
    tracing::info!(name = %options.name, runtime = %runtime.version(), "NDI source announced");

    let stopped = || {
        options
            .stop
            .as_ref()
            .is_some_and(|s| s.load(Ordering::Relaxed))
    };
    let mut frames = 0u64;
    let mut passes = 0u64;
    let mut last_report = Instant::now();
    let mut shape = None;
    let mut sent_audio = false;
    loop {
        passes += 1;
        let pass = send_pass(
            &bytes,
            options,
            &mut sender,
            &stopped,
            &mut frames,
            &mut |n| {
                if last_report.elapsed() >= Duration::from_secs(1) {
                    progress(n);
                    last_report = Instant::now();
                }
            },
        )?;
        shape = shape.or(Some((pass.width, pass.height, pass.frame_rate)));
        sent_audio |= pass.audio;
        if stopped() || !options.repeat {
            break;
        }
    }
    let (width, height, frame_rate) = shape.unwrap_or_default();
    Ok(SendOutcome {
        name: sender.name().to_string(),
        width,
        height,
        frame_rate,
        frames,
        passes,
        audio: sent_audio,
        stopped: stopped(),
        elapsed: started.elapsed(),
    })
}

struct Pass {
    width: u32,
    height: u32,
    frame_rate: (u32, u32),
    audio: bool,
}

fn send_pass(
    bytes: &[u8],
    options: &SendOptions,
    sender: &mut ndi::Sender,
    stopped: &dyn Fn() -> bool,
    frames: &mut u64,
    progress: &mut dyn FnMut(u64),
) -> Result<Pass> {
    let mut demuxer = streaming::demux_streaming(bytes).context("demux")?;
    let header = demuxer.header().clone();
    let (width, height) = header.upright_dims();
    let frame_rate = rational_frame_rate(header.info.frame_rate);
    let mut spec = OutputSpec::single_file(vec![Rung::new(width, height)]);
    if options.ten_bit {
        spec.color = ColorPolicy::Passthrough;
        spec.bit_depth = BitDepth::Auto;
    } else {
        spec.color = ColorPolicy::TonemapToSdr;
        spec.bit_depth = BitDepth::EightBit;
    }
    let filters = Arc::new(codec::filter::FilterChain::prepare(&spec.filters)?);
    let cfg = DecodePumpConfig::for_source(&header, &spec, filters, None);
    let mut normalizer = FrameNormalizer::new(&cfg)?;
    let decoder =
        decode::create_decoder(&header.codec, header.info.clone()).context("create_decoder")?;
    let mut decoder = decode::RotatingDecoder::new(decoder, header.rotation_degrees);

    let mut audio = if options.audio {
        demuxer.audio().cloned().and_then(|track| {
            AudioFeed::new(track)
                .map_err(|e| tracing::warn!("the file's audio is not sent: {e:#}"))
                .ok()
        })
    } else {
        None
    };
    let pass = Pass {
        width,
        height,
        frame_rate,
        audio: audio.is_some(),
    };
    let frame_us = 1_000_000.0 * f64::from(frame_rate.1) / f64::from(frame_rate.0);
    let mut index = 0u64;
    let mut emit = |frame: VideoFrame,
                    sender: &mut ndi::Sender,
                    audio: &mut Option<AudioFeed>|
     -> Result<()> {
        let frame = normalizer.normalize(frame)?;
        // The audio up to the end of this picture goes first.
        if let Some(a) = audio.as_mut() {
            a.send_until(((index + 1) as f64 * frame_us) as i64, sender)?;
        }
        let picture = ndi::Picture {
            layout: match frame.format {
                PixelFormat::Yuv420p10le => ndi::Layout::Yuv420p10le,
                PixelFormat::Yuv420p => ndi::Layout::Yuv420p,
                other => bail!("the normaliser left {other:?}"),
            },
            width: frame.width,
            height: frame.height,
            data: frame.data.to_vec(),
        };
        sender.send_picture(&picture, frame_rate, None)?;
        index += 1;
        *frames += 1;
        progress(*frames);
        Ok(())
    };
    loop {
        if stopped() {
            return Ok(pass);
        }
        match demuxer.next_video_sample().context("next_video_sample")? {
            Some(sample) => {
                decoder.push_sample(&sample.data).context("push_sample")?;
                while let Some(frame) = decoder.decode_next().context("decode_next")? {
                    emit(frame, sender, &mut audio)?;
                    if stopped() {
                        return Ok(pass);
                    }
                }
            }
            None => {
                decoder.finish().context("decoder.finish")?;
                while let Some(frame) = decoder.decode_next().context("decode_next")? {
                    emit(frame, sender, &mut audio)?;
                }
                if let Some(a) = audio.as_mut() {
                    a.send_until(i64::MAX, sender)?;
                }
                return Ok(pass);
            }
        }
    }
}

/// The file's audio, decoded a packet at a time as the pictures need it.
struct AudioFeed {
    track: container::demux::AudioTrack,
    decoder: Box<dyn AudioDecoder>,
    next: usize,
    /// Microseconds of audio sent.
    sent_us: i64,
    flushed: bool,
}

impl AudioFeed {
    fn new(track: container::demux::AudioTrack) -> Result<Self> {
        let codec = track.codec.to_ascii_lowercase();
        let extra = if codec == "aac" {
            &track.asc
        } else {
            &track.codec_private
        };
        let extra = (!extra.is_empty()).then_some(extra.as_slice());
        let decoder = codec::audio::create_decoder(
            &codec,
            extra,
            track.sample_rate,
            track.channels.min(255) as u8,
        )
        .with_context(|| format!("no decoder for {codec} audio"))?;
        Ok(Self {
            track,
            decoder,
            next: 0,
            sent_us: 0,
            flushed: false,
        })
    }

    /// Decode and send until `until_us` of audio has gone out (or the
    /// track ends).
    fn send_until(&mut self, until_us: i64, sender: &mut ndi::Sender) -> Result<()> {
        while self.sent_us < until_us {
            let frames = if self.next < self.track.samples.len() {
                let packet = &self.track.samples[self.next];
                self.next += 1;
                match self.decoder.decode(packet, self.sent_us) {
                    Ok(f) => f,
                    Err(e) => {
                        tracing::debug!("skipping an audio packet that does not decode: {e}");
                        continue;
                    }
                }
            } else if !self.flushed {
                self.flushed = true;
                self.decoder.flush().unwrap_or_default()
            } else {
                return Ok(());
            };
            for f in frames {
                let ch = usize::from(f.channels.max(1));
                let n = f.samples.len() / ch;
                if n == 0 || f.sample_rate == 0 {
                    continue;
                }
                sender.send_audio(f.sample_rate, ch, &f.samples, None)?;
                self.sent_us += (n as i64 * 1_000_000) / i64::from(f.sample_rate);
            }
        }
        Ok(())
    }
}

/// A frame rate as NDI wants it: the NTSC rates exactly (`30000/1001`),
/// others to a thousandth.
fn rational_frame_rate(fps: f64) -> (u32, u32) {
    if !(fps.is_finite() && fps > 0.0) {
        return (30, 1);
    }
    for base in [24u32, 30, 48, 60, 120] {
        let ntsc = f64::from(base) * 1000.0 / 1001.0;
        if (fps - ntsc).abs() < 0.005 {
            return (base * 1000, 1001);
        }
    }
    if (fps - fps.round()).abs() < 0.001 {
        return (fps.round() as u32, 1);
    }
    let n = (fps * 1000.0).round() as u32;
    let g = gcd(n, 1000);
    (n / g, 1000 / g)
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_rates_go_out_as_ndi_names_them() {
        assert_eq!(rational_frame_rate(29.97), (30000, 1001));
        assert_eq!(rational_frame_rate(23.976), (24000, 1001));
        assert_eq!(rational_frame_rate(59.94006), (60000, 1001));
        assert_eq!(rational_frame_rate(25.0), (25, 1));
        assert_eq!(rational_frame_rate(12.5), (25, 2));
        assert_eq!(rational_frame_rate(0.0), (30, 1));
    }
}
