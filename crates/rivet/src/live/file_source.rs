//! A media file as a [`LiveSource`]: what a file played out live (to
//! `ndi://…`) is read through. Pictures are decoded as any rivet input is
//! (hardware decoders first, the container's rotation applied), stamped by
//! their place on the file's frame clock; the audio is decoded alongside and
//! handed out ahead of the picture it accompanies. With `repeat` the file
//! starts again at its end, the clock running on.
//!
//! The source is not real time: it yields as fast as it decodes, and the
//! destination paces it (an NDI sender clocks its video), so the engine
//! waits for room rather than dropping a picture
//! ([`LiveSource::is_realtime`]).

use std::time::Duration;

use anyhow::{Context, Result};
use bytes::Bytes;
use codec::audio::AudioDecoder;
use codec::decode::{self, Decoder};
use container::streaming::{self, DemuxHeader, StreamingDemuxer};

use super::{LiveAudio, LiveEvent, LiveSource, LiveVideo, TICKS_PER_SECOND};

/// A file read as a live source.
pub struct FileSource {
    name: String,
    input: Bytes,
    repeat: bool,
    header: DemuxHeader,
    frame_rate: (u32, u32),
    pass: Option<Pass>,
    /// Ticks at which the current pass starts.
    pass_start: i64,
    /// Pictures handed out over every pass.
    pictures: u64,
    /// Pictures handed out in this pass.
    pass_pictures: u64,
    ended: bool,
}

struct Pass {
    demuxer: Box<dyn StreamingDemuxer>,
    decoder: Box<dyn Decoder>,
    drained: bool,
    audio: Option<AudioFeed>,
}

struct AudioFeed {
    track: container::demux::AudioTrack,
    decoder: Box<dyn AudioDecoder>,
    next: usize,
    /// Samples (per channel) handed out this pass, at `rate`.
    samples: u64,
    rate: u32,
    flushed: bool,
}

impl FileSource {
    /// `input` (the file's bytes) as a source named `name`.
    pub fn new(name: impl Into<String>, input: Bytes, repeat: bool) -> Result<Self> {
        let demuxer = streaming::demux_streaming_shared(input.clone()).context("demux")?;
        let header = demuxer.header().clone();
        let frame_rate = rational_frame_rate(header.info.frame_rate);
        Ok(Self {
            name: name.into(),
            input,
            repeat,
            header,
            frame_rate,
            pass: None,
            pass_start: 0,
            pictures: 0,
            pass_pictures: 0,
            ended: false,
        })
    }

    fn open_pass(&mut self) -> Result<Pass> {
        let demuxer = streaming::demux_streaming_shared(self.input.clone()).context("demux")?;
        let decoder = decode::create_decoder(&self.header.codec, self.header.info.clone())
            .context("create_decoder")?;
        let decoder = decode::RotatingDecoder::new(decoder, self.header.rotation_degrees);
        let audio = demuxer.audio().cloned().and_then(|track| {
            AudioFeed::new(track)
                .map_err(|e| tracing::warn!("the file's audio is left out: {e:#}"))
                .ok()
        });
        Ok(Pass {
            demuxer,
            decoder,
            drained: false,
            audio,
        })
    }

    fn picture_time(&self, index_in_pass: u64) -> i64 {
        let (n, d) = self.frame_rate;
        self.pass_start
            + (i128::from(index_in_pass) * i128::from(TICKS_PER_SECOND) * i128::from(d)
                / i128::from(n)) as i64
    }
}

impl LiveSource for FileSource {
    fn next_event(&mut self, _timeout: Duration) -> Result<Option<LiveEvent>> {
        loop {
            if self.ended {
                return Ok(Some(LiveEvent::End));
            }
            if self.pass.is_none() {
                self.pass = Some(self.open_pass()?);
                self.pass_pictures = 0;
            }
            let next_picture_at = self.picture_time(self.pass_pictures + 1);
            let pass_start = self.pass_start;
            let pass = self.pass.as_mut().expect("opened above");
            // The audio up to the end of the next picture goes first.
            if let Some(feed) = pass.audio.as_mut()
                && let Some(audio) = feed.next_until(next_picture_at - pass_start)?
            {
                return Ok(Some(LiveEvent::Audio(LiveAudio {
                    time: pass_start + audio.time,
                    ..audio
                })));
            }
            if let Some(frame) = pass.decoder.decode_next().context("decode_next")? {
                let time = self.picture_time(self.pass_pictures);
                self.pass_pictures += 1;
                self.pictures += 1;
                return Ok(Some(LiveEvent::Video(LiveVideo {
                    frame,
                    color: self.header.info.color_metadata,
                    frame_rate: self.frame_rate,
                    time,
                })));
            }
            if pass.drained {
                // The pass is over: once more from the top, or the end.
                if self.repeat && self.pass_pictures > 0 {
                    self.pass_start = self.picture_time(self.pass_pictures);
                    self.pass = None;
                    continue;
                }
                self.ended = true;
                continue;
            }
            match pass
                .demuxer
                .next_video_sample()
                .context("next_video_sample")?
            {
                Some(sample) => pass
                    .decoder
                    .push_sample(&sample.data)
                    .context("push_sample")?,
                None => {
                    pass.decoder.finish().context("decoder.finish")?;
                    pass.drained = true;
                }
            }
        }
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn kind(&self) -> &'static str {
        "file"
    }

    fn is_realtime(&self) -> bool {
        false
    }
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
            rate: track.sample_rate.max(1),
            track,
            decoder,
            next: 0,
            samples: 0,
            flushed: false,
        })
    }

    /// The next decoded run of audio, if what was handed out so far ends
    /// before `until` (ticks into the pass) and the track has more.
    fn next_until(&mut self, until: i64) -> Result<Option<LiveAudio>> {
        loop {
            let at = (i128::from(self.samples) * i128::from(TICKS_PER_SECOND)
                / i128::from(self.rate)) as i64;
            if at >= until {
                return Ok(None);
            }
            let frames = if self.next < self.track.samples.len() {
                let packet = &self.track.samples[self.next];
                self.next += 1;
                match self.decoder.decode(packet, 0) {
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
                return Ok(None);
            };
            let mut samples = Vec::new();
            let (mut rate, mut channels) = (0u32, 0u8);
            for f in frames {
                if f.samples.is_empty() || f.channels == 0 || f.sample_rate == 0 {
                    continue;
                }
                rate = f.sample_rate;
                channels = f.channels;
                samples.extend_from_slice(&f.samples);
            }
            if samples.is_empty() {
                continue;
            }
            // A decoder that codes at another rate than the container says
            // (HE-AAC's SBR doubling it) is timed at its own.
            if rate != self.rate && self.samples == 0 {
                self.rate = rate;
            }
            let time = (i128::from(self.samples) * i128::from(TICKS_PER_SECOND)
                / i128::from(self.rate)) as i64;
            self.samples += (samples.len() / usize::from(channels)) as u64;
            return Ok(Some(LiveAudio {
                samples,
                sample_rate: rate,
                channels,
                time,
            }));
        }
    }
}

/// A frame rate as a fraction: the NTSC rates exactly (`30000/1001`), others
/// to a thousandth.
pub fn rational_frame_rate(fps: f64) -> (u32, u32) {
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
    fn frame_rates_are_fractions_ndi_and_mp4_agree_on() {
        assert_eq!(rational_frame_rate(29.97), (30000, 1001));
        assert_eq!(rational_frame_rate(23.976), (24000, 1001));
        assert_eq!(rational_frame_rate(59.94006), (60000, 1001));
        assert_eq!(rational_frame_rate(25.0), (25, 1));
        assert_eq!(rational_frame_rate(12.5), (25, 2));
        assert_eq!(rational_frame_rate(0.0), (30, 1));
    }
}
