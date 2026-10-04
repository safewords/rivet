//! Raw video elementary streams as inputs: the bitstream with no container
//! around it.
//!
//! - **H.264 / HEVC Annex B** ([`annexb`]): NAL units behind start codes
//!   (ITU-T H.264 Annex B, H.265 Annex B), cut into access units the way
//!   the standards define them (H.264 §7.4.1.2.3, H.265 §7.4.2.4.4).
//! - **IVF** ([`ivf`]): the WebM project's simple frame file — a 32-byte
//!   `DKIF` header, then each frame behind its size and timestamp. VP8, VP9
//!   and AV1.
//! - **AV1 OBU streams** ([`obu`]): the low-overhead bitstream format of the
//!   AV1 specification §5 (every OBU with its size, a temporal delimiter
//!   opening each temporal unit), and the length-delimited format of its
//!   Annex B.
//! - **MPEG-1 / MPEG-2 video** ([`mpegv`]): ISO/IEC 11172-2 / 13818-2
//!   start-code units from a sequence header on, one sample per coded frame.
//!
//! Each reader indexes the whole stream when it opens — the input is in
//! memory already — so the header knows the frame count, the dimensions, the
//! pixel format and the colour before the first sample is pulled, and each
//! sample is one coded picture: what the decoders take from MP4 or Matroska.
//!
//! # Frame rate
//!
//! A container states a frame rate; an elementary stream may not. The rate
//! is the stream's own where it states one — H.264 / HEVC VUI timing, the
//! AV1 sequence header's timing info, the MPEG-2 sequence header's
//! `frame_rate_code` (which is mandatory), IVF's timestamps — and otherwise
//! [`DEFAULT_FRAME_RATE`], 25 frames a second: the PAL rate, the rate the
//! lowest MPEG-2 `frame_rate_code` that is a whole number names, a value any
//! encoder and player accept. It is a guess and says so in the log; the
//! `input-fps` setting replaces it (and a stated one) for a raw stream.

mod annexb;
mod bits;
mod ivf;
mod mpegv;
mod obu;
#[cfg(test)]
mod tests;

use anyhow::{Result, bail};
use bytes::Bytes;
use frame::{ColorSpace, PixelFormat, StreamInfo};

use crate::demux::AudioTrack;
use crate::sniff::ContainerKind;
use crate::streaming::{DemuxHeader, Sample, StreamingDemuxer};

/// The frame rate of a raw stream that states none (see the module docs).
pub const DEFAULT_FRAME_RATE: f64 = 25.0;

/// The elementary-stream family `data` opens with, when a chain of its
/// headers parses and agrees; `None` otherwise. IVF first (its magic is
/// unambiguous), then MPEG video (a sequence header start code), the two
/// AV1 OBU formats, and Annex-B H.264 / HEVC, which are told apart by what
/// their NAL unit headers mean.
pub(crate) fn sniff(data: &[u8]) -> Option<ContainerKind> {
    if ivf::sniff(data) {
        return Some(ContainerKind::Ivf);
    }
    if mpegv::sniff(data) {
        return Some(ContainerKind::MpegVideoEs);
    }
    if obu::sniff(data).is_some() {
        return Some(ContainerKind::Av1Obu);
    }
    annexb::sniff(data)
}

/// One sample: a span of the input, or bytes rewritten from it (an Annex-B
/// AV1 temporal unit in the low-overhead form, an MPEG-2 field pair).
enum EsSample {
    Span(std::ops::Range<usize>),
    Owned(Vec<u8>),
}

/// A raw elementary stream, indexed: one sample per coded picture.
pub struct EsStreamingDemuxer {
    data: Bytes,
    header: DemuxHeader,
    samples: std::vec::IntoIter<(EsSample, i64)>,
    /// Ticks one frame lasts, on `header.timescale`.
    frame_ticks: u32,
}

/// What a reader found: the codec, its samples (each with its timestamp on
/// `timescale`, `None` to number them by the frame rate), the frame rate
/// and whether the stream stated it, the frame count the decoder will make,
/// and the access unit its parameters (sequence header, SPS) came from.
struct Indexed {
    codec: &'static str,
    samples: Vec<EsSample>,
    pts: Option<(Vec<i64>, u32)>,
    frame_rate: Option<f64>,
    frames: u64,
    /// Dimensions the format states outside the bitstream (IVF's header).
    dims: Option<(u32, u32)>,
    label: &'static str,
}

pub(crate) fn demux_es_streaming_init(data: Bytes) -> Result<EsStreamingDemuxer> {
    let indexed = match sniff(&data) {
        Some(ContainerKind::H264Es) => annexb::index(&data, annexb::Codec::H264)?,
        Some(ContainerKind::HevcEs) => annexb::index(&data, annexb::Codec::Hevc)?,
        Some(ContainerKind::Ivf) => ivf::index(&data)?,
        Some(ContainerKind::Av1Obu) => obu::index(&data)?,
        Some(ContainerKind::MpegVideoEs) => mpegv::index(&data)?,
        _ => bail!("not a video elementary stream rivet reads"),
    };
    build(data, indexed)
}

fn build(data: Bytes, ix: Indexed) -> Result<EsStreamingDemuxer> {
    let Indexed {
        codec,
        samples,
        pts,
        frame_rate,
        frames,
        dims,
        label,
    } = ix;
    if samples.is_empty() {
        bail!("{label}: the stream holds no picture");
    }
    let frame_rate = match frame_rate {
        Some(r) if r.is_finite() && (1.0..=1000.0).contains(&r) => r,
        _ => {
            tracing::warn!(
                container = label,
                fps = DEFAULT_FRAME_RATE,
                "the elementary stream states no frame rate; assuming {DEFAULT_FRAME_RATE} fps \
                 (input-fps sets it)"
            );
            DEFAULT_FRAME_RATE
        }
    };
    let bytes_of = |s: &EsSample| -> Vec<u8> {
        match s {
            EsSample::Span(r) => data[r.clone()].to_vec(),
            EsSample::Owned(v) => v.clone(),
        }
    };
    // The parameters: the first sample, which every reader starts on one
    // that carries them (an SPS, a sequence header, a key frame).
    let first = bytes_of(&samples[0]);
    let codec_s = codec.to_string();
    let (width, height) = frame::pixel_format::detect_dims(codec, std::slice::from_ref(&first))
        .or(dims)
        .or_else(|| obu::dims(codec, &first))
        .or_else(|| ivf::vp8_dims(codec, &first))
        .unwrap_or((0, 0));
    if width == 0 || height == 0 {
        bail!("{label}: the stream's dimensions could not be read from its headers");
    }
    let duration = frames as f64 / frame_rate;
    let mut info = StreamInfo {
        codec: codec_s.clone(),
        width,
        height,
        frame_rate,
        duration,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        total_frames: frames,
        bitrate: if duration > 0.0 {
            (data.len() as f64 * 8.0 / duration) as u64
        } else {
            0
        },
        color_metadata: Default::default(),
    };
    info.pixel_format = frame::pixel_format::detect(codec, std::slice::from_ref(&first));
    // The colour and the sample shape are the bitstream's: there is nothing
    // else to state them. The window reads past a first access unit that
    // lacks the SEIs.
    let head = crate::demux::hdr::colour_window(
        codec,
        samples
            .iter()
            .take(16)
            .map(bytes_of)
            .collect::<Vec<_>>()
            .iter()
            .map(Vec::as_slice),
        label,
    );
    let head_au = head
        .as_ref()
        .map(|h| h.annexb.clone())
        .unwrap_or_else(|| first.clone());
    let sample_aspect = crate::demux::aspect::resolve(
        None,
        || crate::demux::aspect::from_bitstream(codec, &[], Some(&head_au), width, height),
        label,
    );
    crate::demux::hdr::resolve_source_colour(
        &mut info,
        Default::default(),
        codec,
        &[],
        Some(&head_au),
        label,
    );

    let (timescale, frame_ticks, stamps) = match pts {
        Some((stamps, timescale)) => {
            let ticks = (f64::from(timescale) / frame_rate).round().max(1.0) as u32;
            (timescale, ticks, stamps)
        }
        None => {
            const TIMESCALE: u32 = 90_000;
            let ticks = f64::from(TIMESCALE) / frame_rate;
            let stamps = (0..samples.len())
                .map(|i| (i as f64 * ticks).round() as i64)
                .collect();
            (TIMESCALE, ticks.round().max(1.0) as u32, stamps)
        }
    };
    tracing::info!(
        container = label,
        codec,
        width,
        height,
        fps = frame_rate,
        frames,
        "elementary stream indexed"
    );
    Ok(EsStreamingDemuxer {
        header: DemuxHeader {
            codec: codec_s,
            info,
            timescale,
            rotation_degrees: 0,
            sample_aspect,
        },
        samples: samples
            .into_iter()
            .zip(stamps)
            .collect::<Vec<_>>()
            .into_iter(),
        frame_ticks,
        data,
    })
}

impl StreamingDemuxer for EsStreamingDemuxer {
    fn header(&self) -> &DemuxHeader {
        &self.header
    }

    fn next_video_sample(&mut self) -> Result<Option<Sample>> {
        let Some((sample, pts_ticks)) = self.samples.next() else {
            return Ok(None);
        };
        let data = match sample {
            EsSample::Span(r) => self.data[r].to_vec(),
            EsSample::Owned(v) => v,
        };
        Ok(Some(Sample {
            data,
            pts_ticks,
            duration_ticks: self.frame_ticks,
        }))
    }

    /// An elementary stream is one stream: no audio.
    fn audio(&self) -> Option<&AudioTrack> {
        None
    }
}
