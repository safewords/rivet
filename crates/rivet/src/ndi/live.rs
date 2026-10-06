//! A live source of pictures and sound, and the NDI receiver as one.
//!
//! The recorder ([`super::record`]) reads a [`LiveSource`], not NDI, so it
//! is tested on synthetic sources and the NDI specifics stay here: turning
//! a received frame into a [`VideoFrame`] in one of the planar layouts the
//! colorspace layer normalises, deciding the colour NDI implies, and
//! putting every event on one clock.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use codec::frame::{ColorMetadata, ColorSpace, PixelFormat, TransferFn, VideoFrame};

/// 100 ns ticks per second: NDI's time unit.
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
    /// The source has ended (a finite source; NDI never says this).
    End,
}

/// A live source the recorder pulls from on a thread of its own.
pub trait LiveSource: Send {
    /// The next event, waiting up to `timeout`; `None` when nothing came.
    /// An error ends the recording: [`SourceLost`] as the source going
    /// away (what was recorded is kept), anything else as a failure.
    fn next_event(&mut self, timeout: Duration) -> Result<Option<LiveEvent>>;

    /// The source's name, for messages.
    fn name(&self) -> String;
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

/// The NDI receiver as a [`LiveSource`].
pub struct NdiSource {
    receiver: ndi::Receiver,
}

impl NdiSource {
    /// Find the source `query` names (see [`ndi::Ndi::find_source`]) and
    /// connect to it.
    pub fn connect(
        query: &str,
        find: &ndi::FindOptions,
        options: &ndi::ReceiverOptions,
        wait: Duration,
    ) -> Result<Self> {
        let runtime = ndi::Ndi::load()?;
        tracing::info!(runtime = %runtime.version(), path = %runtime.path().display(), "NDI runtime loaded");
        let source = runtime.find_source(query, find, wait)?;
        tracing::info!(source = %source.name, url = ?source.url, "NDI source found");
        let receiver = runtime.receiver(&source, options)?;
        Ok(Self { receiver })
    }
}

impl LiveSource for NdiSource {
    fn next_event(&mut self, timeout: Duration) -> Result<Option<LiveEvent>> {
        let captured = match self.receiver.capture(timeout) {
            Ok(c) => c,
            Err(ndi::Error::ConnectionLost) => return Err(SourceLost.into()),
            Err(e) => return Err(e.into()),
        };
        Ok(match captured {
            ndi::Capture::Video(v) => {
                let picture = v.to_picture().context("reading an NDI video frame")?;
                let color = ndi_color(&picture, v.color_info().as_ref());
                let frame = picture_to_frame(picture, &color)?;
                Some(LiveEvent::Video(LiveVideo {
                    frame,
                    color,
                    frame_rate: v.frame_rate(),
                    time: v.timestamp().unwrap_or_else(now_ticks),
                }))
            }
            ndi::Capture::Audio(a) => {
                if a.channels() == 0 || a.samples() == 0 {
                    return Ok(None);
                }
                Some(LiveEvent::Audio(LiveAudio {
                    samples: a.interleaved(),
                    sample_rate: a.sample_rate(),
                    channels: a.channels().min(u8::MAX as usize) as u8,
                    time: a.timestamp().unwrap_or_else(now_ticks),
                }))
            }
            ndi::Capture::Metadata(_)
            | ndi::Capture::StatusChange
            | ndi::Capture::SourceChange
            | ndi::Capture::None => None,
        })
    }

    fn name(&self) -> String {
        self.receiver.source().name.clone()
    }
}

/// This machine's clock in NDI's unit, for a sender that stamps nothing.
fn now_ticks() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| (d.as_nanos() / 100) as i64)
}

/// The colour an NDI picture is in. The frame's `ndi_color_info` when it
/// carries one; otherwise NDI's convention — studio-range BT.601 below 720
/// lines, BT.709 from 720 up — and full-range sRGB for RGB, which the
/// colorspace layer matrixes into BT.709.
pub fn ndi_color(picture: &ndi::Picture, info: Option<&ndi::ColorInfo>) -> ColorMetadata {
    let bt601 = || ColorMetadata {
        matrix_coefficients: 6,
        colour_primaries: 6,
        ..ColorMetadata::default()
    };
    let bt709 = || ColorMetadata {
        matrix_coefficients: 1,
        colour_primaries: 1,
        ..ColorMetadata::default()
    };
    let mut color = if picture.layout != ndi::Layout::Rgba && picture.height < 720 {
        bt601()
    } else {
        bt709()
    };
    let Some(info) = info else {
        return color;
    };
    let code = |s: &Option<String>| -> Option<u8> {
        let s = s.as_deref()?.to_ascii_lowercase();
        if s.contains("2020") || s.contains("2100") {
            Some(9)
        } else if s.contains("601") || s.contains("170") {
            Some(6)
        } else if s.contains("709") {
            Some(1)
        } else {
            None
        }
    };
    if let Some(m) = code(&info.matrix) {
        color.matrix_coefficients = m;
    }
    if let Some(p) = code(&info.primaries) {
        color.colour_primaries = p;
    }
    if let Some(t) = info.transfer.as_deref().map(str::to_ascii_lowercase) {
        color.transfer = if t.contains("pq") || t.contains("2084") {
            TransferFn::St2084
        } else if t.contains("hlg") || t.contains("b67") {
            TransferFn::AribStdB67
        } else {
            TransferFn::Bt709
        };
    }
    color
}

/// The colour-space tag a frame in `color` carries.
fn color_space(color: &ColorMetadata) -> ColorSpace {
    match color.matrix_coefficients {
        9 | 10 => ColorSpace::Bt2020,
        5 | 6 => ColorSpace::Bt601,
        _ => ColorSpace::Bt709,
    }
}

/// A received picture as a [`VideoFrame`], cropped to even dimensions (a
/// 4:2:0 or 4:2:2 encoder codes pairs of pixels).
pub fn picture_to_frame(picture: ndi::Picture, color: &ColorMetadata) -> Result<VideoFrame> {
    let format = match picture.layout {
        ndi::Layout::Yuv420p => PixelFormat::Yuv420p,
        ndi::Layout::Yuv420p10le => PixelFormat::Yuv420p10le,
        ndi::Layout::Yuv422p => PixelFormat::Yuv422p,
        ndi::Layout::Yuv422p10le => PixelFormat::Yuv422p10le,
        ndi::Layout::Nv12 => PixelFormat::Nv12,
        ndi::Layout::Rgba => PixelFormat::Rgba32,
    };
    let picture = crop_even(picture)?;
    Ok(VideoFrame::new(
        bytes::Bytes::from(picture.data),
        picture.width,
        picture.height,
        format,
        color_space(color),
        0,
    ))
}

/// `picture` with an odd last column or row dropped.
fn crop_even(picture: ndi::Picture) -> Result<ndi::Picture> {
    let (w, h) = (picture.width as usize, picture.height as usize);
    let (ew, eh) = (w & !1, h & !1);
    if ew == w && eh == h {
        return Ok(picture);
    }
    anyhow::ensure!(ew > 0 && eh > 0, "a {w}x{h} picture is too small to code");
    // (bytes per sample, chroma width divisor, chroma height divisor, planes)
    use ndi::Layout::*;
    let (bps, cw_div, ch_div) = match picture.layout {
        Yuv420p => (1, 2, 2),
        Yuv420p10le => (2, 2, 2),
        Yuv422p => (1, 2, 1),
        Yuv422p10le => (2, 2, 1),
        Nv12 => (1, 2, 2),
        Rgba => (4, 0, 0),
    };
    let mut out = Vec::with_capacity(picture.layout.len(ew, eh));
    let mut copy = |src: &[u8], row_bytes: usize, keep_bytes: usize, rows: usize| {
        for r in 0..rows {
            out.extend_from_slice(&src[r * row_bytes..r * row_bytes + keep_bytes]);
        }
    };
    let luma = w * h * bps;
    copy(&picture.data, w * bps, ew * bps, eh);
    if cw_div > 0 {
        let (cw, ch) = (w.div_ceil(cw_div), h.div_ceil(ch_div));
        let (ecw, ech) = (ew / cw_div, eh / ch_div);
        let chroma = &picture.data[luma..];
        if picture.layout == Nv12 {
            copy(chroma, 2 * cw, 2 * ecw, ech);
        } else {
            let plane = cw * ch * bps;
            copy(chroma, cw * bps, ecw * bps, ech);
            copy(&chroma[plane..], cw * bps, ecw * bps, ech);
        }
    }
    Ok(ndi::Picture {
        layout: picture.layout,
        width: ew as u32,
        height: eh as u32,
        data: out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndi_colour_follows_the_line_count_unless_the_frame_says() {
        let pic = |w, h, layout| ndi::Picture {
            layout,
            width: w,
            height: h,
            data: Vec::new(),
        };
        let sd = ndi_color(&pic(720, 486, ndi::Layout::Yuv422p), None);
        assert_eq!((sd.matrix_coefficients, sd.colour_primaries), (6, 6));
        let hd = ndi_color(&pic(1920, 1080, ndi::Layout::Yuv422p), None);
        assert_eq!((hd.matrix_coefficients, hd.colour_primaries), (1, 1));
        let rgb = ndi_color(&pic(640, 480, ndi::Layout::Rgba), None);
        assert_eq!(rgb.matrix_coefficients, 1, "RGB is matrixed into BT.709");

        let info = ndi::ColorInfo {
            transfer: Some("bt_2100_hlg".into()),
            matrix: Some("bt_2020".into()),
            primaries: Some("bt_2020".into()),
        };
        let hlg = ndi_color(&pic(3840, 2160, ndi::Layout::Yuv422p10le), Some(&info));
        assert_eq!(
            (hlg.transfer, hlg.matrix_coefficients, hlg.colour_primaries),
            (TransferFn::AribStdB67, 9, 9)
        );
        assert_eq!(color_space(&hlg), ColorSpace::Bt2020);
    }

    #[test]
    fn an_odd_picture_is_cropped_to_even_in_every_plane() {
        // 3x3 4:2:0: chroma 2x2.
        let data: Vec<u8> = (0..9).chain(100..104).chain(200..204).collect();
        let pic = ndi::Picture {
            layout: ndi::Layout::Yuv420p,
            width: 3,
            height: 3,
            data,
        };
        let f = picture_to_frame(pic, &ColorMetadata::default()).unwrap();
        assert_eq!((f.width, f.height, f.format), (2, 2, PixelFormat::Yuv420p));
        assert_eq!(&f.data[..], &[0, 1, 3, 4, 100, 200]);

        let nv12 = ndi::Picture {
            layout: ndi::Layout::Nv12,
            width: 3,
            height: 2,
            data: vec![1, 2, 3, 4, 5, 6, 10, 20, 30, 40],
        };
        let f = picture_to_frame(nv12, &ColorMetadata::default()).unwrap();
        assert_eq!(&f.data[..], &[1, 2, 4, 5, 10, 20]);
    }
}
