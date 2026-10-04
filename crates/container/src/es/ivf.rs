//! IVF, the WebM project's frame file for VP8, VP9 and AV1.
//!
//! A 32-byte header — `DKIF`, a version (0), the header's length (32), the
//! codec's fourcc (`VP80`, `VP90`, `AV01`), the width and height, the time
//! base as a rate and a scale (a timestamp `t` is at `t * scale / rate`
//! seconds), the frame count — then each frame behind a 12-byte header: its
//! size (32-bit) and its timestamp (64-bit), little-endian throughout.
//!
//! A frame is what the codec's own decoder takes: a VP8 frame (RFC 6386), a
//! VP9 frame or superframe, an AV1 temporal unit in the low-overhead format
//! (its temporal delimiter is dropped, as an MP4 or Matroska sample holds
//! none). The frame rate is the time base over the timestamps' usual step.

use anyhow::{Result, bail};

use super::{EsSample, Indexed};

const HEADER: usize = 32;
const FRAME_HEADER: usize = 12;

pub(super) fn sniff(data: &[u8]) -> bool {
    data.len() >= HEADER
        && &data[..4] == b"DKIF"
        && u16::from_le_bytes([data[4], data[5]]) == 0
        && usize::from(u16::from_le_bytes([data[6], data[7]])) >= HEADER
}

pub(super) fn index(data: &[u8]) -> Result<Indexed> {
    let u16_at = |at: usize| u16::from_le_bytes([data[at], data[at + 1]]);
    let u32_at = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
    let fourcc: [u8; 4] = data[8..12].try_into().unwrap();
    let codec = match &fourcc.to_ascii_uppercase()[..] {
        b"VP80" => "vp8",
        b"VP90" => "vp9",
        b"AV01" => "av1",
        other => bail!(
            "IVF: unsupported codec fourcc {:?}",
            String::from_utf8_lossy(other)
        ),
    };
    let (width, height) = (u32::from(u16_at(12)), u32::from(u16_at(14)));
    let (rate, scale) = (u32_at(16), u32_at(20));
    let mut pos = usize::from(u16_at(6));
    let mut samples = Vec::new();
    let mut stamps = Vec::new();
    let mut shown = 0u64;
    while pos + FRAME_HEADER <= data.len() {
        let size = u32_at(pos) as usize;
        let pts = i64::from_le_bytes(data[pos + 4..pos + 12].try_into().unwrap());
        let body = pos + FRAME_HEADER;
        let Some(end) = body.checked_add(size).filter(|&e| e <= data.len()) else {
            tracing::warn!(
                offset = pos,
                size,
                "IVF: a frame runs past the end of the file; truncated there"
            );
            break;
        };
        pos = end;
        if size == 0 {
            continue;
        }
        let start = if codec == "av1" {
            body + super::obu::leading_temporal_delimiter(&data[body..end])
        } else {
            body
        };
        // A VP8 frame with show_frame clear (RFC 6386 §9.1) makes no
        // picture — libvpx writes an alt-ref that way.
        if codec != "vp8" || (data[body] >> 4) & 1 == 1 {
            shown += 1;
        }
        samples.push(EsSample::Span(start..end));
        stamps.push(pts);
    }
    let frame_rate = typical_step(&stamps)
        .filter(|_| rate > 0 && scale > 0)
        .map(|step| f64::from(rate) / (f64::from(scale) * step as f64));
    // Timestamps on the time base's ticks: `t * scale` per second at `rate`.
    let pts = (rate > 0 && scale > 0).then(|| {
        (
            stamps
                .iter()
                .map(|&t| t.saturating_mul(i64::from(scale)))
                .collect(),
            rate,
        )
    });
    Ok(Indexed {
        codec,
        samples,
        pts,
        frame_rate,
        frames: shown,
        dims: (width > 0 && height > 0).then_some((width, height)),
        label: "ivf",
    })
}

/// The median step between consecutive timestamps, when there are any.
fn typical_step(stamps: &[i64]) -> Option<i64> {
    let mut steps: Vec<i64> = stamps
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d > 0)
        .collect();
    if steps.is_empty() {
        return None;
    }
    steps.sort_unstable();
    Some(steps[steps.len() / 2])
}

/// The dimensions a VP8 key frame states (RFC 6386 §9.1: the start code
/// `9d 01 2a`, then 14-bit width and height, each with two scaling bits).
pub(super) fn vp8_dims(codec: &str, frame: &[u8]) -> Option<(u32, u32)> {
    if codec != "vp8" || frame.len() < 10 || frame[0] & 1 != 0 || frame[3..6] != [0x9d, 0x01, 0x2a]
    {
        return None;
    }
    let w = u32::from(u16::from_le_bytes([frame[6], frame[7]]) & 0x3fff);
    let h = u32::from(u16::from_le_bytes([frame[8], frame[9]]) & 0x3fff);
    (w > 0 && h > 0).then_some((w, h))
}
