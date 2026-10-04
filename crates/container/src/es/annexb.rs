//! H.264 and HEVC Annex-B byte streams: NAL units behind `00 00 01` start
//! codes (H.264 Annex B / H.265 Annex B), cut into one sample per coded
//! frame.
//!
//! # Telling the two apart
//!
//! Both are start codes and NAL units, so the start codes say nothing. The
//! NAL unit headers do: an H.264 header is one byte (`forbidden_zero_bit`,
//! `nal_ref_idc`, `nal_unit_type`), an HEVC header two (`forbidden_zero_bit`,
//! six bits of type, six of layer, three of `nuh_temporal_id_plus1`, which
//! is never zero). A stream is taken as one of them only when every NAL unit
//! at its head is a legal header of that codec, a sequence parameter set
//! among them parses with the codec's own parser, and a coded slice follows.
//! HEVC's VPS (`40 01`) reads in H.264 as type 0, which is unspecified; its
//! access unit delimiter (`46 01`) as an SEI with a nonzero `nal_ref_idc`,
//! which H.264 §7.4.1 forbids — so the two readings exclude each other.
//!
//! # Access units
//!
//! A sample is one access unit: the NAL units of one primary coded picture
//! and the non-VCL units ahead of it. A new one starts (H.264 §7.4.1.2.3,
//! H.265 §7.4.2.4.4) at the first of the parameter sets, SEI, delimiter and
//! reserved units that precede the first slice of a new picture — a slice
//! with `first_mb_in_slice` 0 (H.264), or `first_slice_segment_in_pic_flag`
//! set (HEVC). An H.264 field pair is one frame, so the second field (the
//! same `frame_num`, the other parity) joins its first field's sample, as
//! the transport stream reader joins them; the decoders make one picture of
//! the pair.

use std::collections::HashMap;

use anyhow::{Result, bail};

use super::bits::Bits;
use super::{EsSample, Indexed};
use crate::sniff::ContainerKind;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Codec {
    H264,
    Hevc,
}

/// One NAL unit: where its start code begins (a zero byte before a
/// three-byte start code counts with it), and the unit's own bytes.
struct Nal {
    start: usize,
    body: std::ops::Range<usize>,
}

/// The NAL units of `data`, at most `limit` of them.
fn nals(data: &[u8], limit: usize) -> Vec<Nal> {
    let mut codes = Vec::new();
    let mut i = 0usize;
    while i + 2 < data.len() && codes.len() <= limit {
        if data[i + 2] > 1 {
            i += 3;
        } else if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            codes.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(codes.len());
    for (k, &sc) in codes.iter().enumerate() {
        let begin = sc + 3;
        let mut end = codes.get(k + 1).copied().unwrap_or(data.len());
        // trailing_zero_8bits (and the leading zero of a four-byte start
        // code) belong to no NAL unit.
        while end > begin && data[end - 1] == 0 {
            end -= 1;
        }
        let start = if sc > 0 && data[sc - 1] == 0 {
            sc - 1
        } else {
            sc
        };
        out.push(Nal {
            start,
            body: begin..end,
        });
    }
    if out.len() > limit {
        out.truncate(limit);
    }
    out
}

/// The RBSP of a NAL unit's payload (after its header), at most `max` bytes
/// of it — enough for the header fields read here.
fn rbsp(payload: &[u8], max: usize) -> Vec<u8> {
    h26x::nal::unescape_rbsp(&payload[..payload.len().min(max)])
}

/// How many NAL units the sniff reads, and how far into the stream.
const SNIFF_NALS: usize = 48;
const SNIFF_BYTES: usize = 1 << 20;

pub(super) fn sniff(data: &[u8]) -> Option<ContainerKind> {
    // A byte stream opens with a start code (after any zero_bytes).
    let lead = data.iter().take(64).take_while(|&&b| b == 0).count();
    if lead < 2 || data.get(lead) != Some(&1) {
        return None;
    }
    let head = &data[..data.len().min(SNIFF_BYTES)];
    let units = nals(head, SNIFF_NALS);
    // The last unit may be cut by the window; it is not judged.
    let judged = if units.len() > 1 && head.len() < data.len() {
        &units[..units.len() - 1]
    } else {
        &units[..]
    };
    if judged.len() < 2 {
        return None;
    }
    if looks_h264(head, judged) {
        Some(ContainerKind::H264Es)
    } else if looks_hevc(head, judged) {
        Some(ContainerKind::HevcEs)
    } else {
        None
    }
}

fn looks_h264(data: &[u8], units: &[Nal]) -> bool {
    let mut sps = false;
    let mut slice_after_sps = false;
    for nal in units {
        let body = &data[nal.body.clone()];
        let Some(&h) = body.first() else { return false };
        let (forbidden, ref_idc, kind) = (h >> 7, (h >> 5) & 3, h & 0x1f);
        if forbidden != 0 || kind == 0 || kind >= 24 {
            return false;
        }
        match kind {
            // SEI, AUD, end of sequence / stream, filler: nal_ref_idc 0.
            6 | 9..=12 if ref_idc != 0 => return false,
            // An IDR slice is a reference picture.
            5 if ref_idc == 0 => return false,
            7 => {
                let Ok(s) = h26x::h264::Sps::parse(&rbsp(&body[1..], 512)) else {
                    return false;
                };
                if s.pic_width_in_mbs == 0 {
                    return false;
                }
                sps = true;
            }
            1 | 5 if sps => slice_after_sps = true,
            _ => {}
        }
    }
    sps && slice_after_sps
}

fn looks_hevc(data: &[u8], units: &[Nal]) -> bool {
    let (mut vps, mut sps, mut slice_after_sps) = (false, false, false);
    for nal in units {
        let body = &data[nal.body.clone()];
        let Some(h) = h26x::nal::HevcNalHeader::parse(body) else {
            return false;
        };
        // Reserved VCL types (22..=31 are reserved or IRAP-reserved, kept
        // legal up to 23) and reserved non-VCL types 41..=47.
        if (24..=31).contains(&h.unit_type) || (41..=47).contains(&h.unit_type) {
            return false;
        }
        match h.unit_type {
            // video_parameter_set_rbsp(): sixteen bits of ids and counts, then
            // vps_reserved_0xffff_16bits (H.265 §7.3.2.1).
            32 => vps = rbsp(&body[2..], 8).get(2..4) == Some(&[0xff, 0xff][..]),
            33 => {
                let Ok(s) = h26x::hevc::Sps::parse(&rbsp(&body[2..], 1024)) else {
                    return false;
                };
                if s.width == 0 || s.height == 0 {
                    return false;
                }
                sps = true;
            }
            0..=21 if sps => slice_after_sps = true,
            _ => {}
        }
    }
    vps && sps && slice_after_sps
}

/// What an H.264 slice header's head says about which frame it belongs to.
#[derive(Clone, Copy)]
struct SliceHead {
    first_mb: u32,
    frame_num: u32,
    field: bool,
    bottom: bool,
}

/// The `seq_parameter_set` fields a slice header's head depends on.
#[derive(Clone, Copy)]
struct SpsFields {
    log2_max_frame_num: u32,
    frame_mbs_only: bool,
    separate_colour_plane: bool,
}

fn h264_slice_head(
    payload: &[u8],
    pps: &HashMap<u32, u32>,
    sps: &HashMap<u32, SpsFields>,
) -> Option<SliceHead> {
    let rb = rbsp(payload, 64);
    let mut r = Bits::new(&rb);
    let first_mb = r.ue()?;
    r.ue()?; // slice_type
    let pps_id = r.ue()?;
    let s = sps.get(pps.get(&pps_id)?)?;
    if s.separate_colour_plane {
        r.bits(2)?;
    }
    let frame_num = r.bits(s.log2_max_frame_num)?;
    let field = !s.frame_mbs_only && r.bit()? == 1;
    let bottom = field && r.bit()? == 1;
    Some(SliceHead {
        first_mb,
        frame_num,
        field,
        bottom,
    })
}

/// The stream's access units as samples, from the first that carries a
/// sequence parameter set (what comes before it cannot be decoded), and the
/// frame rate its first SPS's VUI timing states.
pub(super) fn index(data: &[u8], codec: Codec) -> Result<Indexed> {
    let units = nals(data, usize::MAX);
    let mut cuts: Vec<usize> = Vec::new();
    // Where the non-VCL units ahead of the next picture began.
    let mut lead: Option<usize> = None;
    let mut frame_rate: Option<f64> = None;
    let mut seen_sps = false;
    // H.264 parameter sets for slice parsing, and an unpaired first field.
    let mut sps_fields: HashMap<u32, SpsFields> = HashMap::new();
    let mut pps_sps: HashMap<u32, u32> = HashMap::new();
    let mut open_field: Option<(u32, bool)> = None;
    for nal in &units {
        let body = &data[nal.body.clone()];
        let (opens_au, picture_start) = match codec {
            Codec::H264 => {
                let Some(&h) = body.first() else { continue };
                match h & 0x1f {
                    7 => {
                        if let Ok(s) = h26x::h264::Sps::parse(&rbsp(&body[1..], 512)) {
                            if frame_rate.is_none() {
                                frame_rate =
                                    s.vui.as_ref().and_then(|v| v.timing).and_then(|(n, t)| {
                                        (n > 0 && t > 0)
                                            .then(|| f64::from(t) / (2.0 * f64::from(n)))
                                    });
                            }
                            sps_fields.insert(
                                s.id,
                                SpsFields {
                                    log2_max_frame_num: s.log2_max_frame_num,
                                    frame_mbs_only: s.frame_mbs_only,
                                    separate_colour_plane: s.separate_colour_plane,
                                },
                            );
                        }
                        seen_sps = true;
                        (true, false)
                    }
                    8 => {
                        let rb = rbsp(&body[1..], 16);
                        let mut r = Bits::new(&rb);
                        if let (Some(p), Some(s)) = (r.ue(), r.ue()) {
                            pps_sps.insert(p, s);
                        }
                        (true, false)
                    }
                    6 | 9 | 13..=18 => (true, false),
                    1 | 5 => match h264_slice_head(&body[1..], &pps_sps, &sps_fields) {
                        Some(sh) if sh.first_mb == 0 => {
                            // The second field of a pair continues its frame.
                            let second = sh.field
                                && open_field.is_some_and(|(num, bottom)| {
                                    num == sh.frame_num && bottom != sh.bottom
                                });
                            open_field = (sh.field && !second).then_some((sh.frame_num, sh.bottom));
                            (false, !second)
                        }
                        _ => (false, false),
                    },
                    _ => (false, false),
                }
            }
            Codec::Hevc => {
                let Some(h) = h26x::nal::HevcNalHeader::parse(body) else {
                    continue;
                };
                match h.unit_type {
                    33 => {
                        if frame_rate.is_none()
                            && let Ok(s) = h26x::hevc::Sps::parse(&rbsp(&body[2..], 1024))
                        {
                            frame_rate =
                                s.vui.as_ref().and_then(|v| v.timing).and_then(|(n, t)| {
                                    (n > 0 && t > 0).then(|| f64::from(t) / f64::from(n))
                                });
                        }
                        seen_sps = true;
                        (true, false)
                    }
                    32 | 34 | 35 | 39 | 41..=44 | 48..=55 => (true, false),
                    0..=31 => (false, body.get(2).is_some_and(|b| b & 0x80 != 0)),
                    _ => (false, false),
                }
            }
        };
        if opens_au {
            lead.get_or_insert(nal.start);
        } else if picture_start {
            cuts.push(lead.take().unwrap_or(nal.start));
        } else if matches!(codec, Codec::H264)
            && body.first().is_some_and(|h| matches!(h & 0x1f, 1 | 5))
            || matches!(codec, Codec::Hevc) && body.first().is_some_and(|h| (h >> 1) & 0x3f <= 31)
        {
            // A further slice of the picture: anything gathered before it
            // was inside this access unit after all.
            lead = None;
        }
    }
    let label = match codec {
        Codec::H264 => "h264",
        Codec::Hevc => "hevc",
    };
    if !seen_sps {
        bail!("{label}: the stream has no sequence parameter set");
    }
    // Samples from the first access unit holding an SPS.
    let mut spans: Vec<std::ops::Range<usize>> = cuts
        .iter()
        .enumerate()
        .map(|(k, &c)| c..cuts.get(k + 1).copied().unwrap_or(data.len()))
        .collect();
    let has_sps = |r: &std::ops::Range<usize>| {
        nals(&data[r.clone()], 8).iter().any(|n| {
            let b = &data[r.start + n.body.start..r.start + n.body.end];
            match codec {
                Codec::H264 => b.first().is_some_and(|h| h & 0x1f == 7),
                Codec::Hevc => b.first().is_some_and(|h| (h >> 1) & 0x3f == 33),
            }
        })
    };
    let skip = spans.iter().position(has_sps).unwrap_or(spans.len());
    if skip > 0 {
        tracing::warn!(
            container = label,
            dropped = skip,
            "the stream opens before its first SPS; those access units are dropped"
        );
    }
    spans.drain(..skip);
    Ok(Indexed {
        codec: match codec {
            Codec::H264 => "h264",
            Codec::Hevc => "h265",
        },
        frames: spans.len() as u64,
        samples: spans.into_iter().map(EsSample::Span).collect(),
        pts: None,
        frame_rate,
        dims: None,
        label,
    })
}
