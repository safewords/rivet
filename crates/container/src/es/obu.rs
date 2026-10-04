//! AV1 OBU streams (`.obu`), in either of the two formats the AV1 Bitstream
//! & Decoding Process Specification defines:
//!
//! - the **low-overhead bitstream format** (§5.2): OBUs back to back, each
//!   with `obu_has_size_field` set, a temporal delimiter opening every
//!   temporal unit;
//! - the **length-delimited format** of Annex B: `temporal_unit(size)`s,
//!   each a run of `frame_unit(size)`s, each a run of OBUs behind their
//!   `obu_length` — all sizes `leb128()`.
//!
//! A sample is one temporal unit — exactly one shown frame (§7.5) — in the
//! low-overhead form, without its temporal delimiter: what an MP4 `av01` or
//! a Matroska `V_AV1` sample holds and what rivet's AV1 decoders take. An
//! Annex-B OBU without a size field gets one.

use anyhow::{Result, bail};

use super::bits::{Bits, leb128, write_leb128};
use super::{EsSample, Indexed};

const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_TEMPORAL_DELIMITER: u8 = 2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Format {
    LowOverhead,
    AnnexB,
}

/// An OBU header (§5.3.1): its type, whether it carries a size field, and
/// its length (1, or 2 with the extension byte). `None` when the forbidden
/// or reserved bit is set or the type is reserved.
fn header(data: &[u8]) -> Option<(u8, bool, usize)> {
    let h = *data.first()?;
    let kind = (h >> 3) & 0xf;
    if h & 0x80 != 0 || h & 0x01 != 0 || !matches!(kind, 1..=8 | 15) {
        return None;
    }
    let len = if h & 0x04 != 0 { 2 } else { 1 };
    (data.len() >= len).then_some((kind, h & 0x02 != 0, len))
}

/// One low-overhead OBU at the head of `data`: its type and total length.
/// An OBU without a size field runs to the end of `data`.
fn low_overhead_obu(data: &[u8]) -> Option<(u8, usize)> {
    let (kind, has_size, hlen) = header(data)?;
    if !has_size {
        return Some((kind, data.len()));
    }
    let (size, n) = leb128(&data[hlen..])?;
    let total = hlen
        .checked_add(n)?
        .checked_add(usize::try_from(size).ok()?)?;
    (total <= data.len()).then_some((kind, total))
}

/// How many bytes a temporal delimiter at the head of `tu` takes (0 when it
/// opens with something else).
pub(crate) fn leading_temporal_delimiter(tu: &[u8]) -> usize {
    match low_overhead_obu(tu) {
        Some((OBU_TEMPORAL_DELIMITER, len)) if len < tu.len() => len,
        _ => 0,
    }
}

pub(super) fn sniff(data: &[u8]) -> Option<Format> {
    if sniff_low_overhead(data) {
        Some(Format::LowOverhead)
    } else if annex_b_temporal_unit(data, 0).is_some_and(|(tu, _)| has_sequence_header(&tu)) {
        Some(Format::AnnexB)
    } else {
        None
    }
}

/// A temporal delimiter with no payload first, then OBUs whose headers and
/// sizes all hold, a sequence header that parses among them.
fn sniff_low_overhead(data: &[u8]) -> bool {
    let Some((OBU_TEMPORAL_DELIMITER, 2)) = low_overhead_obu(data).filter(|_| data[0] & 0x02 != 0)
    else {
        return false;
    };
    let mut pos = 2;
    for _ in 0..8 {
        if pos >= data.len() {
            break;
        }
        // Every OBU of the format carries its size.
        if data[pos] & 0x02 == 0 {
            return false;
        }
        let Some((kind, len)) = low_overhead_obu(&data[pos..]) else {
            return false;
        };
        if kind == OBU_SEQUENCE_HEADER {
            return has_sequence_header(&data[pos..pos + len]);
        }
        pos += len;
    }
    false
}

fn has_sequence_header(low_overhead: &[u8]) -> bool {
    frame::pixel_format::parse_av1_sequence_header(low_overhead).is_some()
}

/// The Annex-B temporal unit at `pos`, its OBUs rewritten in the
/// low-overhead form (temporal delimiter dropped), with where the next one
/// starts. `None` when its sizes do not nest exactly.
fn annex_b_temporal_unit(data: &[u8], pos: usize) -> Option<(Vec<u8>, usize)> {
    let (tu_size, n) = leb128(data.get(pos..)?)?;
    let start = pos + n;
    let end = start.checked_add(usize::try_from(tu_size).ok()?)?;
    if tu_size == 0 || end > data.len() {
        return None;
    }
    let mut out = Vec::with_capacity(end - start + 16);
    let mut at = start;
    while at < end {
        let (fu_size, n) = leb128(&data[at..end])?;
        let fu_start = at + n;
        let fu_end = fu_start.checked_add(usize::try_from(fu_size).ok()?)?;
        if fu_size == 0 || fu_end > end {
            return None;
        }
        let mut o = fu_start;
        while o < fu_end {
            let (obu_len, n) = leb128(&data[o..fu_end])?;
            let obu_start = o + n;
            let obu_end = obu_start.checked_add(usize::try_from(obu_len).ok()?)?;
            if obu_len == 0 || obu_end > fu_end {
                return None;
            }
            let obu = &data[obu_start..obu_end];
            let (kind, has_size, hlen) = header(obu)?;
            if kind != OBU_TEMPORAL_DELIMITER {
                if has_size {
                    out.extend_from_slice(obu);
                } else {
                    out.push(obu[0] | 0x02);
                    out.extend_from_slice(&obu[1..hlen]);
                    write_leb128((obu.len() - hlen) as u64, &mut out);
                    out.extend_from_slice(&obu[hlen..]);
                }
            }
            o = obu_end;
        }
        at = fu_end;
    }
    Some((out, end))
}

pub(super) fn index(data: &[u8]) -> Result<Indexed> {
    let mut samples = Vec::new();
    match sniff(data) {
        Some(Format::LowOverhead) => {
            // A temporal unit runs from one temporal delimiter to the next.
            let mut pos = 0;
            let mut tu_start: Option<usize> = None;
            while pos < data.len() {
                let Some((kind, len)) = low_overhead_obu(&data[pos..]) else {
                    tracing::warn!(
                        offset = pos,
                        "AV1 OBU stream: an OBU does not parse; the stream ends there"
                    );
                    break;
                };
                if kind == OBU_TEMPORAL_DELIMITER {
                    if let Some(s) = tu_start.take().filter(|&s| s < pos) {
                        samples.push(EsSample::Span(s..pos));
                    }
                    tu_start = Some(pos + len);
                }
                pos += len;
            }
            if let Some(s) = tu_start.filter(|&s| s < pos) {
                samples.push(EsSample::Span(s..pos));
            }
        }
        Some(Format::AnnexB) => {
            let mut pos = 0;
            while pos < data.len() {
                let Some((tu, next)) = annex_b_temporal_unit(data, pos) else {
                    tracing::warn!(
                        offset = pos,
                        "AV1 Annex B: a temporal unit's sizes do not nest; the stream ends there"
                    );
                    break;
                };
                if !tu.is_empty() {
                    samples.push(EsSample::Owned(tu));
                }
                pos = next;
            }
        }
        None => bail!("not an AV1 OBU stream"),
    }
    // From the first temporal unit with a sequence header: before it the
    // stream cannot be decoded.
    let first_with_header = samples
        .iter()
        .position(|s| match s {
            EsSample::Span(r) => has_sequence_header(&data[r.clone()]),
            EsSample::Owned(v) => has_sequence_header(v),
        })
        .unwrap_or(samples.len());
    samples.drain(..first_with_header);
    let frame_rate = samples.first().and_then(|s| match s {
        EsSample::Span(r) => timing_frame_rate(&data[r.clone()]),
        EsSample::Owned(v) => timing_frame_rate(v),
    });
    Ok(Indexed {
        codec: "av1",
        frames: samples.len() as u64,
        samples,
        pts: None,
        frame_rate,
        dims: None,
        label: "obu",
    })
}

/// The frame rate a sequence header's `timing_info()` states (§5.5.1,
/// §5.5.3): `time_scale / num_units_in_display_tick`, over the ticks per
/// picture when `equal_picture_interval` gives them. `None` without timing
/// info.
fn timing_frame_rate(tu: &[u8]) -> Option<f64> {
    let mut pos = 0;
    while pos < tu.len() {
        let (kind, len) = low_overhead_obu(&tu[pos..])?;
        if kind == OBU_SEQUENCE_HEADER {
            let (_, _, hlen) = header(&tu[pos..])?;
            let (_, n) = leb128(&tu[pos + hlen..])?;
            let mut r = Bits::new(&tu[pos + hlen + n..pos + len]);
            r.bits(3)?; // seq_profile
            r.bit()?; // still_picture
            if r.bit()? == 1 {
                return None; // reduced_still_picture_header
            }
            if r.bit()? == 0 {
                return None; // timing_info_present_flag
            }
            let units = r.bits(32)?;
            let scale = r.bits(32)?;
            let ticks = if r.bit()? == 1 {
                u64::from(r.uvlc()?) + 1
            } else {
                1
            };
            return (units > 0 && scale > 0)
                .then(|| f64::from(scale) / (f64::from(units) * ticks as f64));
        }
        pos += len;
    }
    None
}

/// The largest frame a sequence header allows, for an AV1 sample.
pub(super) fn dims(codec: &str, sample: &[u8]) -> Option<(u32, u32)> {
    if codec != "av1" {
        return None;
    }
    let seq = frame::pixel_format::parse_av1_sequence_header(sample)?;
    Some((
        seq.max_frame_width_minus1 + 1,
        seq.max_frame_height_minus1 + 1,
    ))
}
