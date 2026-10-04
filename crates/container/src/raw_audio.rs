//! Audio files with no container of their own, read for their audio alone:
//! RIFF WAVE (and RF64 / BW64), and the bare elementary streams — ADTS AAC
//! (`.aac`), AC-3 / E-AC-3 (`.ac3`, `.eac3`) and DTS (`.dts`).
//!
//! # WAVE
//!
//! A `RIFF` (or `RF64` / `BW64`) file of form `WAVE` (Microsoft's RIFF
//! multimedia specification; EBU Tech 3306 for RF64): a `fmt ` chunk holding
//! a WAVEFORMATEX, a `data` chunk holding the samples. Linear PCM
//! (`WAVE_FORMAT_PCM`, 8 to 32 bits), IEEE float (`WAVE_FORMAT_IEEE_FLOAT`,
//! 32 / 64 bits) and `WAVE_FORMAT_EXTENSIBLE` naming either are read; MPEG
//! audio and AC-3 under their WAVE tags are read as those streams; any other
//! format is surfaced by name with no packets. RF64's `ds64` chunk gives the
//! 64-bit `data` size its 32-bit field cannot; a `data` size of zero or one
//! that runs past the end (a recorder that never came back to write it) is
//! read to the end of the file. The samples are cut into packets of 4096
//! frames, each lasting its frames. The channels are taken in WAVE order
//! for their count — `dwChannelMask` is not read.
//!
//! # Elementary streams
//!
//! The frames are self-describing and are sliced exactly as the transport
//! stream reader slices them (`ts::audio`): ADTS headers stripped and an
//! AudioSpecificConfig synthesised from the first (ISO/IEC 14496-3 §1.A.2,
//! 13818-7 §6.2); AC-3 / E-AC-3 syncframes by the size their BSI states
//! (ETSI TS 102 366); DTS core frames with their extension (ETSI TS 102 114).
//! An ID3v2 tag in front of an ADTS stream is skipped. None of them states
//! an encoder delay, so the track has no edit.
//!
//! # Sniffing
//!
//! A sync word is a few bits of pattern, so an elementary stream is only
//! taken when several frames chain: each header's frame length lands on the
//! next header (three in a row, or to the end of a shorter file).

use anyhow::{Context, Result, bail};

use crate::demux::AudioTrack;
use crate::demux::audio::qt::PcmLayout;

/// How many chained frames make an elementary stream.
const CHAIN: usize = 3;

/// The length of an ID3v2 tag (and any more after it) at the start.
fn id3v2_len(data: &[u8]) -> usize {
    let mut start = 0usize;
    // id3.org v2.4 §3.1: "ID3", version, flags, a 28-bit syncsafe size that
    // excludes the 10-byte header and a 10-byte footer if flagged.
    while data.len() >= start + 10 && &data[start..start + 3] == b"ID3" {
        let s = &data[start + 6..start + 10];
        let size = s
            .iter()
            .fold(0usize, |acc, &b| (acc << 7) | usize::from(b & 0x7F));
        let footer = if data[start + 5] & 0x10 != 0 { 10 } else { 0 };
        start += 10 + size + footer;
    }
    start
}

/// Whether frames chain from `at`: `len` gives a frame's length (None when
/// no frame starts there); `CHAIN` in a row, or every one to the end.
fn chains(data: &[u8], mut at: usize, len: impl Fn(&[u8]) -> Option<usize>) -> bool {
    for n in 0..CHAIN {
        let Some(l) = data.get(at..).and_then(&len) else {
            return false;
        };
        if l == 0 {
            return false;
        }
        at += l;
        if at >= data.len() {
            // A short file: the frames it has must fill it, and there must be two.
            return at == data.len() && n >= 1;
        }
    }
    true
}

/// An ADTS header's frame length, when one starts here.
fn adts_len(b: &[u8]) -> Option<usize> {
    // syncword 0xFFF, layer '00' (ISO/IEC 13818-7 §6.2.1); a valid rate.
    if b.len() < 7 || b[0] != 0xFF || b[1] & 0xF6 != 0xF0 || (b[2] >> 2) & 0x0F > 12 {
        return None;
    }
    let len = (usize::from(b[3] & 0x03) << 11) | (usize::from(b[4]) << 3) | usize::from(b[5] >> 5);
    (len >= 7).then_some(len)
}

/// An AC-3 or E-AC-3 syncframe's length, when one starts here.
fn ac3_len(b: &[u8]) -> Option<usize> {
    if b.len() < 6 || b[0] != 0x0B || b[1] != 0x77 {
        return None;
    }
    crate::ts::audio::ac3_syncframe_len(b)
}

/// A DTS core frame's length with its extension, when one starts here.
fn dts_len(b: &[u8]) -> Option<usize> {
    let core = crate::dts_sync::parse_core_sync(b).ok()?;
    if core.frame_size < 96 {
        return None;
    }
    if crate::dts_sync::has_hd_extension(b, &core) {
        // The extension substream's header states its size (ETSI TS 102 114
        // §7.4: after the 32-bit sync, 8 bits of user data, 2 bits of
        // extension index, then a 1-bit header-size-type selecting 8 + 16 or
        // 12 + 20 bits of header and substream size minus one).
        let ext = &b[core.frame_size..];
        let bits = u64::from_be_bytes(ext.get(4..12)?.try_into().ok()?);
        let long = (bits >> 53) & 1 == 1;
        let size = if long {
            ((bits >> 21) & 0xF_FFFF) + 1
        } else {
            ((bits >> 29) & 0xFFFF) + 1
        };
        let end = core.frame_size + size as usize;
        if end == b.len() || b.get(end..end + 4) == Some(&[0x7F, 0xFE, 0x80, 0x01]) {
            return Some(end);
        }
        // Not where the header says: the next core sync ends the frame.
        let next = b
            .get(core.frame_size..)?
            .windows(4)
            .position(|w| w == [0x7F, 0xFE, 0x80, 0x01])?;
        return Some(core.frame_size + next);
    }
    Some(core.frame_size)
}

/// Whether `data` is a RIFF / RF64 / BW64 WAVE file.
pub fn sniff_wav(data: &[u8]) -> bool {
    data.len() >= 12 && matches!(&data[..4], b"RIFF" | b"RF64" | b"BW64") && &data[8..12] == b"WAVE"
}

/// Whether `data` is an ADTS stream (after any ID3v2 tag).
pub fn sniff_adts(data: &[u8]) -> bool {
    chains(data, id3v2_len(data), adts_len)
}

/// Whether `data` is an AC-3 / E-AC-3 elementary stream.
pub fn sniff_ac3(data: &[u8]) -> bool {
    chains(data, 0, ac3_len)
}

/// Whether `data` is a DTS elementary stream.
pub fn sniff_dts(data: &[u8]) -> bool {
    chains(data, 0, dts_len)
}

/// A bare ADTS AAC stream as a track.
pub fn read_adts(data: &[u8]) -> Result<AudioTrack> {
    let es = &data[id3v2_len(data).min(data.len())..];
    let (track, _) = crate::ts::audio::aac_from_adts_es(es)?.context("no ADTS frame")?;
    Ok(track)
}

/// A bare AC-3 or E-AC-3 stream as a track: the first syncframe's `bsid`
/// says which.
pub fn read_ac3(data: &[u8]) -> Result<AudioTrack> {
    let eac3 = matches!(
        crate::ac3_sync::parse_sync_info(data),
        Ok(crate::ac3_sync::SyncInfo::Eac3(_))
    );
    let found = if eac3 {
        crate::ts::audio::eac3_from_es(data)?
    } else {
        crate::ts::audio::ac3_from_es(data)?
    };
    Ok(found.context("no AC-3 syncframe")?.0)
}

/// A bare DTS stream as a track.
pub fn read_dts(data: &[u8]) -> Result<AudioTrack> {
    Ok(crate::ts::audio::dts_from_es(data)?
        .context("no DTS core frame")?
        .0)
}

/// Frames to a PCM packet.
const PCM_PACKET_FRAMES: usize = 4096;

/// A WAVE file's audio as a track.
pub fn read_wav(data: &[u8]) -> Result<AudioTrack> {
    if !sniff_wav(data) {
        bail!("not a RIFF WAVE file");
    }
    let rf64 = &data[..4] != b"RIFF";
    let mut fmt: Option<&[u8]> = None;
    let mut ds64_data: Option<u64> = None;
    let mut payload: Option<&[u8]> = None;
    let mut at = 12usize;
    while at + 8 <= data.len() {
        let id = &data[at..at + 4];
        let size32 = u32::from_le_bytes(data[at + 4..at + 8].try_into().unwrap());
        let body_at = at + 8;
        let size = match (id, ds64_data) {
            // RF64: the data size is the ds64 chunk's when the field says -1.
            (b"data", Some(big)) if rf64 && size32 == u32::MAX => big,
            _ => u64::from(size32),
        };
        let end = usize::try_from(size)
            .ok()
            .and_then(|s| body_at.checked_add(s))
            .unwrap_or(usize::MAX);
        match id {
            b"fmt " => fmt = data.get(body_at..end.min(data.len())),
            // ds64: RIFF size (8), data size (8), sample count (8), table.
            b"ds64" => {
                ds64_data = data
                    .get(body_at + 8..body_at + 16)
                    .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            }
            b"data" => {
                // A size never written (0) or one past the end: to the end.
                let end = if size == 0 || end > data.len() {
                    data.len()
                } else {
                    end
                };
                payload = Some(&data[body_at.min(data.len())..end]);
                break;
            }
            _ => {}
        }
        if end >= data.len() {
            break;
        }
        at = end + (end & 1);
    }
    let fmt = fmt.context("WAVE: no fmt chunk")?;
    let payload = payload.context("WAVE: no data chunk")?;
    if fmt.len() < 16 {
        bail!("WAVE: a fmt chunk of {} bytes", fmt.len());
    }
    let channels = u16::from_le_bytes([fmt[2], fmt[3]]);
    let sample_rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
    let tag = u16::from_le_bytes([fmt[0], fmt[1]]);
    let named = |name: String| AudioTrack {
        codec: name,
        samples: Vec::new(),
        sample_rate,
        channels,
        asc: Vec::new(),
        codec_private: Vec::new(),
        timescale: sample_rate.max(1),
        durations: Vec::new(),
    };
    let layout: PcmLayout = match crate::demux::audio::wave_format_pcm(fmt) {
        Some(Ok(layout)) => layout,
        Some(Err(_)) => {
            let real = if tag == 0xFFFE {
                u16::from_le_bytes([fmt[24], fmt[25]])
            } else {
                tag
            };
            return Ok(match crate::avi::wave_format_codec(real) {
                Some("mp3") => {
                    crate::ts::audio::mpeg_audio_from_es(payload)
                        .context("WAVE: no MPEG audio frame")?
                        .0
                }
                Some("ac3") => read_ac3(payload)?,
                Some("dts") => read_dts(payload)?,
                _ => {
                    let name = crate::avi::wave_format_name(real);
                    tracing::warn!(format_tag = format!("0x{real:04x}"), codec = %name, "WAVE: no reader for this format; surfaced by name");
                    named(name)
                }
            });
        }
        None => bail!("WAVE: a fmt chunk too short to read"),
    };
    if channels == 0 || sample_rate == 0 {
        bail!("WAVE: {channels} channels at {sample_rate} Hz");
    }
    let codec = layout.codec().context("WAVE: no PCM form")?;
    let frame = layout.bytes * usize::from(channels);
    let whole = payload.len() / frame * frame;
    let mut samples = Vec::new();
    let mut durations = Vec::new();
    for chunk in payload[..whole].chunks(PCM_PACKET_FRAMES * frame) {
        let mut p = chunk.to_vec();
        layout.normalise(&mut p);
        durations.push((chunk.len() / frame) as u32);
        samples.push(p);
    }
    if samples.is_empty() {
        bail!("WAVE: the data chunk holds no whole sample frame");
    }
    Ok(AudioTrack {
        codec: codec.into(),
        samples,
        sample_rate,
        channels,
        asc: Vec::new(),
        codec_private: Vec::new(),
        timescale: sample_rate,
        durations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adts(len: usize) -> Vec<u8> {
        let mut h = vec![0xFF, 0xF1, 0x4C, 0x80, 0, 0, 0xFC];
        h[3] |= ((len >> 11) & 3) as u8;
        h[4] = ((len >> 3) & 0xFF) as u8;
        h[5] = (((len & 7) << 5) | 0x1F) as u8;
        h.resize(len, 0x21);
        h
    }

    #[test]
    fn adts_needs_frames_that_chain() {
        let three = [adts(20), adts(30), adts(25)].concat();
        assert!(sniff_adts(&three));
        let mut broken = three.clone();
        broken[20] = 0;
        assert!(
            !sniff_adts(&broken),
            "the second header is not where the first ends"
        );
        assert!(!sniff_adts(&adts(20)), "one frame is not a stream");
        let mut tagged = b"ID3\x04\x00\x00\x00\x00\x00\x05hello".to_vec();
        tagged.extend_from_slice(&three);
        assert!(sniff_adts(&tagged), "behind an ID3v2 tag");
        // MPEG audio (layer bits != 00) is not ADTS.
        assert!(!sniff_adts(&[0xFF, 0xFB, 0x90, 0x44, 0, 0, 0, 0]));
    }

    #[test]
    fn wav_and_rf64_are_recognised() {
        assert!(sniff_wav(b"RIFF\0\0\0\0WAVEfmt "));
        assert!(sniff_wav(b"RF64\xFF\xFF\xFF\xFFWAVEds64"));
        assert!(!sniff_wav(b"RIFF\0\0\0\0AVI LIST"));
    }
}
