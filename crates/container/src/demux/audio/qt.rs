//! The audio track's sample entry as QuickTime and ISO BMFF lay it out, and
//! the linear PCM it may describe.
//!
//! # Sound sample descriptions
//!
//! An audio `stsd` entry is a QuickTime *sound sample description* (Apple,
//! QuickTime File Format Specification, "Sound Sample Descriptions"); ISO/IEC
//! 14496-12 §12.2.3 `AudioSampleEntry` is its version 0. After the 8-byte
//! sample-entry header (six reserved bytes, the data reference index) come
//! `version`, `revision level`, `vendor`, the channel count, the sample size,
//! the compression ID, the packet size and a 16.16 sample rate — 28 bytes in
//! all — and then, by version:
//!
//! - **0**: nothing more; the codec's boxes follow.
//! - **1**: four 32-bit fields — samples per packet, bytes per packet, bytes
//!   per frame, bytes per sample — 44 bytes before the boxes.
//! - **2**: the fixed fields are placeholders (3, 16, -2, 0, 65536), followed
//!   by the size of the structure, a 64-bit float sample rate, the channel
//!   count, `0x7F000000`, the bits per channel, the LPCM format flags, the
//!   bytes per packet and the frames per packet — 64 bytes before the boxes.
//!
//! A version-1 or -2 entry is what QuickTime writers emit for anything but
//! the simplest audio, and its codec configuration often sits one level
//! down, in a `wave` (siDecompressionParam) atom: the ALAC magic cookie, the
//! `enda` byte-order flag of `in24` / `in32` / `fl32` / `fl64`, an `esds`.
//! Reading every entry as a 28-byte version 0 put the walk 16 or 36 bytes
//! short of its boxes, and an ALAC or PCM `.mov` was read as having no audio.
//!
//! ISO/IEC 14496-12 also has an `AudioSampleEntryV1` (entry version 1) that
//! keeps the 28-byte layout; an entry whose version-1 extension does not lead
//! to a box is read that way.
//!
//! # Linear PCM
//!
//! The uncompressed formats (QuickTime File Format, "Sound Sample
//! Descriptions" / "Audio sample description formats"; ISO/IEC 23003-5 for
//! `ipcm` / `fpcm`): `raw ` (8-bit offset binary), `twos` (signed,
//! big-endian), `sowt` (signed, little-endian), `in24` / `in32` (signed 24 /
//! 32 bits) and `fl32` / `fl64` (IEEE float) — big-endian unless an `enda`
//! atom says otherwise — `lpcm` (version 2, the format flags say it all),
//! and `ipcm` / `fpcm` with their `pcmC` box. Each is normalised here to the
//! little-endian form rivet's PCM decoder takes (`pcm_u8`, `pcm_s16le`,
//! `pcm_s24le`, `pcm_s32le`, `pcm_f32le`, `pcm_f64le`); the conversion is a
//! byte swap or a sign flip, so nothing is lost.
//!
//! PCM is read by chunk, not by sample: a QuickTime PCM track's samples are
//! single frames (often with a placeholder `stsz` size of 1), so each chunk
//! of the `stsc` / `stco` tables becomes one packet of `frames × bytes per
//! frame` bytes, and its duration is its frames' `stts` deltas.

use super::super::{direct_children, find_box_body, find_direct_child};
use crate::demux::AudioTrack;

/// One audio track's first sample entry.
#[derive(Debug, Clone)]
pub(crate) struct SoundEntry<'a> {
    pub(crate) fourcc: [u8; 4],
    pub(crate) channels: u16,
    /// Bits per sample from the fixed fields (version 2: the bits per channel).
    pub(crate) sample_size: u16,
    /// Hz; the 16.16 field's integer part, or version 2's float.
    pub(crate) sample_rate: f64,
    /// Version 1's bytes per frame; version 2's bytes per packet. 0 when unset.
    pub(crate) bytes_per_frame: u32,
    /// Version 2's LPCM format flags (`kAudioFormatFlag…`).
    pub(crate) lpcm_flags: u32,
    /// The boxes after the fixed fields.
    pub(crate) boxes: &'a [u8],
}

impl SoundEntry<'_> {
    /// The body of the child box `fourcc`, at the entry's level or inside its
    /// `wave` atom.
    pub(crate) fn config(&self, fourcc: &[u8; 4]) -> Option<&[u8]> {
        find_direct_child(self.boxes, fourcc).or_else(|| {
            let wave = find_direct_child(self.boxes, b"wave")?;
            find_direct_child(wave, fourcc)
        })
    }
}

/// Whether a `trak` is an audio track: a sound media header (ISO/IEC
/// 14496-12 §12.2.2), or a `soun` handler.
pub(crate) fn trak_is_audio(trak: &[u8]) -> bool {
    find_box_body(trak, &[b"mdia", b"minf", b"smhd"]).is_some()
        || find_box_body(trak, &[b"mdia", b"hdlr"]).is_some_and(|h| h.len() >= 12 && &h[8..12] == b"soun")
}

/// The first audio `trak` of the file, in file order.
pub(crate) fn first_audio_trak(data: &[u8]) -> Option<&[u8]> {
    let moov = find_direct_child(data, b"moov")?;
    direct_children(moov, b"trak").find(|t| trak_is_audio(t))
}

/// Every sample entry of a `stsd` body: (fourcc, the entry's body after its
/// 8-byte header).
pub(crate) fn stsd_entries(stsd: &[u8]) -> impl Iterator<Item = ([u8; 4], &[u8])> {
    let mut pos = 8usize; // version/flags, entry_count
    std::iter::from_fn(move || {
        let head = stsd.get(pos..pos + 8)?;
        let size = u32::from_be_bytes(head[..4].try_into().unwrap()) as usize;
        let fourcc: [u8; 4] = head[4..8].try_into().unwrap();
        if size < 8 || pos.checked_add(size)? > stsd.len() {
            return None;
        }
        let body = &stsd[pos + 8..pos + size];
        pos += size;
        Some((fourcc, body))
    })
}

/// Whether `bytes` starts with a plausible box header that fits.
fn opens_a_box(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && {
        let size = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
        size >= 8 && size <= bytes.len() && bytes[4..8].iter().all(|b| b.is_ascii_graphic() || *b == b' ')
    }
}

/// Read one sound sample description (`body` is the entry after its 8-byte
/// box header).
pub(crate) fn parse_sound_entry(fourcc: [u8; 4], body: &[u8]) -> Option<SoundEntry<'_>> {
    if body.len() < 28 {
        return None;
    }
    let u16_at = |at: usize| u16::from_be_bytes([body[at], body[at + 1]]);
    let u32_at = |at: usize| u32::from_be_bytes(body[at..at + 4].try_into().unwrap());
    let version = u16_at(8);
    let mut entry = SoundEntry {
        fourcc,
        channels: u16_at(16),
        sample_size: u16_at(18),
        sample_rate: f64::from(u32_at(24) >> 16),
        bytes_per_frame: 0,
        lpcm_flags: 0,
        boxes: &body[28..],
    };
    match version {
        1 if body.len() >= 44 && (opens_a_box(&body[44..]) || body.len() == 44 || !opens_a_box(&body[28..])) => {
            entry.bytes_per_frame = u32_at(36);
            entry.boxes = &body[44..];
        }
        2 if body.len() >= 64 => {
            entry.sample_rate = f64::from_bits(u64::from_be_bytes(body[32..40].try_into().unwrap()));
            entry.channels = u16::try_from(u32_at(40)).unwrap_or(0);
            entry.sample_size = u16::try_from(u32_at(48)).unwrap_or(0);
            entry.lpcm_flags = u32_at(52);
            entry.bytes_per_frame = u32_at(56);
            entry.boxes = &body[64..];
        }
        _ => {}
    }
    Some(entry)
}

/// The first audio track's first sample entry.
pub(crate) fn first_sound_entry(data: &[u8]) -> Option<SoundEntry<'_>> {
    let trak = first_audio_trak(data)?;
    sound_entry_of(trak)
}

/// A `trak`'s first sample entry, read as a sound description.
pub(crate) fn sound_entry_of(trak: &[u8]) -> Option<SoundEntry<'_>> {
    let stsd = find_box_body(trak, &[b"mdia", b"minf", b"stbl", b"stsd"])?;
    let (fourcc, body) = stsd_entries(stsd).next()?;
    parse_sound_entry(fourcc, body)
}

/// The config box `cfg` of the first audio sample entry named `entry`, in
/// any audio `trak`: at the entry's level or inside its `wave`.
pub(crate) fn audio_entry_config(data: &[u8], entry: &[u8; 4], cfg: &[u8; 4]) -> Option<Vec<u8>> {
    let moov = find_direct_child(data, b"moov")?;
    direct_children(moov, b"trak").find_map(|trak| {
        let stsd = find_box_body(trak, &[b"mdia", b"minf", b"stbl", b"stsd"])?;
        stsd_entries(stsd)
            .filter(|(fourcc, _)| fourcc == entry)
            .find_map(|(fourcc, body)| parse_sound_entry(fourcc, body)?.config(cfg).map(<[u8]>::to_vec))
    })
}

// ─── Linear PCM ──────────────────────────────────────────────────────────────

/// How the entry's samples are stored, when it is linear PCM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PcmLayout {
    /// Bytes per sample of one channel: 1, 2, 3, 4 or 8.
    pub(crate) bytes: usize,
    pub(crate) float: bool,
    pub(crate) big_endian: bool,
    /// Signed integers (two's complement); 8-bit `raw ` is offset binary.
    pub(crate) signed: bool,
}

impl PcmLayout {
    /// The codec name of the little-endian form rivet's PCM decoder takes.
    pub(crate) fn codec(self) -> Option<&'static str> {
        Some(match (self.float, self.bytes) {
            (false, 1) => "pcm_u8",
            (false, 2) => "pcm_s16le",
            (false, 3) => "pcm_s24le",
            (false, 4) => "pcm_s32le",
            (true, 4) => "pcm_f32le",
            (true, 8) => "pcm_f64le",
            _ => return None,
        })
    }

    /// Rewrite `bytes` (whole samples) in place into [`codec`](Self::codec)'s
    /// form: a byte swap for big-endian, a sign flip for signed 8-bit.
    pub(crate) fn normalise(self, bytes: &mut [u8]) {
        if self.big_endian && self.bytes > 1 {
            for s in bytes.chunks_exact_mut(self.bytes) {
                s.reverse();
            }
        }
        if self.bytes == 1 && self.signed {
            for b in bytes.iter_mut() {
                *b ^= 0x80;
            }
        }
    }
}

/// The PCM layout of a sound entry, `None` when it is not linear PCM (or is
/// a depth rivet's PCM decoder has no form for).
pub(crate) fn pcm_layout(entry: &SoundEntry<'_>) -> Option<PcmLayout> {
    // `enda`: a 16-bit flag, 1 = little-endian (QuickTime File Format,
    // "Sound sample description extensions").
    let little = entry.config(b"enda").is_some_and(|b| b.len() >= 2 && u16::from_be_bytes([b[0], b[1]]) & 1 == 1);
    let bits = |b: u16| usize::from(b).div_ceil(8);
    let layout = match &entry.fourcc {
        b"raw " => PcmLayout { bytes: bits(entry.sample_size.max(8)), float: false, big_endian: true, signed: entry.sample_size > 8 },
        b"twos" | b"NONE" => PcmLayout { bytes: bits(entry.sample_size.max(8)), float: false, big_endian: !little, signed: true },
        b"sowt" => PcmLayout { bytes: bits(entry.sample_size.max(8)), float: false, big_endian: false, signed: true },
        b"in24" => PcmLayout { bytes: 3, float: false, big_endian: !little, signed: true },
        b"in32" => PcmLayout { bytes: 4, float: false, big_endian: !little, signed: true },
        b"fl32" => PcmLayout { bytes: 4, float: true, big_endian: !little, signed: true },
        b"fl64" => PcmLayout { bytes: 8, float: true, big_endian: !little, signed: true },
        b"lpcm" => {
            // kAudioFormatFlagIsFloat 1, IsBigEndian 2, IsSignedInteger 4,
            // IsNonInterleaved 32 (CoreAudioBaseTypes, as the QuickTime spec
            // cites them for version 2 descriptions).
            let flags = entry.lpcm_flags;
            if flags & 32 != 0 {
                return None;
            }
            let float = flags & 1 != 0;
            let frame = entry.bytes_per_frame as usize;
            let channels = usize::from(entry.channels.max(1));
            let bytes = if frame > 0 && frame % channels == 0 { frame / channels } else { bits(entry.sample_size) };
            PcmLayout { bytes, float, big_endian: flags & 2 != 0, signed: float || flags & 4 != 0 }
        }
        b"ipcm" | b"fpcm" => {
            // ISO/IEC 23003-5 `pcmC`: FullBox, format_flags (bit 0 set =
            // little-endian), PCM_sample_size in bits.
            let pcmc = entry.config(b"pcmC")?;
            let (flags, size) = (*pcmc.get(4)?, *pcmc.get(5)?);
            PcmLayout { bytes: bits(u16::from(size)), float: &entry.fourcc == b"fpcm", big_endian: flags & 1 == 0, signed: true }
        }
        _ => return None,
    };
    layout.codec().map(|_| layout)
}

/// The `stbl` tables a chunk walk needs.
struct SampleTables<'a> {
    stsc: &'a [u8],
    offsets: Vec<u64>,
    stts: &'a [u8],
}

fn sample_tables(trak: &[u8]) -> Option<SampleTables<'_>> {
    let stbl = find_box_body(trak, &[b"mdia", b"minf", b"stbl"])?;
    let stsc = find_direct_child(stbl, b"stsc")?;
    let stts = find_direct_child(stbl, b"stts")?;
    let offsets = if let Some(stco) = find_direct_child(stbl, b"stco") {
        let n = u32::from_be_bytes(stco.get(4..8)?.try_into().ok()?) as usize;
        (0..n).map(|i| stco.get(8 + 4 * i..12 + 4 * i).map(|b| u64::from(u32::from_be_bytes(b.try_into().unwrap())))).collect::<Option<Vec<_>>>()?
    } else {
        let co64 = find_direct_child(stbl, b"co64")?;
        let n = u32::from_be_bytes(co64.get(4..8)?.try_into().ok()?) as usize;
        (0..n).map(|i| co64.get(8 + 8 * i..16 + 8 * i).map(|b| u64::from_be_bytes(b.try_into().unwrap()))).collect::<Option<Vec<_>>>()?
    };
    Some(SampleTables { stsc, offsets, stts })
}

/// The `mdhd` timescale of a `trak`.
pub(crate) fn media_timescale(trak: &[u8]) -> Option<u32> {
    let mdhd = find_box_body(trak, &[b"mdia", b"mdhd"])?;
    let at = if mdhd.first() == Some(&1) { 20 } else { 12 };
    Some(u32::from_be_bytes(mdhd.get(at..at + 4)?.try_into().ok()?)).filter(|&t| t > 0)
}

/// Each chunk's `(file offset, frames in it)`, from `stsc` / `stco`.
fn chunk_frames(t: &SampleTables<'_>) -> Option<Vec<(u64, u64)>> {
    let entries = u32::from_be_bytes(t.stsc.get(4..8)?.try_into().ok()?) as usize;
    let entry = |i: usize| -> Option<(u64, u64)> {
        let b = t.stsc.get(8 + 12 * i..20 + 12 * i)?;
        Some((u64::from(u32::from_be_bytes(b[0..4].try_into().unwrap())), u64::from(u32::from_be_bytes(b[4..8].try_into().unwrap()))))
    };
    let mut out = Vec::with_capacity(t.offsets.len());
    for i in 0..entries {
        let (first, per_chunk) = entry(i)?;
        let next_first = if i + 1 < entries { entry(i + 1)?.0 } else { t.offsets.len() as u64 + 1 };
        for chunk in first..next_first {
            let offset = *t.offsets.get(usize::try_from(chunk.checked_sub(1)?).ok()?)?;
            out.push((offset, per_chunk));
        }
    }
    Some(out)
}

/// A QuickTime / ISO BMFF linear PCM track, normalised to little-endian:
/// one packet per chunk. `None` when the first audio track is not PCM.
pub(crate) fn extract_mp4_pcm(data: &[u8]) -> Option<AudioTrack> {
    let trak = first_audio_trak(data)?;
    let entry = sound_entry_of(trak)?;
    let layout = pcm_layout(&entry)?;
    let codec = layout.codec()?;
    let channels = entry.channels;
    if channels == 0 {
        tracing::warn!(fourcc = %String::from_utf8_lossy(&entry.fourcc), "MP4 PCM: no channel count; not read");
        return None;
    }
    let frame_bytes = layout.bytes * usize::from(channels);
    let timescale = media_timescale(trak)?;
    let sample_rate = if entry.sample_rate >= 1.0 { entry.sample_rate.round() as u32 } else { timescale };
    let tables = sample_tables(trak)?;
    let chunks = chunk_frames(&tables)?;
    // stts as (count, delta) runs, consumed frame by frame.
    let runs = u32::from_be_bytes(tables.stts.get(4..8)?.try_into().ok()?) as usize;
    let mut stts: Vec<(u64, u64)> = (0..runs)
        .filter_map(|i| {
            let b = tables.stts.get(8 + 8 * i..16 + 8 * i)?;
            Some((u64::from(u32::from_be_bytes(b[0..4].try_into().unwrap())), u64::from(u32::from_be_bytes(b[4..8].try_into().unwrap()))))
        })
        .collect();
    stts.reverse();
    let mut samples = Vec::with_capacity(chunks.len());
    let mut durations = Vec::with_capacity(chunks.len());
    for (offset, frames) in chunks {
        let Some(bytes) = usize::try_from(frames).ok().and_then(|f| f.checked_mul(frame_bytes)) else { break };
        let start = usize::try_from(offset).ok()?;
        let Some(chunk) = start.checked_add(bytes).and_then(|end| data.get(start..end)) else {
            tracing::warn!(offset, bytes, "MP4 PCM: a chunk runs past the end of the file; truncating the track");
            break;
        };
        let mut packet = chunk.to_vec();
        layout.normalise(&mut packet);
        // The chunk's duration: its frames' deltas.
        let mut left = frames;
        let mut ticks = 0u64;
        while left > 0 {
            let Some(run) = stts.last_mut() else {
                // Past the table: one tick per frame at the media rate.
                ticks += left * u64::from(timescale) / u64::from(sample_rate.max(1));
                break;
            };
            let take = run.0.min(left);
            ticks += take * run.1;
            left -= take;
            run.0 -= take;
            if run.0 == 0 {
                stts.pop();
            }
        }
        if packet.is_empty() {
            continue;
        }
        samples.push(packet);
        durations.push(u32::try_from(ticks).unwrap_or(u32::MAX).max(1));
    }
    if samples.is_empty() {
        tracing::warn!(codec, "MP4 PCM: the track's sample tables locate no samples");
        return None;
    }
    Some(AudioTrack {
        codec: codec.into(),
        samples,
        sample_rate,
        channels,
        asc: Vec::new(),
        codec_private: Vec::new(),
        timescale,
        durations,
    })
}

/// The name rivet reports for an audio sample entry it has no path for, so
/// the job can refuse it by name rather than write the video alone.
pub(crate) fn unsupported_codec_name(fourcc: &[u8; 4]) -> String {
    match fourcc {
        // 3GPP TS 26.244 §6.5: AMR narrowband / wideband sample entries.
        b"samr" => "amr_nb".into(),
        b"sawb" => "amr_wb".into(),
        b"sqcp" => "qcelp".into(),
        b"sevc" => "evrc".into(),
        b"ulaw" => "pcm_mulaw".into(),
        b"alaw" => "pcm_alaw".into(),
        b"ima4" => "adpcm_ima_qt".into(),
        b"mac3" => "mace3".into(),
        b"mac6" => "mace6".into(),
        b"QDM2" => "qdm2".into(),
        b"Qclp" => "qcelp".into(),
        b"agsm" => "gsm".into(),
        b"mha1" | b"mhm1" => "mpegh_3d_audio".into(),
        b"ac-4" => "ac4".into(),
        b"dtse" => "dts_express".into(),
        b"mlpa" => "truehd".into(),
        b"enca" => "encrypted_audio".into(),
        other => {
            let name: String = other.iter().map(|&b| if b.is_ascii_alphanumeric() { char::from(b).to_ascii_lowercase() } else { '_' }).collect();
            format!("mp4_audio_{}", name.trim_end_matches('_'))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(fourcc: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = ((8 + body.len()) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(fourcc);
        out.extend_from_slice(body);
        out
    }

    /// A version-1 sound description body with the given children.
    fn v1_entry(channels: u16, bits: u16, rate: u32, children: &[u8]) -> Vec<u8> {
        let mut b = vec![0u8; 6];
        b.extend_from_slice(&1u16.to_be_bytes()); // data reference index
        b.extend_from_slice(&1u16.to_be_bytes()); // version
        b.extend_from_slice(&[0; 6]); // revision, vendor
        b.extend_from_slice(&channels.to_be_bytes());
        b.extend_from_slice(&bits.to_be_bytes());
        b.extend_from_slice(&(-2i16).to_be_bytes()); // compression id
        b.extend_from_slice(&0u16.to_be_bytes());
        b.extend_from_slice(&(rate << 16).to_be_bytes());
        let bytes = u32::from(bits / 8);
        for v in [1, bytes, bytes * u32::from(channels), bytes] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        b.extend_from_slice(children);
        b
    }

    #[test]
    fn a_version_1_entry_reaches_its_wave_children() {
        let mut wave = boxed(b"frma", b"in24");
        wave.extend(boxed(b"enda", &[0, 1]));
        let body = v1_entry(2, 24, 48_000, &boxed(b"wave", &wave));
        let e = parse_sound_entry(*b"in24", &body).unwrap();
        assert_eq!((e.channels, e.sample_rate as u32, e.bytes_per_frame), (2, 48_000, 6));
        assert_eq!(e.config(b"enda"), Some(&[0u8, 1][..]));
        let l = pcm_layout(&e).unwrap();
        assert_eq!((l.codec(), l.big_endian), (Some("pcm_s24le"), false));
    }

    #[test]
    fn big_endian_and_signed_8_bit_pcm_normalise_to_the_decoders_forms() {
        let twos16 = PcmLayout { bytes: 2, float: false, big_endian: true, signed: true };
        let mut s = vec![0x12, 0x34, 0xFF, 0xFE];
        twos16.normalise(&mut s);
        assert_eq!(s, [0x34, 0x12, 0xFE, 0xFF]);
        let twos8 = PcmLayout { bytes: 1, float: false, big_endian: true, signed: true };
        let mut s = vec![0x00, 0x80, 0x7F];
        twos8.normalise(&mut s);
        assert_eq!(s, [0x80, 0x00, 0xFF], "signed 0 is unsigned 128");
        assert_eq!(twos8.codec(), Some("pcm_u8"));
    }

    #[test]
    fn unknown_entries_are_named() {
        assert_eq!(unsupported_codec_name(b"samr"), "amr_nb");
        assert_eq!(unsupported_codec_name(b"xyz "), "mp4_audio_xyz");
    }
}
