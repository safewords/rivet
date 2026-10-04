//! PAT and PMT section parsing.
//!
//! Surfaces per-program and per-stream metadata from the TS program-specific
//! information tables. Used by both the legacy (`demux_ts`) and streaming
//! (`demux_ts_streaming_init`) entry points.

use super::{
    AudioCodecKind, AudioStreamInfo, DESC_TAG_REGISTRATION, PatProgram, REG_AC3, REG_DTS1,
    REG_DTS2, REG_DTS3, REG_EAC3, REG_OPUS, STREAM_TYPE_AAC_ADTS, STREAM_TYPE_AAC_LATM,
    STREAM_TYPE_AC3, STREAM_TYPE_BD_EAC3, STREAM_TYPE_BD_LPCM, STREAM_TYPE_BD_SECONDARY_DTS,
    STREAM_TYPE_BD_SECONDARY_EAC3, STREAM_TYPE_DTS, STREAM_TYPE_DTS_HD_HR, STREAM_TYPE_DTS_HD_MA,
    STREAM_TYPE_EAC3, STREAM_TYPE_H264, STREAM_TYPE_HEVC, STREAM_TYPE_MPEG1_AUDIO,
    STREAM_TYPE_MPEG1_VIDEO, STREAM_TYPE_MPEG2_AUDIO, STREAM_TYPE_MPEG2_VIDEO,
    STREAM_TYPE_MPEG4_AUDIO_RAW, STREAM_TYPE_PES_PRIVATE, STREAM_TYPE_TRUEHD, VideoStreamInfo,
};

/// Walk the PAT section and return every `(program_number, pmt_pid)`
/// pair (skipping `program_number == 0`, which carries the network_PID
/// per ISO/IEC 13818-1 §2.4.4.3 — not a real program).
pub(super) fn parse_pat_all_programs(section: &[u8]) -> Vec<PatProgram> {
    // PAT section:
    //   table_id(8)=0, section_syntax_indicator(1), '0'(1), reserved(2),
    //   section_length(12), transport_stream_id(16), reserved(2),
    //   version(5), current_next(1), section_number(8),
    //   last_section_number(8), then N × (program_number u16, reserved(3),
    //   PID(13)), followed by CRC_32.
    let mut out = Vec::new();
    if section.len() < 12 {
        return out;
    }
    if section[0] != 0x00 {
        return out;
    }
    let section_length = (((section[1] & 0x0F) as usize) << 8) | section[2] as usize;
    let total = 3 + section_length;
    if total > section.len() {
        return out;
    }
    let loop_start = 8;
    let loop_end = total - 4;
    let mut i = loop_start;
    while i + 4 <= loop_end {
        let program = u16::from_be_bytes([section[i], section[i + 1]]);
        let pid = (((section[i + 2] & 0x1F) as u16) << 8) | section[i + 3] as u16;
        if program != 0 {
            out.push(PatProgram {
                program_number: program,
                pmt_pid: pid,
            });
        }
        i += 4;
    }
    out
}

/// Back-compat shim — returns the first program's PMT PID. Kept so the
/// streaming-init path stays a one-line change; the multi-program walk
/// uses `parse_pat_all_programs` directly.
pub(super) fn parse_pat_first_pmt_pid(section: &[u8]) -> Option<u16> {
    parse_pat_all_programs(section).first().map(|p| p.pmt_pid)
}

/// The PMT section's `PCR_PID` (ISO/IEC 13818-1 §2.4.4.9): the PID whose
/// adaptation fields carry the program clock reference. `None` for a section
/// too short to hold it or the value 0x1FFF (no PCR, a private stream).
pub(super) fn parse_pmt_pcr_pid(section: &[u8]) -> Option<u16> {
    if section.len() < 12 || section[0] != 0x02 {
        return None;
    }
    let pid = (((section[8] & 0x1F) as u16) << 8) | section[9] as u16;
    (pid != 0x1FFF).then_some(pid)
}

/// Walk the PMT section once and return every recognised video stream
/// (PID + stream_type) plus every recognised audio stream (PID +
/// stream_type + codec kind). Audio is optional — TS files without an
/// audio track demux video-only, matching MP4/MKV behaviour at the
/// demuxer layer.
///
/// AC-3 / E-AC-3 detection (Squad-37): we honour the ATSC A/53
/// stream_types (0x81 / 0x87) directly AND the DVB form where the
/// stream_type is 0x06 (PES private) and the ES descriptor loop carries
/// a `registration_descriptor` (tag 0x05) with the 4-char identifier
/// "AC-3" or "EAC3".
pub(super) fn parse_pmt_streams(
    section: &[u8],
) -> Option<(Vec<VideoStreamInfo>, Vec<AudioStreamInfo>)> {
    // PMT section:
    //   table_id(8)=0x02, section_syntax_indicator(1), '0'(1), reserved(2),
    //   section_length(12), program_number(16), reserved(2), version(5),
    //   current_next(1), section_number(8), last_section_number(8),
    //   reserved(3), PCR_PID(13), reserved(4), program_info_length(12),
    //   program_info_descriptors...,
    //   then N × (stream_type(8), reserved(3), elementary_PID(13),
    //     reserved(4), ES_info_length(12), ES_info_descriptors...),
    //   CRC_32.
    if section.len() < 12 {
        return None;
    }
    if section[0] != 0x02 {
        return None;
    }
    let section_length = (((section[1] & 0x0F) as usize) << 8) | section[2] as usize;
    let total = 3 + section_length;
    if total > section.len() {
        return None;
    }
    if section.len() < 12 {
        return None;
    }
    let pil = (((section[10] & 0x0F) as usize) << 8) | section[11] as usize;
    let mut i = 12 + pil;
    let loop_end = total - 4; // strip CRC
    let mut video: Vec<VideoStreamInfo> = Vec::new();
    let mut audio: Vec<AudioStreamInfo> = Vec::new();
    while i + 5 <= loop_end {
        let stype = section[i];
        let pid = (((section[i + 1] & 0x1F) as u16) << 8) | section[i + 2] as u16;
        let esi_len = (((section[i + 3] & 0x0F) as usize) << 8) | section[i + 4] as usize;
        let desc_start = i + 5;
        let desc_end = (desc_start + esi_len).min(loop_end);
        let descriptors = if desc_start <= desc_end {
            &section[desc_start..desc_end]
        } else {
            &[][..]
        };

        match stype {
            STREAM_TYPE_MPEG1_VIDEO
            | STREAM_TYPE_MPEG2_VIDEO
            | STREAM_TYPE_H264
            | STREAM_TYPE_HEVC => {
                video.push(VideoStreamInfo {
                    pid,
                    stream_type: stype,
                });
            }
            STREAM_TYPE_AAC_ADTS => {
                audio.push(AudioStreamInfo {
                    pid,
                    stream_type: stype,
                    kind: AudioCodecKind::AacAdts,
                });
            }
            STREAM_TYPE_MPEG1_AUDIO | STREAM_TYPE_MPEG2_AUDIO => {
                audio.push(AudioStreamInfo {
                    pid,
                    stream_type: stype,
                    kind: AudioCodecKind::MpegAudio,
                });
            }
            STREAM_TYPE_AC3 => {
                audio.push(AudioStreamInfo {
                    pid,
                    stream_type: stype,
                    kind: AudioCodecKind::Ac3,
                });
            }
            STREAM_TYPE_EAC3 | STREAM_TYPE_BD_EAC3 | STREAM_TYPE_BD_SECONDARY_EAC3 => {
                audio.push(AudioStreamInfo {
                    pid,
                    stream_type: stype,
                    kind: AudioCodecKind::Eac3,
                });
            }
            STREAM_TYPE_DTS | STREAM_TYPE_DTS_HD_HR | STREAM_TYPE_DTS_HD_MA => {
                audio.push(AudioStreamInfo {
                    pid,
                    stream_type: stype,
                    kind: AudioCodecKind::Dts,
                });
            }
            STREAM_TYPE_BD_LPCM => {
                audio.push(AudioStreamInfo {
                    pid,
                    stream_type: stype,
                    kind: AudioCodecKind::BdLpcm,
                });
            }
            STREAM_TYPE_TRUEHD
            | STREAM_TYPE_BD_SECONDARY_DTS
            | STREAM_TYPE_AAC_LATM
            | STREAM_TYPE_MPEG4_AUDIO_RAW => {
                let name = match stype {
                    STREAM_TYPE_TRUEHD => "truehd",
                    STREAM_TYPE_BD_SECONDARY_DTS => "dts_express",
                    STREAM_TYPE_AAC_LATM => "aac_latm",
                    _ => "mpeg4_audio_raw",
                };
                audio.push(AudioStreamInfo {
                    pid,
                    stream_type: stype,
                    kind: AudioCodecKind::Unsupported(name),
                });
            }
            STREAM_TYPE_PES_PRIVATE => {
                // DVB (ETSI TS 101 154) carries AC-3, E-AC-3, DTS and Opus
                // here: the codec is named by a registration_descriptor
                // ("AC-3", "EAC3", "DTS1".."DTS3", "Opus") or by a DVB
                // descriptor (ETSI EN 300 468 Annex D / G). A PES-private
                // stream with neither (subtitles, teletext, data) is not audio.
                if let Some(kind) = private_audio_kind(descriptors) {
                    audio.push(AudioStreamInfo {
                        pid,
                        stream_type: stype,
                        kind,
                    });
                }
            }
            _ => {}
        }
        i += 5 + esi_len;
    }
    Some((video, audio))
}

/// The audio codec a PES-private (stream_type 0x06) stream's descriptors
/// name, if any.
fn private_audio_kind(descriptors: &[u8]) -> Option<AudioCodecKind> {
    match find_registration(descriptors) {
        Some(REG_AC3) => return Some(AudioCodecKind::Ac3),
        Some(REG_EAC3) => return Some(AudioCodecKind::Eac3),
        Some(REG_DTS1 | REG_DTS2 | REG_DTS3) => return Some(AudioCodecKind::Dts),
        Some(REG_OPUS) => {
            // The opus_audio_descriptor: DVB's extension descriptor (tag
            // 0x7F) with descriptor_tag_extension 0x80, then
            // channel_config_code.
            let code = descriptor_bodies(descriptors)
                .find(|&(tag, body)| tag == 0x7F && body.first() == Some(&0x80))
                .and_then(|(_, body)| body.get(1).copied());
            return Some(match code {
                Some(c) => AudioCodecKind::Opus {
                    channel_config_code: c,
                },
                // No descriptor: stereo, the default of the mapping's most
                // common configuration, read with the decoder's checks.
                None => AudioCodecKind::Opus {
                    channel_config_code: 0x02,
                },
            });
        }
        _ => {}
    }
    // ETSI EN 300 468 descriptors: AC-3 (0x6A), enhanced AC-3 (0x7A), DTS
    // (0x7B), AAC (0x7C), and the extension descriptor (0x7F) for DTS-HD
    // (0x0E), DTS Neural (0x0F) and AC-4 (0x15).
    descriptor_bodies(descriptors).find_map(|(tag, body)| match (tag, body.first()) {
        (0x6A, _) => Some(AudioCodecKind::Ac3),
        (0x7A, _) => Some(AudioCodecKind::Eac3),
        (0x7B, _) => Some(AudioCodecKind::Dts),
        (0x7C, _) => Some(AudioCodecKind::Unsupported("aac_dvb_private")),
        (0x7F, Some(0x0E | 0x0F)) => Some(AudioCodecKind::Dts),
        (0x7F, Some(0x15)) => Some(AudioCodecKind::Unsupported("ac4")),
        _ => None,
    })
}

/// Every `(descriptor_tag, body)` of a descriptor loop.
fn descriptor_bodies(descriptors: &[u8]) -> impl Iterator<Item = (u8, &[u8])> {
    let mut i = 0usize;
    std::iter::from_fn(move || {
        let tag = *descriptors.get(i)?;
        let len = usize::from(*descriptors.get(i + 1)?);
        let body = descriptors.get(i + 2..i + 2 + len)?;
        i += 2 + len;
        Some((tag, body))
    })
}

/// Walk the ES descriptor loop and look for a registration_descriptor
/// (tag 0x05) carrying a 4-byte format_identifier; return the BE u32 of
/// that identifier or None.
fn find_registration(descriptors: &[u8]) -> Option<u32> {
    let mut i = 0usize;
    while i + 2 <= descriptors.len() {
        let tag = descriptors[i];
        let len = descriptors[i + 1] as usize;
        let body_start = i + 2;
        let body_end = body_start + len;
        if body_end > descriptors.len() {
            break;
        }
        if tag == DESC_TAG_REGISTRATION && len >= 4 {
            let id = u32::from_be_bytes([
                descriptors[body_start],
                descriptors[body_start + 1],
                descriptors[body_start + 2],
                descriptors[body_start + 3],
            ]);
            return Some(id);
        }
        i = body_end;
    }
    None
}

// (Squad-13's `parse_pmt_video_and_audio` shim was retired in Squad-37
// — both the legacy `demux_ts` path and the streaming demuxer now use
// `parse_pmt_streams` directly so the AC-3 / E-AC-3 / multi-program
// surface stays a single-walker invariant.)
