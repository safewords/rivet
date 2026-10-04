//! Blu-ray / BDAV `.m2ts` sources end to end.
//!
//! The defect this pins: a `.m2ts` — 192-byte source packets, each a 4-byte
//! `TP_extra_header` before the 188-byte transport packet — was refused as
//! "unsupported container: unknown" by the container sniffer, which looked
//! for the sync byte on the 188-byte grid only, although the transport
//! stream reader behind it strips the header. And its LPCM audio
//! (stream_type 0x80) was named and refused.
//!
//! Every source is made here: the pictures and the audio by this
//! workspace's own encoders, the transport stream by the test kit's own
//! writer (`synth::ts_av`), and the BDAV framing by [`bdav`], written from
//! the format's description (a 2-bit copy_permission_indicator and a 30-bit
//! arrival_time_stamp, in 27 MHz ticks, before each packet).

use std::sync::Arc;

use bytes::Bytes;
use container::{ContainerKind, sniff_container};

use super::lossless_tests::{decode, signal};
use crate::progress::NullSink;
use crate::settings::TranscodeSettings;
use crate::synth;
use crate::{JobOutput, RungArtifact};

/// A 188-byte transport stream rewrapped as BDAV source packets: before
/// each packet a `TP_extra_header`, copy_permission_indicator 0 and an
/// arrival_time_stamp that advances by `ats_step` 27 MHz ticks a packet
/// (0 leaves every header zero, as some tools write them).
fn bdav(ts: &[u8], ats_step: u32) -> Vec<u8> {
    assert_eq!(ts.len() % 188, 0);
    let mut out = Vec::with_capacity(ts.len() / 188 * 192);
    for (i, pkt) in ts.chunks_exact(188).enumerate() {
        let ats = (i as u32).wrapping_mul(ats_step) & 0x3FFF_FFFF;
        out.extend_from_slice(&ats.to_be_bytes()); // top two bits: copy_permission_indicator 0
        out.extend_from_slice(pkt);
    }
    out
}

/// A 188-byte transport stream with 16 bytes after each packet, where a
/// 204-byte stream keeps its Reed-Solomon parity (the reader passes over
/// them, so their value does not matter).
fn with_parity(ts: &[u8]) -> Vec<u8> {
    ts.chunks_exact(188).flat_map(|p| p.iter().copied().chain([0xA5; 16])).collect()
}

/// A second of H.264 pictures, 25 a second.
fn pictures(fps: u32) -> Vec<synth::Coded> {
    let pics = (0..fps).map(|t| synth::test_pattern(64, 48, u64::from(t), h26x::ChromaFormat::Yuv420));
    synth::encode_h264(&synth::H264::new(64, 48, fps), pics)
}

/// Half a second of a stereo tone, coded by rivet's own encoder for
/// `codec`, one access unit per packet.
fn coded_audio(codec: codec::audio::AudioCodec, channels: u8) -> Vec<(Vec<u8>, u32)> {
    use codec::audio::{AudioEncoderConfig, AudioFrame, create_encoder};
    let mut enc = create_encoder(AudioEncoderConfig::new(codec, 48_000, channels, 0)).unwrap();
    let frames = 24_000;
    let samples = (0..frames * usize::from(channels))
        .map(|i| 0.3 * ((i / usize::from(channels)) as f32 * 0.05 * (1 + i % usize::from(channels)) as f32).sin())
        .collect();
    let mut packets = enc.encode(&AudioFrame { samples, sample_rate: 48_000, channels, pts: 0 }).unwrap();
    packets.extend(enc.flush().unwrap());
    packets.into_iter().map(|p| (p.data, p.duration as u32)).collect()
}

/// Blu-ray LPCM frames for `pcm` (interleaved, WAVE order, `channels` of
/// them, `bits` 16 or 24), 240 sample frames (5 ms at 48 kHz) to a frame:
/// each the 4-byte LPCM header and the samples big-endian in the format's
/// own channel order, an odd count padded with an empty channel.
fn bd_lpcm(pcm: &[i32], channels: usize, bits: u32) -> Vec<(Vec<u8>, u32)> {
    // channel_assignment, stored channels, and for each stored channel the
    // WAVE-order channel it carries (None: the padding).
    let (assignment, stored): (u8, &[Option<usize>]) = match channels {
        1 => (1, &[Some(0), None]),
        2 => (3, &[Some(0), Some(1)]),
        // L R C Ls Rs LFE from WAVE's L R C LFE Ls Rs.
        6 => (9, &[Some(0), Some(1), Some(2), Some(4), Some(5), Some(3)]),
        // L R C Ls Lrs Rrs Rs LFE from WAVE's L R C LFE Lrs Rrs Ls Rs.
        8 => (11, &[Some(0), Some(1), Some(2), Some(6), Some(4), Some(5), Some(7), Some(3)]),
        _ => unreachable!(),
    };
    let width = (bits / 8) as usize;
    pcm.chunks(240 * channels)
        .map(|chunk| {
            let frames = chunk.len() / channels;
            let payload = frames * stored.len() * width;
            let bits_code = if bits == 16 { 1u8 } else { 3 };
            let mut f = vec![(payload >> 8) as u8, payload as u8, (assignment << 4) | 1, bits_code << 6];
            for frame in chunk.chunks_exact(channels) {
                for slot in stored {
                    let v = slot.map_or(0, |c| frame[c]);
                    f.extend_from_slice(&v.to_be_bytes()[4 - width..]);
                }
            }
            (f, frames as u32)
        })
        .collect()
}

fn run(input: &[u8], line: &str) -> anyhow::Result<JobOutput> {
    let probed = crate::probe::probe_bytes(input)?;
    let spec = TranscodeSettings::parse_kv_line(line)?.into_spec_for(&probed)?;
    crate::run_job_blocking(input, &spec, None, Arc::new(NullSink))
}

fn file(out: &JobOutput) -> &[u8] {
    match &out.rungs[0].artifact {
        RungArtifact::File(b) => b,
        other => panic!("expected a file, got {other:?}"),
    }
}

/// The 192-byte BDAV form of a transport stream — H.264 or HEVC, beside each
/// audio kind the Blu-ray stream types name that rivet reads — sniffs as a
/// transport stream and demuxes to exactly what the 188-byte stream does:
/// the same pictures, the same audio frames. So does the 204-byte form, and
/// a BDAV stream whose headers are all zero.
#[test]
fn a_bdav_stream_reads_as_its_transport_stream() {
    let h264 = pictures(25);
    let pics = (0..25u64).map(|t| synth::test_pattern(64, 48, t, h26x::ChromaFormat::Yuv420));
    let hevc = synth::encode_h265(64, 48, 25, pics);
    let ac3 = coded_audio(codec::audio::AudioCodec::Ac3, 2);
    let eac3 = coded_audio(codec::audio::AudioCodec::Eac3, 2);
    let dts: Vec<(Vec<u8>, u32)> = synth::dts_5_1(0.5).frames;
    for (video_type, video, video_codec, stream_type, units, codec) in [
        (0x1Bu8, &h264, "h264", 0x81u8, &ac3, "ac3"), // AC-3
        (0x24, &hevc, "h265", 0x84, &eac3, "eac3"),   // E-AC-3, the Blu-ray stream_type
        (0x1B, &h264, "h264", 0xA1, &eac3, "eac3"),   // E-AC-3, Blu-ray secondary audio
        (0x24, &hevc, "h265", 0x82, &dts, "dts"),     // DTS
        (0x1B, &h264, "h264", 0x86, &dts, "dts"),     // DTS-HD Master Audio: its core
    ] {
        let audio = synth::TsAudio { stream_type, stream_id: 0xBD, rate: 48_000, units };
        let ts = synth::ts_av(video, video_type, 25, Some(audio));
        let plain = container::demux::demux(&ts).unwrap();
        let plain_audio = plain.audio.as_ref().unwrap();
        assert_eq!(plain_audio.codec, codec, "stream_type {stream_type:#04x}");
        for (label, wrapped) in
            [("192", bdav(&ts, 1_000)), ("192, zero headers", bdav(&ts, 0)), ("204", with_parity(&ts))]
        {
            let what = format!("{codec} (stream_type {stream_type:#04x}) in {label}-byte packets");
            assert_eq!(sniff_container(&wrapped), ContainerKind::MpegTs, "{what}");
            let d = container::demux::demux(&wrapped).unwrap();
            assert_eq!((d.codec.as_str(), d.info.width, d.info.height), (video_codec, 64, 48), "{what}");
            assert_eq!(d.samples, plain.samples, "{what}: the same pictures");
            let a = d.audio.as_ref().unwrap();
            assert_eq!((&a.codec, &a.samples, &a.durations), (&plain_audio.codec, &plain_audio.samples, &plain_audio.durations), "{what}");
            let probed = crate::probe::probe_bytes(&wrapped).unwrap();
            assert_eq!(probed.container, "ts", "{what}");
        }
    }
}

/// An `.m2ts` with H.264 and AC-3 transcodes like any transport stream: the
/// pictures re-encoded, the AC-3 passed through.
#[test]
fn an_m2ts_transcodes() {
    let ac3 = coded_audio(codec::audio::AudioCodec::Ac3, 2);
    let audio = synth::TsAudio { stream_type: 0x81, stream_id: 0xBD, rate: 48_000, units: &ac3 };
    let m2ts = bdav(&synth::ts_av(&pictures(25), 0x1B, 25, Some(audio)), 1_000);
    let out = run(&m2ts, "codec=mpeg4").unwrap();
    assert_eq!(out.audio_handling, "ac3 passthrough");
    assert_eq!(out.rungs[0].frames, 25);
    let d = container::demux::demux(file(&out)).unwrap();
    assert_eq!(d.audio.unwrap().samples, ac3.into_iter().map(|(f, _)| f).collect::<Vec<_>>());
}

/// Blu-ray LPCM in an `.m2ts` — mono (stored with an empty second channel),
/// stereo and 5.1 at 16 bits, 7.1 at 24 — read into WAVE order and written
/// to FLAC, decodes back to the source PCM sample for sample.
#[test]
fn blu_ray_lpcm_is_read_sample_for_sample_in_wave_order() {
    let video = pictures(25);
    for (channels, bits) in [(1usize, 16u32), (2, 16), (6, 16), (8, 24)] {
        let pcm = signal(12_000, channels, bits);
        let units = bd_lpcm(&pcm, channels, bits);
        let audio = synth::TsAudio { stream_type: 0x80, stream_id: 0xBD, rate: 48_000, units: &units };
        let m2ts = bdav(&synth::ts_av(&video, 0x1B, 25, Some(audio)), 1_000);
        let probed = crate::probe::probe_bytes(&m2ts).unwrap();
        let a = probed.audio.as_ref().unwrap();
        let codec = if bits == 16 { "pcm_s16le" } else { "pcm_s24le" };
        assert_eq!((a.codec.as_str(), a.channels, a.sample_rate), (codec, channels as u16, 48_000));
        let out = run(&m2ts, "mode=audio audio=flac").unwrap();
        assert_eq!(decode(file(&out)), ("flac".into(), pcm, bits), "{channels} channels, {bits}-bit");
    }
}

/// Dolby TrueHD (stream_type 0x83), which rivet has no reader for, is
/// refused by name, never dropped for a video-only output.
#[test]
fn truehd_in_an_m2ts_is_refused_by_name() {
    let units = vec![(vec![0u8; 64], 40u32); 20];
    let audio = synth::TsAudio { stream_type: 0x83, stream_id: 0xBD, rate: 48_000, units: &units };
    let m2ts = bdav(&synth::ts_av(&pictures(25), 0x1B, 25, Some(audio)), 1_000);
    let d = container::demux::demux(&m2ts).unwrap();
    assert_eq!(d.audio.as_ref().map(|a| a.codec.as_str()), Some("truehd"));
    let e = format!("{:#}", run(&m2ts, "codec=mpeg4").unwrap_err());
    assert!(e.contains("truehd"), "{e}");
}

/// The audio reader (`demux_audio`, what `--mode audio` and probing read)
/// takes the BDAV framing as well.
#[test]
fn the_audio_reader_reads_lpcm_from_an_m2ts() {
    let pcm = signal(4_800, 2, 16);
    let units = bd_lpcm(&pcm, 2, 16);
    let audio = synth::TsAudio { stream_type: 0x80, stream_id: 0xBD, rate: 48_000, units: &units };
    let m2ts = bdav(&synth::ts_av(&pictures(25), 0x1B, 25, Some(audio)), 1_000);
    let track = container::streaming::demux_audio(Bytes::from(m2ts)).unwrap().unwrap().track;
    assert_eq!((track.codec.as_str(), track.channels), ("pcm_s16le", 2));
    let got: Vec<i32> = track
        .samples
        .iter()
        .flat_map(|s| s.chunks_exact(2).map(|b| i32::from(i16::from_le_bytes([b[0], b[1]]))))
        .collect();
    assert_eq!(got, pcm);
}
