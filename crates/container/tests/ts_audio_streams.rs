//! Transport-stream audio rivet reads beyond AAC / MPEG audio / AC-3 /
//! E-AC-3: Opus (the Opus-in-TS mapping: stream_type 0x06, the "Opus"
//! registration, the `opus_audio_descriptor`, control headers on each
//! access unit), DTS (stream_type 0x82, or 0x06 with a "DTS1" registration),
//! audio-only transport streams and AVIs, and audio streams it has no reader
//! for surfaced by name. Every file is written here from ISO/IEC 13818-1,
//! the mappings and the RIFF/AVI layout, carrying packets rivet's own
//! encoders made.

use bytes::Bytes;
use codec::audio::{AudioCodec, AudioEncoderConfig, AudioFrame, create_decoder, create_encoder};
use container::streaming::demux_audio;

const PMT_PID: u16 = 0x100;
const VIDEO_PID: u16 = 0x200;
const AUDIO_PID: u16 = 0x101;

/// TS packets of one PES packet on `pid`, the last padded with adaptation
/// field stuffing (ISO/IEC 13818-1 §2.4.3.5), continuity counters from `cc`.
fn packetize(pid: u16, pes: &[u8], cc: &mut u8) -> Vec<u8> {
    let mut out = Vec::new();
    let mut at = 0;
    let mut first = true;
    while at < pes.len() {
        let take = (pes.len() - at).min(184);
        let mut p = vec![
            0x47,
            (if first { 0x40 } else { 0 }) | (pid >> 8) as u8,
            pid as u8,
        ];
        if take < 184 {
            // Adaptation field + payload: the field fills what the payload leaves.
            let af_len = 184 - take - 1;
            p.push(0x30 | (*cc & 0x0F));
            p.push(af_len as u8);
            if af_len > 0 {
                p.push(0x00);
                p.extend(std::iter::repeat_n(0xFF, af_len - 1));
            }
        } else {
            p.push(0x10 | (*cc & 0x0F));
        }
        p.extend_from_slice(&pes[at..at + take]);
        assert_eq!(p.len(), 188);
        out.extend(p);
        *cc = cc.wrapping_add(1);
        at += take;
        first = false;
    }
    out
}

/// A PES packet with a PTS (`stream_id`, 90 kHz `pts`).
fn pes(stream_id: u8, pts: u64, es: &[u8]) -> Vec<u8> {
    let mut p = vec![0, 0, 1, stream_id];
    let len = (3 + 5 + es.len()).min(0xFFFF) as u16;
    p.extend_from_slice(&if stream_id == 0xE0 { 0 } else { len }.to_be_bytes());
    p.extend_from_slice(&[0x80, 0x80, 5]);
    p.push(0x21 | (((pts >> 30) & 7) << 1) as u8);
    p.extend_from_slice(&((((pts >> 15) & 0x7FFF) << 1 | 1) as u16).to_be_bytes());
    p.extend_from_slice(&((((pts) & 0x7FFF) << 1 | 1) as u16).to_be_bytes());
    p.extend_from_slice(es);
    p
}

/// A PSI section in one packet (pointer_field 0). The CRC is not checked by
/// the reader and is left zero.
fn psi(pid: u16, section: &[u8], cc: &mut u8) -> Vec<u8> {
    let mut payload = vec![0u8];
    payload.extend_from_slice(section);
    payload.resize(184, 0xFF);
    let mut p = vec![
        0x47,
        0x40 | (pid >> 8) as u8,
        pid as u8,
        0x10 | (*cc & 0x0F),
    ];
    *cc = cc.wrapping_add(1);
    p.extend(payload);
    p
}

/// PAT (one program, PMT on `PMT_PID`) and a PMT naming `streams`
/// (stream_type, PID, descriptors).
fn tables(streams: &[(u8, u16, Vec<u8>)]) -> Vec<u8> {
    let mut cc = 0;
    let mut pat = vec![0x00, 0xB0, 13, 0x00, 0x01, 0xC1, 0x00, 0x00, 0x00, 0x01];
    pat.extend_from_slice(&[0xE0 | (PMT_PID >> 8) as u8, PMT_PID as u8, 0, 0, 0, 0]);
    let mut body = vec![
        0x00,
        0x01,
        0xC1,
        0x00,
        0x00,
        0xE0 | (AUDIO_PID >> 8) as u8,
        AUDIO_PID as u8,
        0xF0,
        0x00,
    ];
    for (st, pid, desc) in streams {
        body.extend_from_slice(&[
            *st,
            0xE0 | (pid >> 8) as u8,
            *pid as u8,
            0xF0 | (desc.len() >> 8) as u8,
            desc.len() as u8,
        ]);
        body.extend_from_slice(desc);
    }
    body.extend_from_slice(&[0; 4]);
    let mut pmt = vec![0x02, 0xB0 | (body.len() >> 8) as u8, body.len() as u8];
    pmt.extend(body);
    let mut out = psi(0, &pat, &mut cc);
    let mut cc2 = 0;
    out.extend(psi(PMT_PID, &pmt, &mut cc2));
    out
}

/// A transport stream: the tables, then one PES per audio access unit
/// (`aus`, each lasting `au_ticks` at 90 kHz) on `AUDIO_PID`, and — when
/// `video` — an MPEG-2 video PES ahead of them.
fn transport_stream(
    streams: &[(u8, u16, Vec<u8>)],
    audio_stream_id: u8,
    aus: &[Vec<u8>],
    au_ticks: u64,
    video: bool,
) -> Vec<u8> {
    let mut out = tables(streams);
    let mut vcc = 0;
    if video {
        // A sequence header and a picture start: enough for the video demuxer.
        let es = [
            0, 0, 1, 0xB3, 0x0A, 0x00, 0x78, 0x13, 0xFF, 0xFF, 0xE0, 0x18, 0, 0, 1, 0x00, 0x00,
            0x0F, 0xFF, 0xF8,
        ];
        out.extend(packetize(VIDEO_PID, &pes(0xE0, 9000, &es), &mut vcc));
    }
    let mut cc = 0;
    for (i, au) in aus.iter().enumerate() {
        out.extend(packetize(
            AUDIO_PID,
            &pes(audio_stream_id, 9000 + i as u64 * au_ticks, au),
            &mut cc,
        ));
    }
    out
}

/// `seconds` of a two-tone test signal, interleaved, `channels` wide.
fn signal(rate: u32, channels: u8, seconds: f32) -> Vec<f32> {
    let n = (rate as f32 * seconds) as usize;
    (0..n)
        .flat_map(|i| {
            let t = i as f32 / rate as f32;
            (0..channels).map(move |c| {
                0.3 * (2.0 * std::f32::consts::PI * (300.0 + 110.0 * c as f32) * t).sin()
            })
        })
        .collect()
}

/// Encode `pcm` with rivet's encoder: the packets and the codec's private data.
fn encode(codec: AudioCodec, rate: u32, channels: u8, pcm: &[f32]) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut enc =
        create_encoder(AudioEncoderConfig::new(codec, rate, channels, 0)).expect("encoder");
    let mut packets = enc
        .encode(&AudioFrame {
            samples: pcm.to_vec(),
            sample_rate: rate,
            channels,
            pts: 0,
        })
        .unwrap();
    packets.extend(enc.flush().unwrap());
    (
        packets.into_iter().map(|p| p.data).collect(),
        enc.extra_data(),
    )
}

/// Decode a demuxed track with rivet's decoder; the interleaved samples.
fn decode(track: &container::demux::AudioTrack) -> Vec<f32> {
    let extra = (!track.codec_private.is_empty()).then_some(track.codec_private.as_slice());
    let mut dec = create_decoder(&track.codec, extra, track.sample_rate, track.channels as u8)
        .expect("decoder");
    let mut out = Vec::new();
    for p in &track.samples {
        for f in dec.decode(p, 0).expect("decode") {
            out.extend(f.samples);
        }
    }
    out
}

/// An Opus access unit: the control header (prefix 0x3FF, the start-trim
/// flag when `start_trim` is given), the payload size in 0xFF-continued
/// bytes, the trim, then the packet.
fn opus_au(packet: &[u8], start_trim: Option<u16>) -> Vec<u8> {
    let mut au = vec![0x7F, 0xE0 | if start_trim.is_some() { 0x10 } else { 0 }];
    let mut size = packet.len();
    while size >= 255 {
        au.push(0xFF);
        size -= 255;
    }
    au.push(size as u8);
    if let Some(t) = start_trim {
        au.extend_from_slice(&(t & 0x1FFF).to_be_bytes());
    }
    au.extend_from_slice(packet);
    au
}

#[test]
fn opus_in_a_transport_stream_reads_beside_video_and_alone() {
    for (channels, code) in [(2u8, 0x02u8), (6, 0x06)] {
        let pcm = signal(48_000, channels, 0.5);
        let (packets, head) = encode(AudioCodec::Opus, 48_000, channels, &pcm);
        let pre_skip = u16::from_le_bytes([head[2], head[3]]);
        let aus: Vec<Vec<u8>> = packets
            .iter()
            .enumerate()
            .map(|(i, p)| opus_au(p, (i == 0).then_some(pre_skip)))
            .collect();
        // registration "Opus", then DVB's extension descriptor 0x7F with
        // the opus_audio_descriptor extension tag 0x80.
        let desc = [vec![0x05, 4], b"Opus".to_vec(), vec![0x7F, 2, 0x80, code]].concat();
        for video in [true, false] {
            let mut streams = vec![(0x06u8, AUDIO_PID, desc.clone())];
            if video {
                streams.insert(0, (0x02, VIDEO_PID, Vec::new()));
            }
            let ts = transport_stream(&streams, 0xBD, &aus, 1800, video);
            let name = format!("{channels}ch, video={video}");
            let src = demux_audio(Bytes::from(ts))
                .expect("demux")
                .unwrap_or_else(|| panic!("{name}: no audio"));
            assert_eq!(src.has_video, video, "{name}");
            let t = &src.track;
            assert_eq!(
                (t.codec.as_str(), t.channels, t.timescale),
                ("opus", u16::from(channels), 48_000),
                "{name}"
            );
            assert_eq!(
                t.samples, packets,
                "{name}: the control headers stripped, the packets verbatim"
            );
            assert_eq!(
                u16::from_le_bytes([t.codec_private[2], t.codec_private[3]]),
                pre_skip,
                "{name}: start trim as pre-skip"
            );
            assert_eq!(
                &t.codec_private[9..],
                &head[9..],
                "{name}: the mapping the descriptor names"
            );
            assert_eq!(
                decode(t).len(),
                packets.len() * 960 * usize::from(channels),
                "{name}"
            );
        }
    }
}

#[test]
fn dts_in_a_transport_stream_reads_under_either_signalling() {
    let pcm = signal(48_000, 2, 0.3);
    let (frames, _) = encode(AudioCodec::Dts, 48_000, 2, &pcm);
    // Two frames to a PES packet, as muxers do.
    let aus: Vec<Vec<u8>> = frames.chunks(2).map(|c| c.concat()).collect();
    for (stream_type, desc) in [
        (0x82u8, Vec::new()),
        (0x06, [vec![0x05, 4], b"DTS1".to_vec()].concat()),
    ] {
        let ts = transport_stream(&[(stream_type, AUDIO_PID, desc)], 0xBD, &aus, 1920, false);
        let src = demux_audio(Bytes::from(ts))
            .expect("demux")
            .expect("the DTS track");
        let t = &src.track;
        assert_eq!(
            (t.codec.as_str(), t.sample_rate, t.channels),
            ("dts", 48_000, 2),
            "stream_type {stream_type:#x}"
        );
        assert_eq!(t.samples, frames, "one sample per core frame");
        assert!(t.durations.iter().all(|&d| d == 512));
        assert_eq!(decode(t).len(), frames.len() * 512 * 2);
    }
}

#[test]
fn an_audio_only_transport_stream_is_read() {
    let pcm = signal(48_000, 2, 0.3);
    let (frames, _) = encode(AudioCodec::Ac3, 48_000, 2, &pcm);
    let ts = transport_stream(&[(0x81, AUDIO_PID, Vec::new())], 0xBD, &frames, 2880, false);
    let src = demux_audio(Bytes::from(ts.clone()))
        .expect("demux")
        .expect("the AC-3 track");
    assert!(!src.has_video);
    assert_eq!(src.track.codec, "ac3");
    assert_eq!(src.track.samples, frames);
    // The video demuxer still refuses it: it has no video.
    assert!(container::streaming::demux_streaming(&ts).is_err());
}

#[test]
fn a_transport_stream_audio_codec_without_a_reader_is_named() {
    // Dolby TrueHD (Blu-ray stream_type 0x83), beside a stream rivet reads
    // the one it reads is chosen; alone, it is named.
    let ts = transport_stream(
        &[(0x83, AUDIO_PID, Vec::new())],
        0xBD,
        &[vec![0xF8, 0x72, 0x6F, 0xBA, 0, 0]],
        1800,
        false,
    );
    let src = demux_audio(Bytes::from(ts))
        .expect("demux")
        .expect("a named track");
    assert_eq!(src.track.codec, "truehd");
    assert!(src.track.samples.is_empty());
}

/// A RIFF chunk, padded to an even length.
fn chunk(fourcc: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = fourcc.to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(body);
    if body.len() % 2 == 1 {
        out.push(0);
    }
    out
}

fn list(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    chunk(b"LIST", &[kind.as_slice(), body].concat())
}

#[test]
fn an_audio_only_avi_is_read() {
    // One `auds` stream of 16-bit stereo PCM (WAVEFORMATEX) and no video
    // stream: `avih`, one `strl` (`strh`, `strf`), and `movi` with the
    // stream's `00wb` chunks.
    let values: Vec<i16> = (0..4800)
        .map(|i| ((i * 37) % 30000 - 15000) as i16)
        .collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut avih = vec![0u8; 56];
    avih[24..28].copy_from_slice(&1u32.to_le_bytes()); // dwStreams
    let mut strh = vec![0u8; 56];
    strh[..4].copy_from_slice(b"auds");
    strh[20..24].copy_from_slice(&1u32.to_le_bytes()); // dwScale
    strh[24..28].copy_from_slice(&48_000u32.to_le_bytes()); // dwRate
    strh[44..48].copy_from_slice(&4u32.to_le_bytes()); // dwSampleSize
    let mut strf = Vec::new();
    strf.extend_from_slice(&1u16.to_le_bytes());
    strf.extend_from_slice(&2u16.to_le_bytes());
    strf.extend_from_slice(&48_000u32.to_le_bytes());
    strf.extend_from_slice(&192_000u32.to_le_bytes());
    strf.extend_from_slice(&4u16.to_le_bytes());
    strf.extend_from_slice(&16u16.to_le_bytes());
    strf.extend_from_slice(&0u16.to_le_bytes());
    let hdrl = list(
        b"hdrl",
        &[
            chunk(b"avih", &avih),
            list(
                b"strl",
                &[chunk(b"strh", &strh), chunk(b"strf", &strf)].concat(),
            ),
        ]
        .concat(),
    );
    let movi = list(
        b"movi",
        &bytes
            .chunks(4800)
            .map(|c| chunk(b"00wb", c))
            .collect::<Vec<_>>()
            .concat(),
    );
    let body = [b"AVI ".to_vec(), hdrl, movi].concat();
    let avi = chunk(b"RIFF", &body);
    let src = demux_audio(Bytes::from(avi))
        .expect("demux")
        .expect("the PCM track");
    assert!(!src.has_video);
    assert_eq!(
        (
            src.track.codec.as_str(),
            src.track.channels,
            src.track.sample_rate
        ),
        ("pcm_s16le", 2, 48_000)
    );
    assert_eq!(src.track.samples.concat(), bytes);
    let got = decode(&src.track);
    let want: Vec<f32> = values.iter().map(|&v| f32::from(v) / 32768.0).collect();
    assert_eq!(got, want);
}
