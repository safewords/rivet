//! Audio files with no container of their own: RIFF WAVE (PCM, float,
//! WAVE_FORMAT_EXTENSIBLE, RF64 with its `ds64`, a `data` size never
//! written), and bare ADTS AAC, AC-3, E-AC-3 and DTS streams. Each file is
//! written here (the WAVE layout from the RIFF specification and EBU Tech
//! 3306, ADTS headers from ISO/IEC 13818-7 §6.2) around samples or packets
//! rivet's own encoders made, then sniffed, read by `demux_audio` and
//! decoded.

use bytes::Bytes;
use codec::audio::{AudioCodec, AudioEncoderConfig, AudioFrame, create_decoder, create_encoder};
use container::sniff::{ContainerKind, sniff_container};
use container::streaming::demux_audio;

fn decode(track: &container::demux::AudioTrack) -> Vec<f32> {
    let private = if track.codec == "aac" { &track.asc } else { &track.codec_private };
    let extra = (!private.is_empty()).then_some(private.as_slice());
    let mut dec = create_decoder(&track.codec, extra, track.sample_rate, track.channels as u8).expect("decoder");
    let mut out = Vec::new();
    for p in &track.samples {
        for f in dec.decode(p, 0).expect("decode") {
            out.extend(f.samples);
        }
    }
    for f in dec.flush().expect("flush") {
        out.extend(f.samples);
    }
    out
}

fn signal(rate: u32, channels: u8, seconds: f32) -> Vec<f32> {
    let n = (rate as f32 * seconds) as usize;
    (0..n)
        .flat_map(|i| {
            let t = i as f32 / rate as f32;
            (0..channels).map(move |c| 0.3 * (2.0 * std::f32::consts::PI * (330.0 + 97.0 * c as f32) * t).sin())
        })
        .collect()
}

fn encode(codec: AudioCodec, rate: u32, channels: u8, pcm: &[f32]) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut enc = create_encoder(AudioEncoderConfig::new(codec, rate, channels, 0)).expect("encoder");
    let mut packets = enc.encode(&AudioFrame { samples: pcm.to_vec(), sample_rate: rate, channels, pts: 0 }).unwrap();
    packets.extend(enc.flush().unwrap());
    (packets.into_iter().map(|p| p.data).collect(), enc.extra_data())
}

fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(body);
    if body.len() % 2 == 1 {
        out.push(0);
    }
    out
}

/// A WAVEFORMATEX (`extensible`: WAVE_FORMAT_EXTENSIBLE with the sub-format
/// GUID's first two bytes `tag`).
fn fmt(tag: u16, channels: u16, rate: u32, bits: u16, extensible: bool) -> Vec<u8> {
    let block = channels * bits / 8;
    let mut f = Vec::new();
    f.extend_from_slice(&(if extensible { 0xFFFEu16 } else { tag }).to_le_bytes());
    f.extend_from_slice(&channels.to_le_bytes());
    f.extend_from_slice(&rate.to_le_bytes());
    f.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
    f.extend_from_slice(&block.to_le_bytes());
    f.extend_from_slice(&bits.to_le_bytes());
    if extensible {
        f.extend_from_slice(&22u16.to_le_bytes());
        f.extend_from_slice(&bits.to_le_bytes()); // wValidBitsPerSample
        f.extend_from_slice(&0x3Fu32.to_le_bytes()); // FL FR FC LFE BL BR
        f.extend_from_slice(&tag.to_le_bytes());
        // The rest of KSDATAFORMAT_SUBTYPE_PCM / _IEEE_FLOAT's GUID.
        f.extend_from_slice(&[0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71]);
    }
    f
}

fn wav(fmt: &[u8], data: &[u8]) -> Vec<u8> {
    let body = [b"WAVE".to_vec(), chunk(b"fmt ", fmt), chunk(b"data", data)].concat();
    chunk(b"RIFF", &body)
}

#[test]
fn wave_files_read_and_decode_exactly() {
    let ints: Vec<i32> = (0..9000 * 6).map(|i| ((i * 7919) % 16_000_000) - 8_000_000).collect();
    // (name, file, codec, channels, expected samples)
    let s16: Vec<u8> = ints[..9000 * 2].iter().flat_map(|&v| ((v >> 8) as i16).to_le_bytes()).collect();
    let s16_want: Vec<f32> = ints[..9000 * 2].iter().map(|&v| f32::from((v >> 8) as i16) / 32768.0).collect();
    let s24: Vec<u8> = ints.iter().flat_map(|&v| v.to_le_bytes()[..3].to_vec()).collect();
    let s24_want: Vec<f32> = ints.iter().map(|&v| v as f32 / 8_388_608.0).collect();
    let f32_want: Vec<f32> = ints[..9000 * 2].iter().map(|&v| v as f32 / 8_388_608.0).collect();
    let f32b: Vec<u8> = f32_want.iter().flat_map(|v| v.to_le_bytes()).collect();
    // RF64: the RIFF and data sizes -1, the real ones in `ds64`.
    let rf64 = {
        let mut ds64 = Vec::new();
        ds64.extend_from_slice(&0u64.to_le_bytes());
        ds64.extend_from_slice(&(s16.len() as u64).to_le_bytes());
        ds64.extend_from_slice(&9000u64.to_le_bytes());
        ds64.extend_from_slice(&0u32.to_le_bytes());
        let mut data = b"data".to_vec();
        data.extend_from_slice(&u32::MAX.to_le_bytes());
        data.extend_from_slice(&s16);
        let body = [b"WAVE".to_vec(), chunk(b"ds64", &ds64), chunk(b"fmt ", &fmt(1, 2, 48_000, 16, false)), data].concat();
        [b"RF64".to_vec(), u32::MAX.to_le_bytes().to_vec(), body].concat()
    };
    // A recorder that never came back for the sizes: data size 0.
    let no_size = {
        let mut w = wav(&fmt(1, 2, 48_000, 16, false), &s16);
        let at = w.windows(4).position(|x| x == b"data").unwrap();
        w[at + 4..at + 8].copy_from_slice(&0u32.to_le_bytes());
        w
    };
    type Case<'a> = (&'a str, Vec<u8>, &'a str, u16, &'a Vec<f32>);
    let cases: Vec<Case> = vec![
        ("pcm 16", wav(&fmt(1, 2, 48_000, 16, false), &s16), "pcm_s16le", 2, &s16_want),
        ("extensible 24-bit 5.1", wav(&fmt(1, 6, 48_000, 24, true), &s24), "pcm_s24le", 6, &s24_want),
        ("float 32", wav(&fmt(3, 2, 48_000, 32, false), &f32b), "pcm_f32le", 2, &f32_want),
        ("rf64", rf64, "pcm_s16le", 2, &s16_want),
        ("data size unwritten", no_size, "pcm_s16le", 2, &s16_want),
    ];
    for (name, file, codec, channels, want) in cases {
        assert_eq!(sniff_container(&file), ContainerKind::Wav, "{name}");
        let src = demux_audio(Bytes::from(file)).expect("demux").unwrap_or_else(|| panic!("{name}: no audio"));
        assert!(!src.has_video);
        let t = &src.track;
        assert_eq!((t.codec.as_str(), t.channels, t.sample_rate), (codec, channels, 48_000), "{name}");
        assert_eq!(t.durations.iter().map(|&d| d as usize).sum::<usize>(), want.len() / usize::from(channels), "{name}");
        assert_eq!(&decode(t), want, "{name}");
    }
    // A format with no reader is named, not hidden.
    let adpcm = wav(&fmt(2, 2, 48_000, 4, false), &[0; 512]);
    let src = demux_audio(Bytes::from(adpcm)).expect("demux").expect("named");
    assert_eq!(src.track.codec, "adpcm_ms");
    assert!(src.track.samples.is_empty());
}

/// An ADTS header (no CRC) for an AAC-LC access unit of `payload` bytes.
fn adts_header(sr_index: u8, channels: u8, payload: usize) -> [u8; 7] {
    let len = payload + 7;
    [
        0xFF,
        0xF1, // MPEG-4, layer 0, protection absent
        (1 << 6) | (sr_index << 2) | (channels >> 2), // profile LC (object type 2 - 1)
        ((channels & 3) << 6) | ((len >> 11) & 3) as u8,
        ((len >> 3) & 0xFF) as u8,
        (((len & 7) << 5) as u8) | 0x1F,
        0xFC,
    ]
}

#[test]
fn a_bare_adts_stream_reads_behind_an_id3_tag() {
    let pcm = signal(48_000, 2, 0.5);
    let (aus, asc) = encode(AudioCodec::Aac, 48_000, 2, &pcm);
    let mut file = b"ID3\x04\x00\x00\x00\x00\x00\x04TAG!".to_vec();
    for au in &aus {
        file.extend_from_slice(&adts_header(3, 2, au.len()));
        file.extend_from_slice(au);
    }
    assert_eq!(sniff_container(&file), ContainerKind::Adts);
    let src = demux_audio(Bytes::from(file)).expect("demux").expect("the AAC track");
    let t = &src.track;
    assert_eq!((t.codec.as_str(), t.sample_rate, t.channels), ("aac", 48_000, 2));
    assert_eq!(t.samples, aus, "the ADTS headers stripped");
    assert_eq!(t.asc[..2], asc[..2], "the AudioSpecificConfig the headers state");
    assert_eq!(decode(t).len(), aus.len() * 1024 * 2);
}

#[test]
fn bare_ac3_eac3_and_dts_streams_read() {
    let pcm = signal(48_000, 2, 0.4);
    for (codec, name) in [(AudioCodec::Ac3, "ac3"), (AudioCodec::Eac3, "eac3"), (AudioCodec::Dts, "dts")] {
        let (frames, _) = encode(codec, 48_000, 2, &pcm);
        let file = frames.concat();
        let kind = if name == "dts" { ContainerKind::DtsEs } else { ContainerKind::Ac3Es };
        assert_eq!(sniff_container(&file), kind, "{name}");
        let src = demux_audio(Bytes::from(file)).expect("demux").expect("the track");
        let t = &src.track;
        assert_eq!((t.codec.as_str(), t.sample_rate, t.channels), (name, 48_000, 2));
        assert_eq!(t.samples, frames, "{name}: one sample per frame");
        let total: usize = t.durations.iter().map(|&d| d as usize).sum();
        assert_eq!(decode(t).len(), total * 2, "{name}: each frame lasts what it decodes to");
    }
}
