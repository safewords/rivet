//! Matroska linear PCM (`A_PCM/INT/LIT`, `A_PCM/INT/BIG`,
//! `A_PCM/FLOAT/IEEE`, and `A_MS/ACM` with a PCM WAVEFORMATEX), on files
//! written here from the Matroska specification (RFC 9559 elements, the
//! codec mappings' PCM rules), read through `demux_audio` and decoded.
//! An audio codec rivet has no path for is named, not hidden.

use bytes::Bytes;
use container::streaming::demux_audio;

fn size_vint_8(size: u64) -> [u8; 8] {
    ((1u64 << 56) | size).to_be_bytes()
}

fn el(id: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&size_vint_8(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

fn el_uint(id: &[u8], v: u64) -> Vec<u8> {
    let bytes = v.to_be_bytes();
    let skip = bytes.iter().take(7).take_while(|&&b| b == 0).count();
    el(id, &bytes[skip..])
}

/// One audio track, `blocks` as SimpleBlocks of one cluster, 10 ms apart.
fn mkv(
    codec_id: &str,
    codec_private: &[u8],
    rate: f64,
    channels: u64,
    bits: u64,
    blocks: &[Vec<u8>],
) -> Vec<u8> {
    let ebml = el(
        &[0x1A, 0x45, 0xDF, 0xA3],
        &[
            el(&[0x42, 0x82], b"matroska"),
            el_uint(&[0x42, 0x87], 4),
            el_uint(&[0x42, 0x85], 2),
        ]
        .concat(),
    );
    // MuxingApp and WritingApp are mandatory Info children.
    let info = el(
        &[0x15, 0x49, 0xA9, 0x66],
        &[
            el_uint(&[0x2A, 0xD7, 0xB1], 1_000_000),
            el(&[0x4D, 0x80], b"test"),
            el(&[0x57, 0x41], b"test"),
        ]
        .concat(),
    );
    let mut audio = el(&[0xB5], &rate.to_be_bytes());
    audio.extend(el_uint(&[0x9F], channels));
    if bits > 0 {
        audio.extend(el_uint(&[0x62, 0x64], bits));
    }
    let mut entry = [
        el_uint(&[0xD7], 1),
        el_uint(&[0x73, 0xC5], 7),
        el_uint(&[0x83], 2),
        el(&[0x86], codec_id.as_bytes()),
    ]
    .concat();
    if !codec_private.is_empty() {
        entry.extend(el(&[0x63, 0xA2], codec_private));
    }
    entry.extend(el(&[0xE1], &audio));
    let tracks = el(&[0x16, 0x54, 0xAE, 0x6B], &el(&[0xAE], &entry));
    let mut cluster = el_uint(&[0xE7], 0);
    for (i, b) in blocks.iter().enumerate() {
        let mut block = vec![0x81];
        block.extend_from_slice(&((i * 10) as i16).to_be_bytes());
        block.push(0x80);
        block.extend_from_slice(b);
        cluster.extend(el(&[0xA3], &block));
    }
    let segment = [info, tracks, el(&[0x1F, 0x43, 0xB6, 0x75], &cluster)].concat();
    [ebml, el(&[0x18, 0x53, 0x80, 0x67], &segment)].concat()
}

fn decode(track: &container::demux::AudioTrack) -> Vec<f32> {
    let mut dec =
        codec::audio::create_decoder(&track.codec, None, track.sample_rate, track.channels as u8)
            .expect("decoder");
    let mut out = Vec::new();
    for p in &track.samples {
        for f in dec.decode(p, 0).expect("decode") {
            out.extend(f.samples);
        }
    }
    out
}

#[test]
fn matroska_pcm_in_each_mapping_reads_and_decodes_exactly() {
    // 480 stereo frames per block (10 ms at 48 kHz), three blocks.
    let values: Vec<i32> = (0..3 * 480 * 2)
        .map(|i| ((i * 7919) % 65_536) - 32_768)
        .collect();
    let waveformatex = |tag: u16, bits: u16| {
        let mut f = Vec::new();
        f.extend_from_slice(&tag.to_le_bytes());
        f.extend_from_slice(&2u16.to_le_bytes());
        f.extend_from_slice(&48_000u32.to_le_bytes());
        f.extend_from_slice(&(48_000u32 * 2 * u32::from(bits / 8)).to_le_bytes());
        f.extend_from_slice(&(2 * bits / 8).to_le_bytes());
        f.extend_from_slice(&bits.to_le_bytes());
        f.extend_from_slice(&0u16.to_le_bytes());
        f
    };
    // (codec ID, CodecPrivate, BitDepth, encode one sample, its expected value)
    type Enc = fn(i32) -> (Vec<u8>, f32);
    let cases: Vec<(&str, Vec<u8>, u64, Enc, &str)> = vec![
        (
            "A_PCM/INT/LIT",
            vec![],
            16,
            |v| ((v as i16).to_le_bytes().to_vec(), v as f32 / 32768.0),
            "pcm_s16le",
        ),
        (
            "A_PCM/INT/BIG",
            vec![],
            16,
            |v| ((v as i16).to_be_bytes().to_vec(), v as f32 / 32768.0),
            "pcm_s16le",
        ),
        (
            "A_PCM/INT/BIG",
            vec![],
            24,
            |v| {
                (
                    (v << 8).to_be_bytes()[1..].to_vec(),
                    (v << 8) as f32 / 8_388_608.0,
                )
            },
            "pcm_s24le",
        ),
        // 8-bit is unsigned in both integer mappings.
        (
            "A_PCM/INT/LIT",
            vec![],
            8,
            |v| (vec![((v >> 8) + 128) as u8], (v >> 8) as f32 / 128.0),
            "pcm_u8",
        ),
        (
            "A_PCM/FLOAT/IEEE",
            vec![],
            32,
            |v| {
                (
                    (v as f32 / 32768.0).to_le_bytes().to_vec(),
                    v as f32 / 32768.0,
                )
            },
            "pcm_f32le",
        ),
        (
            "A_MS/ACM",
            waveformatex(1, 16),
            0,
            |v| ((v as i16).to_le_bytes().to_vec(), v as f32 / 32768.0),
            "pcm_s16le",
        ),
    ];
    for (codec_id, private, bits, enc, codec) in cases {
        let name = format!("{codec_id} {bits}");
        let mut expect = Vec::new();
        let blocks: Vec<Vec<u8>> = values
            .chunks(960)
            .map(|block| {
                block
                    .iter()
                    .flat_map(|&v| {
                        let (b, e) = enc(v);
                        expect.push(e);
                        b
                    })
                    .collect()
            })
            .collect();
        let file = mkv(codec_id, &private, 48_000.0, 2, bits, &blocks);
        let src = demux_audio(Bytes::from(file))
            .expect("demux")
            .unwrap_or_else(|| panic!("{name}: no audio"));
        assert_eq!(src.track.codec, codec, "{name}");
        assert_eq!(
            (src.track.sample_rate, src.track.channels),
            (48_000, 2),
            "{name}"
        );
        assert_eq!(
            src.track.durations,
            [480, 480, 480],
            "{name}: a block lasts its frames"
        );
        let got = decode(&src.track);
        assert_eq!(got.len(), expect.len(), "{name}");
        for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
            assert!((g - e).abs() < 1e-6, "{name}: sample {i}: {g} vs {e}");
        }
    }
}

#[test]
fn a_matroska_audio_codec_without_a_reader_is_named() {
    let file = mkv("A_TRUEHD", &[], 48_000.0, 6, 24, &[vec![0; 64]]);
    let src = demux_audio(Bytes::from(file))
        .expect("demux")
        .expect("a named track");
    assert_eq!(src.track.codec, "truehd");
    assert!(src.track.samples.is_empty());
    assert_eq!(src.track.channels, 6);
}
