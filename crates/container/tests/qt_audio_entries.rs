//! QuickTime sound sample descriptions (version 0, 1 and 2) read through
//! `demux_audio`, on movies written here from the QuickTime File Format
//! specification: linear PCM in each of its encodings, and ALAC whose magic
//! cookie sits in a `wave` atom — then decoded, and compared with the
//! samples that went in.

use bytes::Bytes;
use container::streaming::demux_audio;

fn boxed(fourcc: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = ((8 + body.len()) as u32).to_be_bytes().to_vec();
    out.extend_from_slice(fourcc);
    out.extend_from_slice(body);
    out
}

fn full(fourcc: &[u8; 4], version: u8, body: &[u8]) -> Vec<u8> {
    let mut b = vec![version, 0, 0, 0];
    b.extend_from_slice(body);
    boxed(fourcc, &b)
}

fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

/// A sound description of `version` (0, 1 or 2) for `fourcc`, with
/// `children` after its fixed fields.
fn sound_entry(fourcc: &[u8; 4], version: u16, channels: u16, bits: u16, rate: u32, children: &[u8]) -> Vec<u8> {
    let mut b = vec![0u8; 6];
    b.extend_from_slice(&1u16.to_be_bytes()); // data reference index
    b.extend_from_slice(&version.to_be_bytes());
    b.extend_from_slice(&[0; 6]); // revision level, vendor
    let bytes = u32::from(bits).div_ceil(8);
    match version {
        2 => {
            // Placeholders, then the version 2 fields (QuickTime File Format,
            // "Sound Sample Description (Version 2)").
            b.extend_from_slice(&3u16.to_be_bytes());
            b.extend_from_slice(&16u16.to_be_bytes());
            b.extend_from_slice(&(-2i16).to_be_bytes());
            b.extend_from_slice(&0u16.to_be_bytes());
            b.extend_from_slice(&be32(65536));
            b.extend_from_slice(&be32(72)); // sizeOfStructOnly
            b.extend_from_slice(&f64::from(rate).to_bits().to_be_bytes());
            b.extend_from_slice(&be32(u32::from(channels)));
            b.extend_from_slice(&be32(0x7F00_0000));
            b.extend_from_slice(&be32(u32::from(bits)));
            // kAudioFormatFlagIsFloat | IsPacked: little-endian float.
            b.extend_from_slice(&be32(1 | 8));
            b.extend_from_slice(&be32(bytes * u32::from(channels)));
            b.extend_from_slice(&be32(1));
        }
        _ => {
            b.extend_from_slice(&channels.to_be_bytes());
            b.extend_from_slice(&bits.to_be_bytes());
            b.extend_from_slice(&(if version == 1 { -2i16 } else { 0 }).to_be_bytes());
            b.extend_from_slice(&0u16.to_be_bytes());
            b.extend_from_slice(&be32(rate << 16));
            if version == 1 {
                for v in [1, bytes, bytes * u32::from(channels), bytes] {
                    b.extend_from_slice(&be32(v));
                }
            }
        }
    }
    b.extend_from_slice(children);
    boxed(fourcc, &b)
}

/// One audio track's movie: `ftyp qt`, the `mdat` with every chunk, then a
/// `moov` whose tables give `samples_per_chunk` samples of `sample_size`
/// bytes (0: per-sample `sizes`) and `delta` ticks to each chunk.
struct Track<'a> {
    entry: Vec<u8>,
    chunks: &'a [Vec<u8>],
    samples_per_chunk: &'a [u32],
    sizes: Vec<u32>,
    sample_size: u32,
    delta: u32,
    timescale: u32,
}

fn movie(t: Track<'_>) -> Vec<u8> {
    let mut out = boxed(b"ftyp", b"qt  \0\0\x02\0qt  ");
    let mdat_body: Vec<u8> = t.chunks.concat();
    let mdat_at = out.len() + 8;
    out.extend(boxed(b"mdat", &mdat_body));
    let mut offsets = Vec::new();
    let mut at = mdat_at as u32;
    for c in t.chunks {
        offsets.push(at);
        at += c.len() as u32;
    }
    let total: u32 = t.samples_per_chunk.iter().sum();
    let mut stsd = be32(1).to_vec();
    stsd.extend_from_slice(&t.entry);
    let mut stts = be32(1).to_vec();
    stts.extend_from_slice(&be32(total));
    stts.extend_from_slice(&be32(t.delta));
    let mut stsc = be32(t.samples_per_chunk.len() as u32).to_vec();
    for (i, n) in t.samples_per_chunk.iter().enumerate() {
        stsc.extend_from_slice(&be32(i as u32 + 1));
        stsc.extend_from_slice(&be32(*n));
        stsc.extend_from_slice(&be32(1));
    }
    let mut stsz = be32(t.sample_size).to_vec();
    stsz.extend_from_slice(&be32(total));
    if t.sample_size == 0 {
        for s in &t.sizes {
            stsz.extend_from_slice(&be32(*s));
        }
    }
    let mut stco = be32(offsets.len() as u32).to_vec();
    for o in offsets {
        stco.extend_from_slice(&be32(o));
    }
    let stbl = [
        full(b"stsd", 0, &stsd),
        full(b"stts", 0, &stts),
        full(b"stsc", 0, &stsc),
        full(b"stsz", 0, &stsz),
        full(b"stco", 0, &stco),
    ]
    .concat();
    let duration = total * t.delta;
    let mut mdhd = vec![0u8; 8];
    mdhd.extend_from_slice(&be32(t.timescale));
    mdhd.extend_from_slice(&be32(duration));
    mdhd.extend_from_slice(&[0; 4]);
    let mut hdlr = be32(0).to_vec();
    hdlr.extend_from_slice(b"soun");
    hdlr.extend_from_slice(&[0; 13]);
    let mut dref = be32(1).to_vec();
    dref.extend(full(b"url ", 1, &[]));
    let minf = [full(b"smhd", 0, &[0; 4]), boxed(b"dinf", &full(b"dref", 0, &dref)), boxed(b"stbl", &stbl)].concat();
    let mdia = [full(b"mdhd", 0, &mdhd), full(b"hdlr", 0, &hdlr), boxed(b"minf", &minf)].concat();
    let mut tkhd = vec![0u8; 8];
    tkhd.extend_from_slice(&be32(1));
    tkhd.extend_from_slice(&[0; 4]);
    tkhd.extend_from_slice(&be32(duration));
    tkhd.extend_from_slice(&[0; 52]);
    tkhd.extend_from_slice(&[0; 8]);
    let trak = [full(b"tkhd", 0, &tkhd), boxed(b"mdia", &mdia)].concat();
    let mut mvhd = vec![0u8; 8];
    mvhd.extend_from_slice(&be32(t.timescale));
    mvhd.extend_from_slice(&be32(duration));
    mvhd.extend_from_slice(&[0; 80]);
    let moov = [full(b"mvhd", 0, &mvhd), boxed(b"trak", &trak)].concat();
    out.extend(boxed(b"moov", &moov));
    out
}

/// A stereo test signal: two ramps, as i32 samples at `bits`.
fn signal(frames: usize, bits: u32) -> Vec<i32> {
    let max = (1i64 << (bits - 1)) - 1;
    (0..frames * 2)
        .map(|i| {
            let (f, c) = ((i / 2) as i64, (i % 2) as i64);
            let v = ((f * 37 + c * 1000) % (2 * max)) - max;
            v as i32
        })
        .collect()
}

/// Decode a demuxed track to f32 with rivet's decoder for its codec.
fn decode(track: &container::demux::AudioTrack) -> Vec<f32> {
    let extra = if track.codec_private.is_empty() { None } else { Some(track.codec_private.as_slice()) };
    let mut dec =
        codec::audio::create_decoder(&track.codec, extra, track.sample_rate, track.channels as u8).expect("decoder");
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

/// Split interleaved bytes into chunks of `frames` frames.
fn chunked(bytes: &[u8], frame_bytes: usize, frames: &[u32]) -> Vec<Vec<u8>> {
    let mut at = 0;
    frames
        .iter()
        .map(|&n| {
            let len = n as usize * frame_bytes;
            let c = bytes[at..at + len].to_vec();
            at += len;
            c
        })
        .collect()
}

/// The PCM encodings, each written as its sample description lays it out,
/// demux to the decoder's little-endian forms and decode to the signal.
#[test]
fn every_quicktime_pcm_encoding_reads_and_decodes_exactly() {
    const FRAMES: usize = 3000;
    let frames_per_chunk = [1024u32, 1024, 952];
    let wave_enda = |fourcc: &[u8; 4]| boxed(b"wave", &[boxed(b"frma", fourcc), boxed(b"enda", &[0, 1])].concat());
    // (fourcc, version, bits, little-endian, float, children, expected codec)
    let cases: Vec<(&[u8; 4], u16, u16, bool, bool, Vec<u8>, &str)> = vec![
        (b"twos", 0, 16, false, false, Vec::new(), "pcm_s16le"),
        (b"sowt", 0, 16, true, false, Vec::new(), "pcm_s16le"),
        (b"twos", 0, 8, false, false, Vec::new(), "pcm_u8"),
        (b"raw ", 0, 8, false, false, Vec::new(), "pcm_u8"),
        (b"in24", 1, 24, false, false, Vec::new(), "pcm_s24le"),
        (b"in24", 1, 24, true, false, wave_enda(b"in24"), "pcm_s24le"),
        (b"in32", 1, 32, false, false, Vec::new(), "pcm_s32le"),
        (b"fl32", 1, 32, false, true, Vec::new(), "pcm_f32le"),
        (b"fl64", 1, 64, true, true, wave_enda(b"fl64"), "pcm_f64le"),
        (b"lpcm", 2, 32, true, true, Vec::new(), "pcm_f32le"),
    ];
    for (fourcc, version, bits, little, float, children, codec) in cases {
        let name = format!("{} v{version} {bits}-bit", String::from_utf8_lossy(fourcc));
        let bytes_per = usize::from(bits) / 8;
        let ints = signal(FRAMES, 24);
        // The expected decode, and the stored bytes.
        let mut stored = Vec::new();
        let mut expect = Vec::new();
        for &v in &ints {
            let (le, value): (Vec<u8>, f64) = if float {
                let x = f64::from(v) / f64::from(1 << 23);
                (if bits == 32 { (x as f32).to_le_bytes().to_vec() } else { x.to_le_bytes().to_vec() }, x)
            } else {
                match bits {
                    8 => {
                        let s = (v >> 16) as i8;
                        let raw = fourcc == b"raw ";
                        // `raw ` is offset binary; `twos` two's complement.
                        (vec![if raw { (s as u8) ^ 0x80 } else { s as u8 }], f64::from(s) / 128.0)
                    }
                    16 => {
                        let s = (v >> 8) as i16;
                        (s.to_le_bytes().to_vec(), f64::from(s) / 32768.0)
                    }
                    24 => (v.to_le_bytes()[..3].to_vec(), f64::from(v) / 8_388_608.0),
                    _ => {
                        let s = v << 8;
                        (s.to_le_bytes().to_vec(), f64::from(s) / 2_147_483_648.0)
                    }
                }
            };
            let mut b = le;
            if !little {
                b.reverse();
            }
            stored.extend(b);
            expect.push(value as f32);
        }
        let entry = sound_entry(fourcc, version, 2, bits, 48_000, &children);
        let chunks = chunked(&stored, bytes_per * 2, &frames_per_chunk);
        let file = movie(Track {
            entry,
            chunks: &chunks,
            samples_per_chunk: &frames_per_chunk,
            sizes: Vec::new(),
            // QuickTime's placeholder for uncompressed audio.
            sample_size: 1,
            delta: 1,
            timescale: 48_000,
        });
        let src = demux_audio(Bytes::from(file)).expect("demux").unwrap_or_else(|| panic!("{name}: no audio"));
        let t = &src.track;
        assert_eq!(t.codec, codec, "{name}");
        assert_eq!((t.sample_rate, t.channels, t.timescale), (48_000, 2, 48_000), "{name}");
        assert_eq!(t.samples.len(), 3, "{name}: one packet per chunk");
        assert_eq!(t.durations, frames_per_chunk, "{name}");
        let got = decode(t);
        assert_eq!(got.len(), expect.len(), "{name}");
        for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
            assert!((g - e).abs() < 1e-6, "{name}: sample {i}: {g} vs {e}");
        }
    }
}

/// ALAC in a `.mov`: a version 1 description whose magic cookie is inside
/// `wave` (`frma`, then the `alac` atom), as QuickTime writes it.
#[test]
fn alac_with_its_cookie_in_a_wave_atom_reads_and_decodes_bit_exact() {
    use codec::audio::{AudioCodec, AudioEncoderConfig, AudioFrame, create_encoder};
    const FRAMES: usize = 10_000;
    let ints = signal(FRAMES, 24);
    let pcm: Vec<f32> = ints.iter().map(|&v| ((v >> 8) as i16) as f32 / 32768.0).collect();
    let mut enc = create_encoder(AudioEncoderConfig::new(AudioCodec::Alac { bits_per_sample: 16 }, 44_100, 2, 0))
    .expect("alac encoder");
    let mut packets = enc.encode(&AudioFrame { samples: pcm.clone(), sample_rate: 44_100, channels: 2, pts: 0 }).unwrap();
    packets.extend(enc.flush().unwrap());
    let cookie = enc.extra_data();
    assert_eq!(cookie.len(), 24, "the bare ALACSpecificConfig");
    let wave = [boxed(b"frma", b"alac"), full(b"alac", 0, &cookie), [0u8, 0, 0, 8, 0, 0, 0, 0].to_vec()].concat();
    let entry = sound_entry(b"alac", 1, 2, 16, 44_100, &boxed(b"wave", &wave));
    let chunks: Vec<Vec<u8>> = packets.iter().map(|p| p.data.clone()).collect();
    let file = movie(Track {
        entry,
        sizes: chunks.iter().map(|c| c.len() as u32).collect(),
        samples_per_chunk: &vec![1; chunks.len()],
        chunks: &chunks,
        sample_size: 0,
        delta: 4096,
        timescale: 44_100,
    });
    let src = demux_audio(Bytes::from(file)).expect("demux").expect("the ALAC track");
    assert_eq!(src.track.codec, "alac");
    assert_eq!(src.track.codec_private, cookie);
    assert_eq!(src.track.samples.len(), chunks.len());
    let got = decode(&src.track);
    assert_eq!(got, pcm, "lossless");
}

/// An audio entry rivet has no reader for (AMR, 3GPP TS 26.244) is not
/// mistaken for "no audio": it is surfaced by name, with no packets.
#[test]
fn an_unsupported_audio_entry_is_named_not_hidden() {
    let damr = boxed(b"damr", &[b'r', b'v', b't', b' ', 0, 0x81, 0xFF, 0, 1]);
    let mut b = vec![0u8; 6];
    b.extend_from_slice(&1u16.to_be_bytes());
    b.extend_from_slice(&[0; 8]);
    b.extend_from_slice(&1u16.to_be_bytes());
    b.extend_from_slice(&16u16.to_be_bytes());
    b.extend_from_slice(&[0; 4]);
    b.extend_from_slice(&be32(8000 << 16));
    b.extend(damr);
    let chunks = vec![vec![0x3C; 32]];
    let file = movie(Track {
        entry: boxed(b"samr", &b),
        chunks: &chunks,
        samples_per_chunk: &[1],
        sizes: vec![32],
        sample_size: 0,
        delta: 160,
        timescale: 8000,
    });
    let src = demux_audio(Bytes::from(file)).expect("demux").expect("a named track");
    assert_eq!(src.track.codec, "amr_nb");
    assert!(src.track.samples.is_empty());
}
