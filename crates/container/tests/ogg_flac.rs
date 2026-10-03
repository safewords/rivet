//! FLAC in Ogg (Xiph's "FLAC to Ogg mapping"), on a file written here with
//! the workspace's own Ogg page writer and FLAC encoder: the mapping's
//! first packet (`0x7F "FLAC"`, version 1.0, the header-packet count,
//! `fLaC`, STREAMINFO), a Vorbis-comment metadata block as the one further
//! header packet, then one frame per packet — read by `demux_audio` and
//! decoded bit for bit.

use bytes::Bytes;
use codec::audio::encode::flac::FlacLevel;
use codec::audio::{AudioCodec, AudioEncoderConfig, AudioFrame, create_decoder, create_encoder};
use container::streaming::demux_audio;
use vorbis::ogg::PacketWriter;

#[test]
fn flac_in_ogg_reads_and_decodes_bit_exact() {
    const RATE: u32 = 44_100;
    let values: Vec<i16> = (0..RATE as usize * 2).map(|i| (((i * 7919) % 50_000) as i32 - 25_000) as i16).collect();
    let pcm: Vec<f32> = values.iter().map(|&v| f32::from(v) / 32768.0).collect();
    let mut enc = create_encoder(AudioEncoderConfig::new(
        AudioCodec::Flac { bits_per_sample: 16, level: FlacLevel::Default },
        RATE,
        2,
        0,
    ))
    .expect("flac encoder");
    let mut frames = enc.encode(&AudioFrame { samples: pcm.clone(), sample_rate: RATE, channels: 2, pts: 0 }).unwrap();
    frames.extend(enc.flush().unwrap());
    let streaminfo = enc.extra_data();
    assert_eq!(streaminfo.len(), 38, "STREAMINFO, flagged last");

    // The first packet: the mapping header, then fLaC and STREAMINFO (not
    // flagged last: a comment block follows as the one more header packet).
    let mut first = vec![0x7F];
    first.extend_from_slice(b"FLAC");
    first.extend_from_slice(&[1, 0]);
    first.extend_from_slice(&1u16.to_be_bytes());
    first.extend_from_slice(b"fLaC");
    first.extend_from_slice(&streaminfo);
    first[13] &= 0x7F;
    // VORBIS_COMMENT (type 4), last: an empty vendor string and no comments.
    let comment = [vec![0x84, 0, 0, 8], vec![0; 8]].concat();

    let mut file = Vec::new();
    {
        let mut w = PacketWriter::new(&mut file, 0x1234);
        w.write_packet(&first, 0, true, false).unwrap();
        w.write_packet(&comment, 0, true, false).unwrap();
        let mut granule = 0i64;
        for (i, f) in frames.iter().enumerate() {
            granule += 4096.min(values.len() as i64 / 2 - granule);
            w.write_packet(&f.data, granule, true, i + 1 == frames.len()).unwrap();
        }
    }
    let src = demux_audio(Bytes::from(file)).expect("demux").expect("the FLAC track");
    let t = &src.track;
    assert_eq!((t.codec.as_str(), t.sample_rate, t.channels), ("flac", RATE, 2));
    assert_eq!(t.samples.len(), frames.len());
    assert_eq!(t.durations.iter().map(|&d| d as usize).sum::<usize>(), values.len() / 2, "each frame's own count");
    let mut dec = create_decoder("flac", Some(&t.codec_private), t.sample_rate, 2).expect("decoder");
    let mut got = Vec::new();
    for p in &t.samples {
        for f in dec.decode(p, 0).unwrap() {
            got.extend(f.samples);
        }
    }
    assert_eq!(got, pcm, "lossless");
}

/// A stream rivet does not read is refused with the reader's own words —
/// not a hint to use the audio-only output mode the caller already chose.
#[test]
fn an_ogg_stream_without_a_reader_says_so() {
    let mut file = Vec::new();
    {
        let mut w = PacketWriter::new(&mut file, 7);
        w.write_packet(b"Speex   1.2rc1\0\0\0\0\0\0\0\0\0\0", 0, true, false).unwrap();
        w.write_packet(&[0; 20], 160, true, true).unwrap();
    }
    let err = demux_audio(Bytes::from(file.clone())).expect_err("no reader").to_string();
    assert!(err.contains("no Opus, Vorbis or FLAC stream"), "{err}");
    let video_err = container::streaming::demux_streaming(&file).err().expect("no video").to_string();
    assert!(!video_err.contains("audio-only output mode"), "{video_err}");
}
