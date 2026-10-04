//! The elementary-stream readers over streams rivet's own encoders write,
//! framed here by spec-derived writers (IVF's header and frame headers, AV1
//! Annex B's length fields): each sniffs as its kind, demuxes to one sample
//! a picture, and every sample decodes.

use bytes::Bytes;
use codec::encode::{Encoder, EncoderConfig, QualityTarget, SpeedTier};
use frame::{ColorMetadata, ColorSpace, PixelFormat, VideoCodec, VideoFrame};

use super::bits::{leb128, write_leb128};
use crate::sniff::{ContainerKind, sniff_container};
use crate::streaming::demux_streaming;

const W: u32 = 176;
const H: u32 = 144;
const FRAMES: u64 = 6;

/// A moving gradient, so every picture differs.
fn picture(n: u64) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let mut data = Vec::with_capacity(w * h * 3 / 2);
    for y in 0..h {
        for x in 0..w {
            data.push(((x + y + n as usize * 9) % 200 + 20) as u8);
        }
    }
    data.extend(std::iter::repeat_n(128u8, w * h / 2));
    VideoFrame::new(
        data.into(),
        W,
        H,
        PixelFormat::Yuv420p,
        ColorSpace::Bt709,
        n,
    )
}

/// `FRAMES` pictures through rivet's own software encoder for `codec`, as
/// the packets it hands out.
fn encode(codec: VideoCodec) -> Vec<Vec<u8>> {
    let cfg = EncoderConfig {
        width: W,
        height: H,
        frame_rate: 30.0,
        quality: u8::MAX,
        speed_preset: u8::MAX,
        keyframe_interval: 30,
        target: QualityTarget::Standard,
        tier: SpeedTier::Draft,
        threads: 1,
        pixel_format: PixelFormat::Yuv420p,
        color_metadata: ColorMetadata::default(),
        gpu_index: None,
        gpu_vendor: None,
        codec,
        constant_qp: false,
        overrides: Default::default(),
    };
    let mut enc: Box<dyn Encoder> = match codec {
        VideoCodec::H264 | VideoCodec::H265 => {
            Box::new(codec::encode::h26x_sw::H26xEncoder::new(cfg).unwrap())
        }
        VideoCodec::Vp8 => Box::new(codec::encode::vp8_sw::Vp8Encoder::new(cfg).unwrap()),
        VideoCodec::Vp9 => Box::new(codec::encode::vp9_sw::Vp9Encoder::new(cfg).unwrap()),
        VideoCodec::Av1 => Box::new(codec::encode::av1_sw::Av1Encoder::new(cfg).unwrap()),
        VideoCodec::Mpeg2 => Box::new(codec::encode::mpeg2_sw::Mpeg2Encoder::new(cfg).unwrap()),
        other => panic!("no fixture encoder for {other:?}"),
    };
    for n in 0..FRAMES {
        enc.send_frame(&picture(n)).unwrap();
    }
    enc.flush().unwrap();
    let mut out = Vec::new();
    while let Some(p) = enc.receive_packet().unwrap() {
        out.push(p.data.to_vec());
    }
    assert_eq!(out.len() as u64, FRAMES, "{codec:?}: one packet a picture");
    out
}

/// Demux `file`, check the header, decode every sample; the frames made.
fn demux_and_decode(
    file: Vec<u8>,
    kind: ContainerKind,
    codec: &str,
) -> (crate::streaming::DemuxHeader, u64) {
    assert_eq!(sniff_container(&file), kind);
    let mut d = demux_streaming(&file).expect("demux");
    let header = d.header().clone();
    assert_eq!(header.codec, codec);
    assert_eq!(
        (header.info.width, header.info.height),
        (W, H),
        "{codec}: dimensions"
    );
    assert_eq!(header.info.pixel_format, PixelFormat::Yuv420p);
    assert!(d.audio().is_none());
    let mut dec = codec::decode::create_decoder(codec, header.info.clone()).expect("decoder");
    let (mut samples, mut frames) = (0u64, 0u64);
    let mut last_pts = -1;
    while let Some(s) = d.next_video_sample().unwrap() {
        assert!(s.pts_ticks > last_pts, "timestamps ascend");
        last_pts = s.pts_ticks;
        samples += 1;
        dec.push_sample(&s.data)
            .unwrap_or_else(|e| panic!("{codec}: sample {samples}: {e:#}"));
        while let Some(f) = dec.decode_next().unwrap() {
            assert_eq!((f.width, f.height), (W, H));
            frames += 1;
        }
    }
    dec.finish().unwrap();
    while dec.decode_next().unwrap().is_some() {
        frames += 1;
    }
    assert_eq!(
        samples, header.info.total_frames,
        "{codec}: one sample a frame"
    );
    (header, frames)
}

#[test]
fn annex_b_h264_is_read_access_unit_by_access_unit() {
    let file: Vec<u8> = encode(VideoCodec::H264).concat();
    let (header, frames) = demux_and_decode(file, ContainerKind::H264Es, "h264");
    assert_eq!(frames, FRAMES);
    assert!(
        (header.info.frame_rate - 30.0).abs() < 1e-6,
        "the VUI's rate: {}",
        header.info.frame_rate
    );
}

#[test]
fn annex_b_hevc_is_read_access_unit_by_access_unit() {
    let file: Vec<u8> = encode(VideoCodec::H265).concat();
    let (header, frames) = demux_and_decode(file, ContainerKind::HevcEs, "h265");
    assert_eq!(frames, FRAMES);
    assert!(
        (header.info.frame_rate - 30.0).abs() < 1e-6,
        "the VUI's rate: {}",
        header.info.frame_rate
    );
}

/// An IVF file (the WebM project's layout): `DKIF`, version 0, header length
/// 32, the fourcc, the size, the time base as rate and scale, the frame
/// count; each frame behind its 32-bit size and 64-bit timestamp.
fn ivf(fourcc: &[u8; 4], frames: &[Vec<u8>], rate: u32, scale: u32, step: u64) -> Vec<u8> {
    let mut f = b"DKIF".to_vec();
    f.extend(0u16.to_le_bytes());
    f.extend(32u16.to_le_bytes());
    f.extend(fourcc);
    f.extend((W as u16).to_le_bytes());
    f.extend((H as u16).to_le_bytes());
    f.extend(rate.to_le_bytes());
    f.extend(scale.to_le_bytes());
    f.extend((frames.len() as u32).to_le_bytes());
    f.extend([0; 4]);
    for (i, frame) in frames.iter().enumerate() {
        f.extend((frame.len() as u32).to_le_bytes());
        f.extend((i as u64 * step).to_le_bytes());
        f.extend(frame);
    }
    f
}

#[test]
fn ivf_carries_vp8_vp9_and_av1() {
    for (codec, fourcc, label) in [
        (VideoCodec::Vp8, b"VP80", "vp8"),
        (VideoCodec::Vp9, b"VP90", "vp9"),
        (VideoCodec::Av1, b"AV01", "av1"),
    ] {
        // A millisecond time base, 40 ms a frame: 25 fps from the timestamps.
        let file = ivf(fourcc, &encode(codec), 1000, 1, 40);
        let (header, frames) = demux_and_decode(file, ContainerKind::Ivf, label);
        assert_eq!(frames, FRAMES, "{label}");
        assert!(
            (header.info.frame_rate - 25.0).abs() < 1e-6,
            "{label}: {}",
            header.info.frame_rate
        );
        assert_eq!(header.timescale, 1000);
    }
}

/// The OBUs of a low-overhead temporal unit: `(header bytes, payload)`.
fn obus(tu: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < tu.len() {
        let h = tu[pos];
        let hlen = if h & 0x04 != 0 { 2 } else { 1 };
        assert!(h & 0x02 != 0, "the encoder writes sized OBUs");
        let (size, n) = leb128(&tu[pos + hlen..]).unwrap();
        let body = pos + hlen + n;
        out.push((
            tu[pos..pos + hlen].to_vec(),
            tu[body..body + size as usize].to_vec(),
        ));
        pos = body + size as usize;
    }
    out
}

#[test]
fn av1_obu_streams_in_both_formats() {
    let packets = encode(VideoCodec::Av1);
    // Low-overhead (§5.2): a temporal delimiter, then the unit's OBUs.
    let mut section5 = Vec::new();
    for p in &packets {
        section5.extend([0x12, 0x00]);
        section5.extend(
            obus(p)
                .into_iter()
                .filter(|(h, _)| (h[0] >> 3) & 0xf != 2)
                .flat_map(|(h, b)| {
                    let mut o = h;
                    write_leb128(b.len() as u64, &mut o);
                    o.extend(b);
                    o
                }),
        );
    }
    // Annex B: temporal_unit(size) { frame_unit(size) { obu_length obu } },
    // the OBUs without size fields — here one frame unit per temporal unit.
    let mut annex_b = Vec::new();
    for p in &packets {
        let mut fu = Vec::new();
        let mut put = |header: &[u8], body: &[u8]| {
            let mut obu = vec![header[0] & !0x02];
            obu.extend(&header[1..]);
            obu.extend(body);
            write_leb128(obu.len() as u64, &mut fu);
            fu.extend(obu);
        };
        put(&[0x10], &[]);
        for (h, b) in obus(p).into_iter().filter(|(h, _)| (h[0] >> 3) & 0xf != 2) {
            put(&h, &b);
        }
        let mut tu = Vec::new();
        write_leb128(fu.len() as u64, &mut tu);
        tu.extend(fu);
        write_leb128(tu.len() as u64, &mut annex_b);
        annex_b.extend(tu);
    }
    for (name, file) in [("section 5", section5), ("annex b", annex_b)] {
        assert!(super::obu::sniff(&file).is_some(), "{name}");
        let (header, frames) = demux_and_decode(file, ContainerKind::Av1Obu, "av1");
        assert_eq!(frames, FRAMES, "{name}");
        // The encoder writes no timing info: the default rate.
        assert!(header.info.frame_rate > 0.0, "{name}");
    }
}

#[test]
fn mpeg2_video_elementary_stream() {
    let file: Vec<u8> = encode(VideoCodec::Mpeg2).concat();
    let (header, frames) = demux_and_decode(file, ContainerKind::MpegVideoEs, "mpeg2");
    assert_eq!(frames, FRAMES);
    // frame_rate_code 5 (30) in the sequence header.
    assert!(
        (header.info.frame_rate - 30.0).abs() < 1e-6,
        "{}",
        header.info.frame_rate
    );
}

#[test]
fn a_stream_stating_no_rate_is_given_the_default() {
    // An AV1 stream without timing info: 25 fps assumed.
    let mut file = Vec::new();
    for p in encode(VideoCodec::Av1) {
        file.extend([0x12, 0x00]);
        file.extend(
            p.iter()
                .copied()
                .skip(super::obu::leading_temporal_delimiter(&p)),
        );
    }
    let d = demux_streaming(&file).unwrap();
    assert_eq!(d.header().info.frame_rate, super::DEFAULT_FRAME_RATE);
    assert_eq!(d.header().info.total_frames, FRAMES);
    assert!((d.header().info.duration - FRAMES as f64 / 25.0).abs() < 1e-9);
}

#[test]
fn h264_and_hevc_do_not_read_as_each_other_and_noise_is_neither() {
    let h264: Vec<u8> = encode(VideoCodec::H264).concat();
    let hevc: Vec<u8> = encode(VideoCodec::H265).concat();
    assert_eq!(super::annexb::sniff(&h264), Some(ContainerKind::H264Es));
    assert_eq!(super::annexb::sniff(&hevc), Some(ContainerKind::HevcEs));
    // Start codes around bytes that are no NAL unit of either codec.
    let mut junk = vec![0, 0, 0, 1, 0x80, 1, 2, 3];
    junk.extend([0, 0, 1, 0xff, 0xee, 0, 0, 1, 0x41, 0x55]);
    assert_eq!(sniff_container(&junk), ContainerKind::Unknown);
    // A stream cut before its first slice says nothing.
    assert_eq!(super::annexb::sniff(&h264[..20]), None);
    // A text file, zeros.
    assert_eq!(sniff_container(&[0u8; 400]), ContainerKind::Unknown);
    assert_eq!(
        super::sniff(b"0000 0001 is not a start code, it is text"),
        None
    );
}

#[test]
fn an_ivf_of_another_codec_is_refused_by_name() {
    let file = ivf(b"XYZ1", &[vec![1, 2, 3]], 30, 1, 1);
    assert_eq!(sniff_container(&file), ContainerKind::Ivf);
    let err = demux_streaming(&file).err().expect("refused").to_string();
    assert!(err.contains("XYZ1"), "{err}");
}

#[test]
fn the_elementary_stream_labels() {
    for (kind, label) in [
        (ContainerKind::H264Es, "h264"),
        (ContainerKind::HevcEs, "hevc"),
        (ContainerKind::Ivf, "ivf"),
        (ContainerKind::Av1Obu, "obu"),
        (ContainerKind::MpegVideoEs, "m2v"),
    ] {
        assert_eq!(kind.label(), label);
        assert_eq!(
            kind.is_video_elementary_stream(),
            kind != ContainerKind::Ivf
        );
    }
    assert!(!ContainerKind::IsoBmff.is_video_elementary_stream());
    // The shared buffer path reads the same.
    let file = Bytes::from(encode(VideoCodec::Mpeg2).concat());
    assert_eq!(
        crate::streaming::demux_streaming_shared(file)
            .unwrap()
            .header()
            .codec,
        "mpeg2"
    );
}
