//! Odd frame sizes through every software encoder whose codec carries them
//! (`codec::encode::codes_odd_sizes`) and back through rivet's own decoder
//! for it: a 351x241 picture (4:2:0 with the rounded-up `176x121` chroma
//! planes the pipeline lays an odd frame out with) comes back 351x241, every
//! column and row in place. H.264 and H.265 are not here: their cropping
//! counts in chroma samples, so an odd 4:2:0 size is not theirs to code, and
//! the pipeline evens their outputs by a crop (`rivet::fit`).

use codec::decode::create_decoder;
use codec::encode::{Encoder, EncoderConfig, QualityTarget, SpeedTier};
use codec::frame::{ColorMetadata, ColorSpace, PixelFormat, ProresProfile, StreamInfo, VideoCodec, VideoFrame};

const W: u32 = 351;
const H: u32 = 241;
const FRAMES: u64 = 4;

/// A picture whose last column and last row differ from their neighbours,
/// so losing or shifting them shows: diagonal ramps in all three planes.
fn picture(pts: u64) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut data = Vec::with_capacity(w * h + 2 * cw * ch);
    for y in 0..h {
        for x in 0..w {
            data.push((32 + (x * 160 / w) + (y * 40 / h) + pts as usize) as u8);
        }
    }
    for plane in 0..2 {
        for y in 0..ch {
            for x in 0..cw {
                data.push((100 + plane * 30 + (x * 40 / cw) + (y * 20 / ch)) as u8);
            }
        }
    }
    VideoFrame::new(data.into(), W, H, PixelFormat::Yuv420p, ColorSpace::Bt709, pts)
}

fn config(codec: VideoCodec) -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        frame_rate: 25.0,
        keyframe_interval: 2,
        target: QualityTarget::Standard,
        tier: SpeedTier::Draft,
        threads: 1,
        pixel_format: PixelFormat::Yuv420p,
        color_metadata: ColorMetadata::default(),
        codec,
        ..EncoderConfig::default()
    }
}

fn encoder(codec: VideoCodec) -> Box<dyn Encoder> {
    let cfg = config(codec);
    match codec {
        VideoCodec::Av1 => Box::new(codec::encode::av1_sw::Av1Encoder::new(cfg).unwrap()),
        VideoCodec::H264 | VideoCodec::H265 => Box::new(codec::encode::h26x_sw::H26xEncoder::new(cfg).unwrap()),
        VideoCodec::Vp8 => Box::new(codec::encode::vp8_sw::Vp8Encoder::new(cfg).unwrap()),
        VideoCodec::Vp9 => Box::new(codec::encode::vp9_sw::Vp9Encoder::new(cfg).unwrap()),
        VideoCodec::Mpeg2 => Box::new(codec::encode::mpeg2_sw::Mpeg2Encoder::new(cfg).unwrap()),
        VideoCodec::Mpeg4 => Box::new(codec::encode::mpeg4_sw::Mpeg4Encoder::new(cfg).unwrap()),
        VideoCodec::ProRes(_) => Box::new(codec::encode::prores_sw::ProresEncoder::new(cfg).unwrap()),
    }
}

/// PSNR of plane `p` (0 luma, 1 and 2 chroma) of `b` against `a`, both
/// odd-sized 8-bit 4:2:0 with rounded-up chroma.
fn psnr(a: &VideoFrame, b: &VideoFrame, p: usize) -> f64 {
    let (w, h) = (W as usize, H as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let (off, len) = match p {
        0 => (0, w * h),
        1 => (w * h, cw * ch),
        _ => (w * h + cw * ch, cw * ch),
    };
    let se: f64 =
        a.data[off..off + len].iter().zip(&b.data[off..off + len]).map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)).sum();
    10.0 * (255.0f64.powi(2) / (se / len as f64).max(1e-9)).log10()
}

#[test]
fn every_software_encoder_codes_an_odd_size_as_it_is() {
    for codec in [
        VideoCodec::Av1,
        VideoCodec::Vp8,
        VideoCodec::Vp9,
        VideoCodec::Mpeg2,
        VideoCodec::Mpeg4,
        VideoCodec::ProRes(ProresProfile::Standard),
        VideoCodec::ProRes(ProresProfile::P4444),
    ] {
        let mut enc = encoder(codec);
        let mut packets = Vec::new();
        let source: Vec<VideoFrame> = (0..FRAMES).map(picture).collect();
        for f in &source {
            enc.send_frame(f).unwrap();
            while let Some(p) = enc.receive_packet().unwrap() {
                packets.push(p);
            }
        }
        enc.flush().unwrap();
        while let Some(p) = enc.receive_packet().unwrap() {
            packets.push(p);
        }
        let label = match codec {
            VideoCodec::ProRes(_) => "prores".to_string(),
            c => c.label().to_string(),
        };
        let info = StreamInfo {
            codec: label.clone(),
            width: W,
            height: H,
            frame_rate: 25.0,
            duration: 0.0,
            pixel_format: PixelFormat::Yuv420p,
            color_space: ColorSpace::Bt709,
            total_frames: FRAMES,
            bitrate: 0,
            color_metadata: ColorMetadata::default(),
        };
        let mut dec = create_decoder(&label, info).unwrap();
        let mut out = Vec::new();
        for p in &packets {
            dec.push_sample(&p.data).unwrap();
            while let Some(f) = dec.decode_next().unwrap() {
                out.push(f);
            }
        }
        dec.finish().unwrap();
        while let Some(f) = dec.decode_next().unwrap() {
            out.push(f);
        }
        assert!(!out.is_empty(), "{codec:?}: nothing decoded");
        let f = &out[0];
        eprintln!("{codec:?}: decoded {}x{} {:?}", f.width, f.height, f.format);
        assert_eq!((f.width, f.height), (W, H), "{codec:?}");
        if f.format == PixelFormat::Yuv420p {
            let q = [psnr(&source[0], f, 0), psnr(&source[0], f, 1), psnr(&source[0], f, 2)];
            eprintln!("{codec:?}: PSNR {:.1} / {:.1} / {:.1} dB", q[0], q[1], q[2]);
            assert!(q.iter().all(|&v| v > 30.0), "{codec:?}: {q:?}");
        }
    }
}
