//! Video tracks in containers rivet did not map before: H.263 in a 3GP
//! (`s263`, 3GPP TS 26.244) and VP8 in an AVI (`VP80`). Each fixture is
//! written here from the container's specification around pictures from
//! rivet's own encoders, then demuxed and decoded.

use codec::encode::{Encoder, EncoderConfig, QualityTarget, SpeedTier};
use container::streaming::demux_streaming;
use frame::{ColorMetadata, ColorSpace, PixelFormat, VideoCodec, VideoFrame};

const W: u32 = 176;
const H: u32 = 144;
const FRAMES: usize = 5;

fn luma(n: usize) -> impl Fn(usize, usize) -> u8 {
    move |x, y| ((x * 2 + y + n * 11) % 190 + 30) as u8
}

/// An ISO BMFF box.
fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut b = ((8 + body.len()) as u32).to_be_bytes().to_vec();
    b.extend(kind);
    b.extend(body);
    b
}

/// A FullBox: version 0, no flags.
fn full(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut b = vec![0, 0, 0, 0];
    b.extend(body);
    bx(kind, &b)
}

/// A 3GP file (ISO/IEC 14496-12 boxes, 3GPP TS 26.244's `s263` entry) with
/// one video track of `pictures`, one per sample, at 15 a second.
fn three_gp(pictures: &[Vec<u8>]) -> Vec<u8> {
    let ftyp = bx(b"ftyp", b"3gp4\0\0\x02\x003gp4isom");
    let timescale = 15_000u32;
    let delta = 1_000u32;
    let duration = delta * pictures.len() as u32;
    let matrix: [u32; 9] = [0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x4000_0000];
    let m: Vec<u8> = matrix.iter().flat_map(|v| v.to_be_bytes()).collect();

    let mut mvhd = Vec::new();
    mvhd.extend([0u8; 8]); // creation / modification
    mvhd.extend(timescale.to_be_bytes());
    mvhd.extend(duration.to_be_bytes());
    mvhd.extend(0x10000u32.to_be_bytes()); // rate
    mvhd.extend(0x100u16.to_be_bytes()); // volume
    mvhd.extend([0u8; 10]);
    mvhd.extend(&m);
    mvhd.extend([0u8; 24]); // pre_defined
    mvhd.extend(2u32.to_be_bytes()); // next_track_ID

    let mut tkhd = vec![0, 0, 0, 3]; // enabled, in movie
    tkhd.extend([0u8; 8]);
    tkhd.extend(1u32.to_be_bytes()); // track_ID
    tkhd.extend([0u8; 4]);
    tkhd.extend(duration.to_be_bytes());
    tkhd.extend([0u8; 8]);
    tkhd.extend([0u8; 8]); // layer, alternate_group, volume, reserved
    tkhd.extend(&m);
    tkhd.extend((W << 16).to_be_bytes());
    tkhd.extend((H << 16).to_be_bytes());
    let tkhd = bx(b"tkhd", &tkhd);

    let mut mdhd = Vec::new();
    mdhd.extend([0u8; 8]);
    mdhd.extend(timescale.to_be_bytes());
    mdhd.extend(duration.to_be_bytes());
    mdhd.extend(0x55c4u16.to_be_bytes()); // 'und'
    mdhd.extend([0u8; 2]);
    let mut hdlr = vec![0u8; 4];
    hdlr.extend(b"vide");
    hdlr.extend([0u8; 12]);
    hdlr.extend(b"VideoHandler\0");

    // VisualSampleEntry (§12.1.3): six reserved bytes, the data reference
    // index, sixteen bytes pre-defined / reserved, the size, 72 dpi twice,
    // reserved, frame_count 1, a 32-byte compressor name, depth 0x18,
    // pre_defined -1; then `d263` (vendor, decoder_version, level, profile).
    let mut s263 = vec![0u8; 6];
    s263.extend(1u16.to_be_bytes());
    s263.extend([0u8; 16]);
    s263.extend((W as u16).to_be_bytes());
    s263.extend((H as u16).to_be_bytes());
    s263.extend(0x0048_0000u32.to_be_bytes());
    s263.extend(0x0048_0000u32.to_be_bytes());
    s263.extend([0u8; 4]);
    s263.extend(1u16.to_be_bytes());
    s263.extend([0u8; 32]);
    s263.extend(0x18u16.to_be_bytes());
    s263.extend((-1i16).to_be_bytes());
    s263.extend(bx(b"d263", b"rivt\x00\x0a\x00"));
    let mut stsd = 1u32.to_be_bytes().to_vec();
    stsd.extend(bx(b"s263", &s263));

    let mut stts = 1u32.to_be_bytes().to_vec();
    stts.extend((pictures.len() as u32).to_be_bytes());
    stts.extend(delta.to_be_bytes());
    let mut stsc = 1u32.to_be_bytes().to_vec();
    stsc.extend(1u32.to_be_bytes());
    stsc.extend((pictures.len() as u32).to_be_bytes());
    stsc.extend(1u32.to_be_bytes());
    let mut stsz = 0u32.to_be_bytes().to_vec();
    stsz.extend((pictures.len() as u32).to_be_bytes());
    for p in pictures {
        stsz.extend((p.len() as u32).to_be_bytes());
    }
    let build = |chunk_offset: u32| {
        let mut stco = 1u32.to_be_bytes().to_vec();
        stco.extend(chunk_offset.to_be_bytes());
        let stbl = [
            full(b"stsd", &stsd),
            full(b"stts", &stts),
            full(b"stsc", &stsc),
            full(b"stsz", &stsz),
            full(b"stco", &stco),
        ]
        .concat();
        let dref = full(b"dref", &[1u32.to_be_bytes().to_vec(), bx(b"url ", &[0, 0, 0, 1])].concat());
        let minf = [full(b"vmhd", &[0u8; 8]), bx(b"dinf", &dref), bx(b"stbl", &stbl)].concat();
        let mdia = [full(b"mdhd", &mdhd), full(b"hdlr", &hdlr), bx(b"minf", &minf)].concat();
        let trak = [tkhd.clone(), bx(b"mdia", &mdia)].concat();
        bx(b"moov", &[full(b"mvhd", &mvhd), bx(b"trak", &trak)].concat())
    };
    let moov_len = build(0).len();
    let mdat_body: Vec<u8> = pictures.concat();
    let offset = (ftyp.len() + moov_len + 8) as u32;
    [ftyp, build(offset), bx(b"mdat", &mdat_body)].concat()
}

#[test]
fn h263_in_3gp_decodes_through_the_short_header_path() {
    let mut cfg = mpeg4::EncoderConfig::new(W, H, 15);
    cfg.short_header = true;
    let mut enc = mpeg4::Encoder::new(cfg).expect("short-header encoder");
    let mut pictures = Vec::new();
    for n in 0..FRAMES {
        let mut f = mpeg4::Frame::new(W, H);
        let paint = luma(n);
        for y in 0..H as usize {
            for x in 0..W as usize {
                f.data[y * W as usize + x] = paint(x, y);
            }
        }
        for s in &mut f.data[(W * H) as usize..] {
            *s = 128;
        }
        pictures.push(enc.encode(&f).expect("encode"));
    }
    let file = three_gp(&pictures);

    let mut d = demux_streaming(&file).expect("demux the 3GP");
    let header = d.header().clone();
    assert_eq!(header.codec, "h263", "s263 is H.263");
    assert_eq!((header.info.width, header.info.height), (W, H));
    assert!((header.info.frame_rate - 15.0).abs() < 0.01, "{}", header.info.frame_rate);
    let mut dec = codec::decode::create_decoder(&header.codec, header.info.clone()).expect("a decoder for h263");
    let mut decoded = Vec::new();
    while let Some(s) = d.next_video_sample().unwrap() {
        dec.push_sample(&s.data).expect("decode");
        while let Some(f) = dec.decode_next().unwrap() {
            decoded.push(f);
        }
    }
    dec.finish().unwrap();
    while let Some(f) = dec.decode_next().unwrap() {
        decoded.push(f);
    }
    assert_eq!(decoded.len(), FRAMES);
    // The pictures are the gradients coded: the first, an intra picture,
    // within a few levels of its source.
    let paint = luma(0);
    let err: f64 = (0..(W * H) as usize)
        .map(|i| (f64::from(decoded[0].data[i]) - f64::from(paint(i % W as usize, i / W as usize))).abs())
        .sum::<f64>()
        / f64::from(W * H);
    assert!(err < 4.0, "mean luma error {err}");
}

/// A RIFF chunk, word-aligned.
fn chunk(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = fourcc.to_vec();
    out.extend((payload.len() as u32).to_le_bytes());
    out.extend(payload);
    if out.len() % 2 == 1 {
        out.push(0);
    }
    out
}

fn list(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    chunk(b"LIST", &[kind.to_vec(), body.to_vec()].concat())
}

#[test]
fn vp8_in_avi_is_read_as_vp8() {
    let cfg = EncoderConfig {
        width: W,
        height: H,
        frame_rate: 25.0,
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
        codec: VideoCodec::Vp8,
        constant_qp: false,
        overrides: Default::default(),
    };
    let mut enc = codec::encode::vp8_sw::Vp8Encoder::new(cfg).expect("vp8 encoder");
    for n in 0..FRAMES {
        let paint = luma(n);
        let mut data: Vec<u8> = (0..(W * H) as usize).map(|i| paint(i % W as usize, i / W as usize)).collect();
        data.extend(std::iter::repeat_n(128u8, (W * H / 2) as usize));
        enc.send_frame(&VideoFrame::new(data.into(), W, H, PixelFormat::Yuv420p, ColorSpace::Bt709, n as u64))
            .unwrap();
    }
    enc.flush().unwrap();
    let mut frames = Vec::new();
    while let Some(p) = enc.receive_packet().unwrap() {
        frames.push(p.data.to_vec());
    }

    // AVI (Microsoft's RIFF reference): `avih`, one `strl` with a `vids`
    // `strh` (handler `VP80`, scale 1 / rate 25) and a BITMAPINFOHEADER
    // `strf` (compression `VP80`), then the frames as `00dc` chunks.
    let mut avih = vec![0u8; 56];
    avih[16..20].copy_from_slice(&(frames.len() as u32).to_le_bytes()); // dwTotalFrames
    let mut strh = b"vidsVP80".to_vec();
    strh.extend([0u8; 12]);
    strh.extend(1u32.to_le_bytes());
    strh.extend(25u32.to_le_bytes());
    strh.extend([0u8; 24]);
    let mut strf = 40u32.to_le_bytes().to_vec();
    strf.extend((W as i32).to_le_bytes());
    strf.extend((H as i32).to_le_bytes());
    strf.extend(1u16.to_le_bytes());
    strf.extend(24u16.to_le_bytes());
    strf.extend(b"VP80");
    strf.extend([0u8; 20]);
    let hdrl = list(b"hdrl", &[chunk(b"avih", &avih), list(b"strl", &[chunk(b"strh", &strh), chunk(b"strf", &strf)].concat())].concat());
    let movi = list(b"movi", &frames.iter().flat_map(|f| chunk(b"00dc", f)).collect::<Vec<u8>>());
    let body = [b"AVI ".to_vec(), hdrl, movi].concat();
    let file = [b"RIFF".to_vec(), (body.len() as u32).to_le_bytes().to_vec(), body].concat();

    let mut d = demux_streaming(&file).expect("demux the AVI");
    assert_eq!(d.header().codec, "vp8");
    assert_eq!((d.header().info.width, d.header().info.height), (W, H));
    let mut dec = codec::decode::create_decoder("vp8", d.header().info.clone()).unwrap();
    let mut n = 0;
    while let Some(s) = d.next_video_sample().unwrap() {
        dec.push_sample(&s.data).unwrap();
        while let Some(f) = dec.decode_next().unwrap() {
            assert_eq!((f.width, f.height), (W, H));
            n += 1;
        }
    }
    dec.finish().unwrap();
    while dec.decode_next().unwrap().is_some() {
        n += 1;
    }
    assert_eq!(n, FRAMES);
}
