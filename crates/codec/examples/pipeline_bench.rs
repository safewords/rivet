//! Per-kernel timings for the shared pipeline's per-pixel and per-sample
//! loops: the scaler, the colour conversions, the tonemap, the chroma
//! layout conversions, the bit-depth narrowing, the temporal denoiser, the
//! audio resampler and the downmix.
//!
//! ```text
//! pipeline_bench [clip.yuv W H] [-r ROUNDS] [-k name,name,..]
//! ```
//!
//! The picture is the clip's first 8-bit 4:2:0 frame when one is given
//! (`bench_corpus --yuv` writes one), else a synthetic 1920x1080 gradient
//! with noise. Each kernel runs `ROUNDS` times (default 15) and prints one
//! `RESULT name median_ms min_ms` line. Run the same binary twice — with
//! and without the kernels' own scalar switches (`RIVET_PIPE_MAX_SIMD=none`,
//! `RIVET_TONEMAP_SCALAR=1`, `RIVET_DENOISE_MAX_SIMD=none`) — for a paired
//! before / after on one machine.

use std::time::Instant;

use bytes::Bytes;
use codec::audio::AudioFrame;
use codec::audio::filter::ChannelLayout;
use codec::audio::remix::Remixer;
use codec::audio::resample::AlignedResampler;
use codec::colorspace;
use codec::filter::{FilterChain, parse_chain};
use codec::frame::{ColorMetadata, TransferFn};
use codec::{ColorSpace, PixelFormat, VideoFrame};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

fn time<F: FnMut()>(rounds: usize, name: &str, mut f: F) {
    f();
    let mut ts: Vec<f64> = (0..rounds)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("RESULT {name:<28} {:>9.3} {:>9.3}", ts[ts.len() / 2], ts[0]);
}

fn main() {
    const USAGE: &str = "usage: pipeline_bench [clip.yuv W H] [-r ROUNDS] [-k name,name,..]";
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return;
    }
    let positional = args.iter().take_while(|a| !a.starts_with('-')).count();
    let mut i = positional;
    while i < args.len() {
        match args[i].as_str() {
            "-r" | "-k" if i + 1 < args.len() => i += 2,
            other => {
                eprintln!("pipeline_bench: unexpected argument {other:?}
{USAGE}");
                std::process::exit(2);
            }
        }
    }
    if positional != 0 && positional != 3 {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    let opt = |n: &str| args.iter().position(|a| a == n).map(|i| args[i + 1].clone());
    let rounds: usize = opt("-r").map_or(15, |s| s.parse().expect("-r N"));
    let only: Option<Vec<String>> = opt("-k").map(|s| s.split(',').map(String::from).collect());
    let want = |n: &str| only.as_ref().is_none_or(|o| o.iter().any(|k| n.starts_with(k.as_str())));
    let (w, h, yuv) = if positional == 3 {
        let (w, h): (usize, usize) = (args[1].parse().unwrap(), args[2].parse().unwrap());
        let mut f = std::fs::read(&args[0]).expect("read the clip");
        f.truncate(w * h * 3 / 2);
        (w, h, f)
    } else {
        let (w, h) = (1920usize, 1080usize);
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        let mut f = vec![0u8; w * h * 3 / 2];
        for (i, v) in f.iter_mut().enumerate() {
            *v = ((i % 251) as u32 + (r.next() & 31)).min(255) as u8;
        }
        (w, h, f)
    };
    let frame8 = VideoFrame::new(Bytes::from(yuv.clone()), w as u32, h as u32, PixelFormat::Yuv420p, ColorSpace::Bt709, 0);
    // 10-bit: the 8-bit picture times four plus a little noise.
    let mut r = Rng(7);
    let s10: Vec<u16> = yuv.iter().map(|&v| ((u16::from(v) << 2) | (r.next() & 3) as u16).min(1023)).collect();
    let b10: Vec<u8> = s10.iter().flat_map(|v| v.to_le_bytes()).collect();
    let frame10 = VideoFrame::new(Bytes::from(b10), w as u32, h as u32, PixelFormat::Yuv420p10le, ColorSpace::Bt2020, 0);
    println!("# picture {w}x{h}, {rounds} rounds; columns: median_ms min_ms");

    if want("scale8") {
        time(rounds, "scale8 1080->720", || {
            colorspace::scale_frame(&frame8, (w * 2 / 3) as u32 & !1, (h * 2 / 3) as u32 & !1).unwrap();
        });
        time(rounds, "scale8 1080->360", || {
            colorspace::scale_frame(&frame8, (w / 3) as u32 & !1, (h / 3) as u32 & !1).unwrap();
        });
    }
    if want("scale10") {
        time(rounds, "scale10 1080->720", || {
            colorspace::scale_frame(&frame10, (w * 2 / 3) as u32 & !1, (h * 2 / 3) as u32 & !1).unwrap();
        });
    }
    if want("bt601") {
        let (mut y, mut u, mut v) = (yuv[..w * h].to_vec(), yuv[w * h..w * h * 5 / 4].to_vec(), yuv[w * h * 5 / 4..].to_vec());
        time(rounds, "bt601->709 8-bit", || colorspace::bt601_to_bt709_planes(&mut y, &mut u, &mut v, w, h));
        let (mut y, mut u, mut v) = (s10[..w * h].to_vec(), s10[w * h..w * h * 5 / 4].to_vec(), s10[w * h * 5 / 4..].to_vec());
        time(rounds, "bt601->709 10-bit", || colorspace::bt601_to_bt709_planes_10bit(&mut y, &mut u, &mut v, w, h));
    }
    if want("tonemap") {
        for (name, tf) in [("tonemap pq", TransferFn::St2084), ("tonemap hlg", TransferFn::AribStdB67)] {
            time(rounds, name, || {
                codec::tonemap::tonemap_yuv420p10le_bt2020_to_yuv420p_bt709(&frame10, tf, None).unwrap();
            });
        }
    }
    if want("sdr2hdr") {
        let src = ColorMetadata::default();
        for (name, tf) in [("sdr->hdr pq", TransferFn::St2084), ("sdr->hdr hlg", TransferFn::AribStdB67)] {
            let conv = colorspace::SdrToHdr::new(&src, tf).unwrap();
            time(rounds, name, || {
                conv.convert(&frame8).unwrap();
            });
        }
    }
    if want("depth") {
        time(rounds, "depth 10->8", || {
            colorspace::convert_bit_depth_frame(&frame10, 8).unwrap();
        });
        time(rounds, "depth 8->10", || {
            colorspace::convert_bit_depth_frame(&frame8, 10).unwrap();
        });
    }
    if want("nv12") {
        let mut nv = yuv[..w * h].to_vec();
        let (u, v) = (&yuv[w * h..w * h * 5 / 4], &yuv[w * h * 5 / 4..]);
        for i in 0..u.len() {
            nv.push(u[i]);
            nv.push(v[i]);
        }
        let f = VideoFrame::new(Bytes::from(nv), w as u32, h as u32, PixelFormat::Nv12, ColorSpace::Bt709, 0);
        time(rounds, "nv12->420", || {
            colorspace::normalize_layout_to_420(&f).unwrap();
        });
    }
    if want("444") {
        let mut p = yuv[..w * h].to_vec();
        p.extend_from_slice(&yuv[..w * h]);
        p.extend_from_slice(&yuv[..w * h]);
        let f = VideoFrame::new(Bytes::from(p), w as u32, h as u32, PixelFormat::Yuv444p, ColorSpace::Bt709, 0);
        time(rounds, "444->420 box", || {
            colorspace::downsample_444_to_420_frame(&f).unwrap();
        });
        let mut p = yuv[..w * h].to_vec();
        p.extend_from_slice(&yuv[..w * h / 2]);
        p.extend_from_slice(&yuv[..w * h / 2]);
        let f = VideoFrame::new(Bytes::from(p), w as u32, h as u32, PixelFormat::Yuv422p, ColorSpace::Bt709, 0);
        time(rounds, "422->420", || {
            colorspace::normalize_layout_to_420(&f).unwrap();
        });
    }
    if want("hqdn3d") {
        let chain = std::sync::Arc::new(FilterChain::prepare(&parse_chain("hqdn3d").unwrap()).unwrap());
        let mut inst = chain.instantiate();
        time(rounds, "hqdn3d", || {
            inst.apply(frame8.clone()).unwrap();
        });
    }
    if want("audio") {
        // One second of audio per round.
        let mut r = Rng(3);
        for (name, from, to, ch) in [
            ("resample 44.1->48 2ch", 44_100u32, 48_000u32, 2u8),
            ("resample 48->44.1 2ch", 48_000, 44_100, 2),
            ("resample 96->48 6ch", 96_000, 48_000, 6),
            ("resample 44.1->48 6ch", 44_100, 48_000, 6),
        ] {
            let samples: Vec<f32> = (0..from as usize * usize::from(ch)).map(|_| (r.next() as f32 / 65536.0) - 0.5).collect();
            let frames: Vec<AudioFrame> = samples
                .chunks(1024 * usize::from(ch))
                .map(|c| AudioFrame { samples: c.to_vec(), sample_rate: from, channels: ch, pts: 0 })
                .collect();
            time(rounds, name, || {
                let mut rs = AlignedResampler::new(from, to, ch).unwrap();
                let mut out = Vec::with_capacity(to as usize * usize::from(ch) + 64);
                for f in &frames {
                    rs.process(f, &mut out).unwrap();
                }
                rs.flush(&mut out).unwrap();
                std::hint::black_box(&out);
            });
        }
        let samples: Vec<f32> = (0..48_000 * 6).map(|_| (r.next() as f32 / 65536.0) - 0.5).collect();
        let frame = AudioFrame { samples, sample_rate: 48_000, channels: 6, pts: 0 };
        let rm = Remixer::new(ChannelLayout::named("5.1"), ChannelLayout::named("stereo"));
        time(rounds, "downmix 5.1->2 1s", || {
            std::hint::black_box(rm.apply(&frame).unwrap());
        });
    }
}
