//! Reads an encoded output back through rivet's own software decoders (AV1,
//! H.264, H.265, VP9 — never a hardware decoder, whatever the build's
//! features) and says what it found, as one JSON line on stdout. For CI jobs
//! that check what a GPU wrote without trusting that GPU to read it back.
//!
//! ```text
//! cargo run --release --example check_decode -- OUTPUT [--ref SOURCE]
//!     [--codec av1|h264|h265|vp9] [--frames N] [--min-psnr DB] [--size WxH]
//! ```
//!
//! `OUTPUT` is an MP4, WebM or MKV file, or an HLS media playlist (`.m3u8`):
//! its init segment and media segments are read in playlist order. The check
//! fails (non-zero exit) when a packet does not decode, when the codec, the
//! frame count or the picture size is not the one given, or when the mean
//! luma PSNR against `--ref` (decoded the same way, frame for frame; the
//! sizes must match) is under `--min-psnr`.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use rivet::codec::decode::Decoder;
use rivet::codec::decode::av1_sw::Av1Decoder;
use rivet::codec::decode::h26x_sw::H26xDecoder;
use rivet::codec::decode::vp9_sw::Vp9Decoder;
use rivet::codec::frame::VideoFrame;

/// The bytes to demux: the file itself, or a media playlist's init segment
/// followed by its segments.
fn read_output(path: &Path) -> Result<Vec<u8>> {
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("m3u8")) {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let dir = path.parent().unwrap_or(Path::new("."));
        let mut bytes = Vec::new();
        let mut segments = 0usize;
        for line in text.lines().map(str::trim) {
            if let Some(map) = line.strip_prefix("#EXT-X-MAP:") {
                let uri = map
                    .split(',')
                    .find_map(|kv| kv.trim().strip_prefix("URI="))
                    .context("EXT-X-MAP without a URI")?
                    .trim_matches('"');
                ensure!(bytes.is_empty(), "a second EXT-X-MAP in {}", path.display());
                bytes.extend(std::fs::read(dir.join(uri)).with_context(|| format!("reading the init segment {uri}"))?);
            } else if !line.is_empty() && !line.starts_with('#') {
                bytes.extend(std::fs::read(dir.join(line)).with_context(|| format!("reading the segment {line}"))?);
                segments += 1;
            }
        }
        ensure!(segments > 0, "{} lists no segments", path.display());
        Ok(bytes)
    } else {
        std::fs::read(path).with_context(|| format!("reading {}", path.display()))
    }
}

fn canonical_codec(codec: &str) -> String {
    match codec.to_ascii_lowercase().as_str() {
        "avc" | "avc1" | "h264" => "h264".into(),
        "hevc" | "h265" | "hvc1" | "hev1" => "h265".into(),
        "av01" | "av1" => "av1".into(),
        "vp09" | "vp9" => "vp9".into(),
        other => other.into(),
    }
}

/// Every frame of `path`, decoded in software, with the codec it carried.
fn decode_all(path: &Path) -> Result<(String, Vec<VideoFrame>)> {
    let bytes = read_output(path)?;
    // The streaming demuxer: the reader the pipeline itself uses, and the one
    // that walks fragmented MP4 (`moof` / `trun`), which HLS segments are.
    let mut demuxer = rivet::container::streaming::demux_streaming(&bytes)
        .with_context(|| format!("demuxing {}", path.display()))?;
    let header = demuxer.header().clone();
    let codec = canonical_codec(&header.codec);
    let mut decoder: Box<dyn Decoder> = match codec.as_str() {
        "h264" | "h265" => Box::new(H26xDecoder::new(header.info.clone())?),
        "av1" => Box::new(Av1Decoder::new(header.info.clone())?),
        "vp9" => Box::new(Vp9Decoder::new(header.info.clone())?),
        other => bail!("{}: no software decoder here for '{other}'", path.display()),
    };
    let mut frames = Vec::new();
    let mut packets = 0usize;
    while let Some(sample) = demuxer.next_video_sample()? {
        decoder
            .push_sample(&sample.data)
            .with_context(|| format!("{}: packet {packets}", path.display()))?;
        packets += 1;
        while let Some(f) = decoder.decode_next()? {
            frames.push(f);
        }
    }
    decoder.finish()?;
    while let Some(f) = decoder.decode_next()? {
        frames.push(f);
    }
    ensure!(!frames.is_empty(), "{}: no frame decoded from {packets} packets", path.display());
    Ok((codec, frames))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut output, mut reference, mut codec, mut frames, mut min_psnr, mut size) = (None, None, None, None, None, None);
    while let Some(a) = args.next() {
        let mut value = || args.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--ref" => reference = Some(value()?),
            "--codec" => codec = Some(canonical_codec(&value()?)),
            "--frames" => frames = Some(value()?.parse::<usize>().context("--frames N")?),
            "--min-psnr" => min_psnr = Some(value()?.parse::<f64>().context("--min-psnr DB")?),
            "--size" => {
                let v = value()?;
                let (w, h) = v.split_once('x').context("--size WxH")?;
                size = Some((w.parse::<u32>()?, h.parse::<u32>()?));
            }
            _ if output.is_none() && !a.starts_with("--") => output = Some(a),
            _ => bail!("unexpected argument {a}"),
        }
    }
    let output = output.context("usage: check_decode OUTPUT [--ref SOURCE] [--codec C] [--frames N] [--min-psnr DB] [--size WxH]")?;

    let (found, decoded) = decode_all(Path::new(&output))?;
    let (w, h) = (decoded[0].width, decoded[0].height);
    ensure!(
        decoded.iter().all(|f| (f.width, f.height) == (w, h)),
        "{output}: the picture size changes mid-stream"
    );

    let mut psnr = None;
    if let Some(reference) = &reference {
        let (_, source) = decode_all(Path::new(reference))?;
        ensure!(
            (source[0].width, source[0].height) == (w, h),
            "{output} is {w}x{h}, the reference {}x{}: PSNR needs equal sizes",
            source[0].width,
            source[0].height
        );
        let n = source.len().min(decoded.len());
        let mut sum = 0.0;
        for (a, b) in source.iter().zip(&decoded).take(n) {
            let score = rivet::codec::quality::score_frame(a, b).context("frames of unequal size")?;
            sum += score.psnr.min(99.0);
        }
        psnr = Some(sum / n as f64);
    }

    println!(
        "{{\"file\":{:?},\"codec\":{:?},\"frames\":{},\"width\":{w},\"height\":{h},\"psnr\":{}}}",
        output,
        found,
        decoded.len(),
        psnr.map_or("null".to_string(), |p| format!("{p:.2}"))
    );

    if let Some(want) = &codec {
        ensure!(&found == want, "{output}: carries {found}, expected {want}");
    }
    if let Some(want) = frames {
        ensure!(decoded.len() == want, "{output}: {} frames decoded, expected {want}", decoded.len());
    }
    if let Some((ww, wh)) = size {
        ensure!((w, h) == (ww, wh), "{output}: {w}x{h}, expected {ww}x{wh}");
    }
    if let (Some(min), Some(p)) = (min_psnr, psnr) {
        ensure!(p >= min, "{output}: mean luma PSNR {p:.2} dB against the reference, under {min} dB");
    }
    Ok(())
}
