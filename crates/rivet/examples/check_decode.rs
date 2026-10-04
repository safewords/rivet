//! Reads an encoded output back through rivet's own software decoders (AV1,
//! H.264, H.265, VP9) and says what it found, as one JSON line on stdout. For
//! CI jobs that check what a GPU wrote without trusting that GPU to read it
//! back, and that check a GPU's decoder against rivet's.
//!
//! ```text
//! cargo run --release --example check_decode [--features qsv] -- OUTPUT
//!     [--ref SOURCE] [--codec av1|h264|h265|vp9] [--frames N] [--min-psnr DB]
//!     [--size WxH] [--rate BPS [--buffer SECONDS]] [--qsv GPU]...
//! ```
//!
//! `OUTPUT` is an MP4, WebM or MKV file, or an HLS media playlist (`.m3u8`):
//! its init segment and media segments are read in playlist order. The check
//! fails (non-zero exit) when a packet does not decode, when the codec, the
//! frame count or the picture size is not the one given, or when the mean
//! luma PSNR against `--ref` (decoded the same way, frame for frame; the
//! sizes must match) is under `--min-psnr`.
//!
//! `--rate BPS` checks a constant-rate stream: the average within 10% of the
//! rate, and no one-second window over the rate plus the buffer (`--buffer`,
//! default 1 s) with a 5% margin.
//!
//! `--qsv GPU` (repeatable; a build with `--features qsv`) also decodes the
//! stream on that Intel card's QSV decoder (rivet's GPU index), with no
//! software fallback, and requires every frame to be bit-identical to rivet's
//! software decoder's.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use rivet::codec::decode::Decoder;
use rivet::codec::decode::av1_sw::Av1Decoder;
use rivet::codec::decode::h26x_sw::H26xDecoder;
use rivet::codec::decode::vp9_sw::Vp9Decoder;
use rivet::codec::frame::{StreamInfo, VideoFrame};

/// The bytes to demux: the file itself, or a media playlist's init segment
/// followed by its segments.
fn read_output(path: &Path) -> Result<Vec<u8>> {
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("m3u8"))
    {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
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
                bytes.extend(
                    std::fs::read(dir.join(uri))
                        .with_context(|| format!("reading the init segment {uri}"))?,
                );
            } else if !line.is_empty() && !line.starts_with('#') {
                bytes.extend(
                    std::fs::read(dir.join(line))
                        .with_context(|| format!("reading the segment {line}"))?,
                );
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

/// A stream's packets, through the streaming demuxer: the reader the
/// pipeline itself uses, and the one that walks fragmented MP4 (`moof` /
/// `trun`), which HLS segments are.
struct Stream {
    codec: String,
    info: StreamInfo,
    packets: Vec<Vec<u8>>,
}

fn demux(path: &Path) -> Result<Stream> {
    let bytes = read_output(path)?;
    let mut demuxer = rivet::container::streaming::demux_streaming(&bytes)
        .with_context(|| format!("demuxing {}", path.display()))?;
    let header = demuxer.header().clone();
    let mut packets = Vec::new();
    while let Some(sample) = demuxer.next_video_sample()? {
        packets.push(sample.data);
    }
    ensure!(!packets.is_empty(), "{}: no video packets", path.display());
    Ok(Stream {
        codec: canonical_codec(&header.codec),
        info: header.info,
        packets,
    })
}

fn software_decoder(stream: &Stream) -> Result<Box<dyn Decoder>> {
    Ok(match stream.codec.as_str() {
        "h264" | "h265" => Box::new(H26xDecoder::new(stream.info.clone())?),
        "av1" => Box::new(Av1Decoder::new(stream.info.clone())?),
        "vp9" => Box::new(Vp9Decoder::new(stream.info.clone())?),
        other => bail!("no software decoder here for '{other}'"),
    })
}

fn run_decoder(
    mut decoder: Box<dyn Decoder>,
    stream: &Stream,
    what: &str,
) -> Result<Vec<VideoFrame>> {
    let mut frames = Vec::new();
    for (i, packet) in stream.packets.iter().enumerate() {
        decoder
            .push_sample(packet)
            .with_context(|| format!("{what}: packet {i} of {}", stream.packets.len()))?;
        while let Some(f) = decoder.decode_next()? {
            frames.push(f);
        }
    }
    decoder.finish()?;
    while let Some(f) = decoder.decode_next()? {
        frames.push(f);
    }
    ensure!(
        !frames.is_empty(),
        "{what}: no frame decoded from {} packets",
        stream.packets.len()
    );
    Ok(frames)
}

#[cfg(feature = "qsv")]
fn qsv_decoder(stream: &Stream, gpu: u32) -> Result<Box<dyn Decoder>> {
    let vendor_index = rivet::codec::gpu::vendor_index_of(gpu)
        .with_context(|| format!("no GPU {gpu} on this host"))?;
    Ok(Box::new(rivet::codec::decode::qsv_dec::QsvDecoder::new(
        stream.info.clone(),
        vendor_index,
    )?))
}

#[cfg(not(feature = "qsv"))]
fn qsv_decoder(_stream: &Stream, _gpu: u32) -> Result<Box<dyn Decoder>> {
    bail!("--qsv needs a build with --features qsv")
}

/// Average bit rate, and the most bits any `fps`-frame (one-second) window
/// spends.
fn rates(stream: &Stream) -> (f64, f64) {
    let fps = stream.info.frame_rate.round().max(1.0) as usize;
    let sizes: Vec<f64> = stream
        .packets
        .iter()
        .map(|p| p.len() as f64 * 8.0)
        .collect();
    let seconds = sizes.len() as f64 / stream.info.frame_rate.max(1.0);
    let average = sizes.iter().sum::<f64>() / seconds;
    let peak = sizes
        .windows(fps.min(sizes.len()))
        .map(|w| w.iter().sum::<f64>())
        .fold(0.0, f64::max);
    (average, peak)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut output, mut reference, mut codec, mut frames, mut min_psnr, mut size) =
        (None, None, None, None, None, None);
    let (mut rate, mut buffer, mut qsv) = (None, 1.0f64, Vec::new());
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
            "--rate" => rate = Some(value()?.parse::<f64>().context("--rate BPS")?),
            "--buffer" => buffer = value()?.parse::<f64>().context("--buffer SECONDS")?,
            "--qsv" => qsv.push(value()?.parse::<u32>().context("--qsv GPU")?),
            _ if output.is_none() && !a.starts_with("--") => output = Some(a),
            _ => bail!("unexpected argument {a}"),
        }
    }
    let output = output.context("usage: check_decode OUTPUT [options]; see the source")?;

    let stream = demux(Path::new(&output))?;
    let decoded = run_decoder(software_decoder(&stream)?, &stream, &output)?;
    let (w, h) = (decoded[0].width, decoded[0].height);
    ensure!(
        decoded.iter().all(|f| (f.width, f.height) == (w, h)),
        "{output}: the picture size changes mid-stream"
    );

    let mut psnr = None;
    if let Some(reference) = &reference {
        let source_stream = demux(Path::new(reference))?;
        let source = run_decoder(software_decoder(&source_stream)?, &source_stream, reference)?;
        ensure!(
            (source[0].width, source[0].height) == (w, h),
            "{output} is {w}x{h}, the reference {}x{}: PSNR needs equal sizes",
            source[0].width,
            source[0].height
        );
        let n = source.len().min(decoded.len());
        let mut sum = 0.0;
        for (a, b) in source.iter().zip(&decoded).take(n) {
            let score =
                rivet::codec::quality::score_frame(a, b).context("frames of unequal size")?;
            sum += score.psnr.min(99.0);
        }
        psnr = Some(sum / n as f64);
    }

    let (average, peak) = rates(&stream);
    println!(
        "{{\"file\":{:?},\"codec\":{:?},\"frames\":{},\"width\":{w},\"height\":{h},\"psnr\":{},\"bps\":{average:.0},\"peak_1s_bits\":{peak:.0}}}",
        output,
        stream.codec,
        decoded.len(),
        psnr.map_or("null".to_string(), |p| format!("{p:.2}"))
    );

    if let Some(want) = &codec {
        ensure!(
            &stream.codec == want,
            "{output}: carries {}, expected {want}",
            stream.codec
        );
    }
    if let Some(want) = frames {
        ensure!(
            decoded.len() == want,
            "{output}: {} frames decoded, expected {want}",
            decoded.len()
        );
    }
    if let Some((ww, wh)) = size {
        ensure!((w, h) == (ww, wh), "{output}: {w}x{h}, expected {ww}x{wh}");
    }
    if let (Some(min), Some(p)) = (min_psnr, psnr) {
        ensure!(
            p >= min,
            "{output}: mean luma PSNR {p:.2} dB against the reference, under {min} dB"
        );
    }
    if let Some(rate) = rate {
        let achieved = average / rate;
        let bound = rate * (1.0 + buffer);
        ensure!(
            (0.90..=1.10).contains(&achieved),
            "{output}: average {average:.0} bit/s against {rate} ({achieved:.3})"
        );
        ensure!(
            peak <= bound * 1.05,
            "{output}: a one-second window spent {peak:.0} bits, over the rate plus the buffer ({bound:.0})"
        );
    }
    for gpu in qsv {
        let what = format!("{output} on QSV, GPU {gpu}");
        let hw = run_decoder(qsv_decoder(&stream, gpu)?, &stream, &what)?;
        ensure!(
            hw.len() == decoded.len(),
            "{what}: {} frames, rivet's decoder {}",
            hw.len(),
            decoded.len()
        );
        for (i, (a, b)) in decoded.iter().zip(&hw).enumerate() {
            ensure!(
                (a.width, a.height, a.format) == (b.width, b.height, b.format),
                "{what}: frame {i} is {}x{} {:?}, rivet's {}x{} {:?}",
                b.width,
                b.height,
                b.format,
                a.width,
                a.height,
                a.format
            );
            if a.data != b.data {
                let first = a.data.iter().zip(b.data.iter()).position(|(x, y)| x != y);
                bail!(
                    "{what}: frame {i} differs from rivet's decoder (first differing byte {first:?} of {})",
                    a.data.len()
                );
            }
        }
        println!(
            "{{\"file\":{output:?},\"qsv_gpu\":{gpu},\"frames\":{},\"bit_exact\":true}}",
            hw.len()
        );
    }
    Ok(())
}
