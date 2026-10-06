//! `rivet transcode` with a live end: an `ndi://` input recorded (or
//! relayed), or a file played out to `ndi://`. The flags are transcode's own
//! and build the same settings; only the run differs.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use rivet::live::{LiveUri, is_live_uri};
use rivet::{JobOutput, LiveTarget, OutputMode, OutputSpec, RungArtifact};

use super::transcode::TranscodeArgs;

pub(crate) fn run(args: TranscodeArgs, live_in: bool) -> Result<()> {
    let settings = args.settings()?;
    let stop = super::ctrl_c_flag();
    let input = args.input.to_string_lossy().into_owned();

    if live_in {
        let wait = Duration::from_secs_f64(settings.live.start_timeout.max(0.0));
        let uri = LiveUri::parse(&input)?;
        eprintln!("waiting for {uri}…");
        let source = rivet::live::open_source(&input, wait)?;
        let (info, source) = rivet::live::probe_source(source, wait)?;
        let spec = settings
            .into_spec_for(&info)
            .context("building output spec")?;
        let target = target_for(args.output.as_deref(), &spec, &uri.file_stem())?;
        announce(&source_label(&info, &uri.to_string()), &target, &spec);
        let sink = Arc::new(super::progress::ProgressPrinter::new(spec.rungs.len()));
        let out = rivet::run_live_job_blocking(source, &spec, target, sink, Some(stop))
            .with_context(|| format!("recording {uri}"))?;
        print_summary(&uri.to_string(), &out, &spec);
        return Ok(());
    }

    // A file played out live.
    let bytes = bytes::Bytes::from(
        std::fs::read(&args.input)
            .with_context(|| format!("reading input {}", args.input.display()))?,
    );
    let probed = rivet::probe_bytes_shared(bytes.clone()).context("probing input")?;
    let spec = settings
        .into_spec_for(&probed)
        .context("building output spec")?;
    let stem = args
        .input
        .file_stem()
        .map_or("rivet".into(), |s| s.to_string_lossy().into_owned());
    let source = rivet::live::FileSource::new(stem, bytes, spec.live.repeat)?;
    let target = target_for(args.output.as_deref(), &spec, "")?;
    announce(&args.input.display().to_string(), &target, &spec);
    let sink = Arc::new(super::progress::ProgressPrinter::new(spec.rungs.len()));
    let out = rivet::run_live_job_blocking(source, &spec, target, sink, Some(stop))
        .with_context(|| format!("sending {}", args.input.display()))?;
    print_summary(&args.input.display().to_string(), &out, &spec);
    Ok(())
}

/// Where a live job writes: an `ndi://` output, else the file or directory
/// the spec's shape needs, named after the source when `-o` is not given.
pub(crate) fn target_for(
    output: Option<&Path>,
    spec: &OutputSpec,
    stem: &str,
) -> Result<LiveTarget> {
    let dir_shaped = matches!(spec.mode, OutputMode::Hls { .. }) || spec.rungs.len() > 1;
    match output {
        Some(o) if is_live_uri(&o.to_string_lossy()) => {
            match LiveUri::parse(&o.to_string_lossy())? {
                LiveUri::Ndi(endpoint) => Ok(LiveTarget::Ndi(endpoint)),
            }
        }
        Some(o) if dir_shaped => Ok(LiveTarget::Dir(o.to_path_buf())),
        Some(o) => Ok(LiveTarget::File(o.to_path_buf())),
        None => {
            let stem = if stem.is_empty() { "live" } else { stem };
            Ok(match spec.mode {
                OutputMode::Hls { .. } => LiveTarget::Dir(PathBuf::from(format!("{stem}.hls"))),
                _ if dir_shaped => LiveTarget::Dir(PathBuf::from(format!(
                    "{stem}.{}",
                    spec.video_codec.codec().label()
                ))),
                _ => LiveTarget::File(PathBuf::from(format!(
                    "{stem}.{}.{}",
                    spec.video_codec.codec().label(),
                    spec.file_extension()
                ))),
            })
        }
    }
}

fn source_label(info: &rivet::MediaInfo, uri: &str) -> String {
    format!(
        "{uri} ({}x{} @ {:.3} fps{})",
        info.width,
        info.height,
        info.frame_rate,
        info.audio.as_ref().map_or(String::new(), |a| format!(
            ", audio {}ch {} Hz",
            a.channels, a.sample_rate
        ))
    )
}

fn announce(source: &str, target: &LiveTarget, spec: &OutputSpec) {
    let to = match target {
        LiveTarget::File(p) | LiveTarget::Dir(p) => p.display().to_string(),
        LiveTarget::Ndi(e) => format!("ndi://{}", e.name),
    };
    let until = match spec.live.duration {
        Some(d) => format!("for {d:.0} s"),
        None => "until the source ends".into(),
    };
    eprintln!("{source} → {to}, {until} (Ctrl+C to stop)");
}

fn print_summary(source: &str, out: &JobOutput, spec: &OutputSpec) {
    println!(
        "{source} ({}x{} @ {:.3} fps)",
        out.source_dims.0, out.source_dims.1, out.source_frame_rate
    );
    match &out.audio_codecs {
        Some(codecs) => println!("  audio: {} [codecs={codecs}]", out.audio_handling),
        None => println!("  audio: {}", out.audio_handling),
    }
    for r in &out.rungs {
        let where_ = match &r.artifact {
            RungArtifact::Written(p) => p.display().to_string(),
            RungArtifact::HlsRendition { relative_dir, .. } => relative_dir.clone(),
            RungArtifact::Ndi { source } => format!("ndi://{source}"),
            RungArtifact::File(_) => spec.file_extension().to_string(),
        };
        println!(
            "  {:<6} {}x{}  {} frames  {:.2} MiB  [{}]",
            r.label,
            r.width,
            r.height,
            r.frames,
            r.bytes as f64 / (1024.0 * 1024.0),
            where_
        );
    }
    if let Some(master) = &out.master_playlist {
        println!("  master playlist: {}", master.display());
    }
    if let Some(live) = &out.live {
        println!(
            "  {:.1} s at {}/{} fps: {} repeated, {} dropped ({} while the encoders were behind) — {}",
            live.seconds(),
            live.frame_rate.0,
            live.frame_rate.1,
            live.repeated,
            live.dropped_early + live.dropped_behind,
            live.dropped_behind,
            live.ended.as_str()
        );
        if live.dropped_behind > 0 {
            eprintln!(
                "note: {} pictures arrived while the encoders were behind and their frames were \
                 repeated; a GPU encoder (--features nvidia|amd|qsv), --codec h264 or a faster \
                 --video-speed keeps up with a live source",
                live.dropped_behind
            );
        }
    }
    println!("  done in {:.2}s", out.elapsed.as_secs_f64());
}
