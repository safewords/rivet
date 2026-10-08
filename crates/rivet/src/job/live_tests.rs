//! The live path on synthetic sources: the same settings a user's flags
//! make, run as a live job. VP9 throughout — rivet's own encoder, in every
//! build — so nothing here needs a GPU or an optional feature.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use codec::frame::{ColorMetadata, ColorSpace, PixelFormat, VideoFrame};

use super::{JobOutput, LiveEnd, LiveTarget, RungArtifact, run_live_job_blocking};
use crate::TranscodeSettings;
use crate::live::{FileSource, LiveAudio, LiveEvent, LiveSource, LiveVideo, TICKS_PER_SECOND};
use crate::spec::OutputSpec;

/// A finite source on a 25 fps clock: `frames` pictures of 4:2:2 grey (one
/// missing at `skip`), and 48 kHz stereo audio in 40 ms runs starting from
/// the fourth picture.
struct Synthetic {
    next: u64,
    frames: u64,
    skip: u64,
    audio_next: bool,
}

impl Synthetic {
    fn new(frames: u64, skip: u64) -> Self {
        Self {
            next: 0,
            frames,
            skip,
            audio_next: false,
        }
    }
}

impl LiveSource for Synthetic {
    fn next_event(&mut self, _timeout: Duration) -> Result<Option<LiveEvent>> {
        let tick = TICKS_PER_SECOND / 25;
        if self.next >= self.frames {
            return Ok(Some(LiveEvent::End));
        }
        let i = self.next;
        if self.audio_next {
            self.audio_next = false;
            self.next += 1;
            if i < 3 {
                return Ok(None);
            }
            let samples: Vec<f32> = (0..1920 * 2)
                .map(|n| ((n / 2) as f32 * 0.05).sin() * 0.25)
                .collect();
            return Ok(Some(LiveEvent::Audio(LiveAudio {
                samples,
                sample_rate: 48_000,
                channels: 2,
                time: 1_000 + i as i64 * tick,
            })));
        }
        self.audio_next = true;
        if i == self.skip {
            return Ok(None);
        }
        let (w, h) = (64u32, 48u32);
        let mut data = vec![100u8; (w * h) as usize];
        data.extend(vec![128u8; (w * h) as usize]);
        Ok(Some(LiveEvent::Video(LiveVideo {
            frame: VideoFrame::new(
                bytes::Bytes::from(data),
                w,
                h,
                PixelFormat::Yuv422p,
                ColorSpace::Bt601,
                0,
            ),
            color: ColorMetadata {
                matrix_coefficients: 6,
                colour_primaries: 6,
                ..ColorMetadata::default()
            },
            frame_rate: (25, 1),
            time: 1_000 + i as i64 * tick,
        })))
    }

    fn name(&self) -> String {
        "SYNTH (test)".into()
    }
}

/// The spec `settings` (as `key=value` pairs, the IPC / manifest
/// vocabulary) make for a 64x48 source.
fn spec(kv: &[(&str, &str)]) -> OutputSpec {
    let mut s = TranscodeSettings::default();
    for (k, v) in kv {
        s.apply_kv(k, v)
            .unwrap_or_else(|e| panic!("{k}={v}: {e:#}"));
    }
    s.into_spec(64, 48).expect("spec")
}

fn run<S: LiveSource + 'static>(source: S, spec: &OutputSpec, target: LiveTarget) -> JobOutput {
    run_live_job_blocking(source, spec, target, Arc::new(crate::fn_sink(|_| {})), None)
        .expect("the live job")
}

fn written(out: &JobOutput, i: usize) -> std::path::PathBuf {
    match &out.rungs[i].artifact {
        RungArtifact::Written(p) => p.clone(),
        other => panic!("expected a written file, got {other:?}"),
    }
}

/// The file is as long as the time recorded: the skipped picture is
/// repeated, not lost, and the audio that started late is padded at the
/// front, so it stays in step.
#[test]
fn a_live_single_file_fills_its_gaps_and_keeps_the_audio_in_step() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rec.mp4");
    let spec = spec(&[
        ("codec", "vp9"),
        ("container", "mp4"),
        ("idle-timeout", "0"),
    ]);
    let out = run(
        Synthetic::new(50, 20),
        &spec,
        LiveTarget::File(path.clone()),
    );
    let live = out.live.as_ref().expect("live stats");
    assert_eq!(live.ended, LiveEnd::SourceEnded);
    assert_eq!((live.frames, live.repeated), (50, 1), "{live:?}");
    assert_eq!(written(&out, 0), path);
    assert_eq!(out.audio_codecs.as_deref(), Some("opus"));

    let info = crate::probe_file(&path).expect("probe");
    assert_eq!(info.video_codec, "vp9");
    assert!((info.duration - 2.0).abs() < 0.05, "{}", info.duration);
    let audio = info.audio.expect("an audio track");
    assert_eq!((audio.codec.as_str(), audio.channels), ("opus", 2));
}

/// `duration` ends the job at exactly its frame count, and a ladder writes
/// a file per rung, each the size it was fitted to.
#[test]
fn duration_ends_it_and_every_rung_gets_its_own_file() {
    let dir = tempfile::tempdir().unwrap();
    let spec = spec(&[
        ("codec", "vp9"),
        ("rungs", "64x48,32x24"),
        ("duration", "400ms"),
        ("audio", "drop"),
    ]);
    let out = run(
        Synthetic::new(1000, u64::MAX),
        &spec,
        LiveTarget::Dir(dir.path().to_path_buf()),
    );
    let live = out.live.as_ref().unwrap();
    assert_eq!((live.frames, live.ended), (10, LiveEnd::Duration));
    assert_eq!(out.rungs.len(), 2);
    for (i, (w, h)) in [(64, 48), (32, 24)].into_iter().enumerate() {
        let file = written(&out, i);
        assert_eq!(file.extension().and_then(|e| e.to_str()), Some("webm"));
        let info = crate::probe_file(&file).expect("probe");
        assert_eq!((info.width, info.height), (w, h));
        assert!(info.audio.is_none(), "audio=drop");
    }
}

/// An HLS package written live ends as the file path writes one: a master
/// playlist, a VOD playlist per rendition closed with ENDLIST, and the audio
/// rendition beside the video.
#[test]
fn a_live_hls_package_ends_as_a_finished_package() {
    let dir = tempfile::tempdir().unwrap();
    let spec = spec(&[("mode", "hls"), ("codec", "vp9"), ("segment-seconds", "1")]);
    let out = run(
        Synthetic::new(75, u64::MAX),
        &spec,
        LiveTarget::Dir(dir.path().to_path_buf()),
    );
    let master = std::fs::read_to_string(out.master_playlist.as_ref().expect("master")).unwrap();
    assert!(
        master.contains("video/64p/playlist.m3u8") || master.contains("video/"),
        "{master}"
    );
    assert!(master.contains("TYPE=AUDIO"), "{master}");
    let video = std::fs::read_to_string(
        dir.path()
            .join("video")
            .join(&out.rungs[0].label)
            .join("playlist.m3u8"),
    )
    .unwrap();
    assert!(
        video.contains("#EXT-X-PLAYLIST-TYPE:VOD") && video.contains("#EXT-X-ENDLIST"),
        "{video}"
    );
    assert_eq!(video.matches("#EXTINF").count(), 3, "{video}");
    let audio = std::fs::read_to_string(dir.path().join("audio").join("audio.m3u8")).unwrap();
    assert!(audio.contains("#EXT-X-ENDLIST"), "{audio}");
}

/// A file read through the live path (what a file played out to NDI is)
/// comes out whole, its sound with it; with `loop` it starts again at its
/// end until `duration`.
#[test]
fn a_file_runs_through_the_live_path_and_loops() {
    let pics =
        (0..25u64).map(|t| crate::synth::test_pattern(64, 48, t, h26x::ChromaFormat::Yuv420));
    let coded = crate::synth::encode_h264(&crate::synth::H264::new(64, 48, 25), pics);
    let tone = crate::synth::aac_sine(440.0, 1.0, 2, 128_000);
    let file = bytes::Bytes::from(crate::synth::mp4(&coded, 64, 48, 25, Some(&tone), None));

    let dir = tempfile::tempdir().unwrap();
    let once = dir.path().join("once.mp4");
    let spec_once = spec(&[("codec", "vp9"), ("container", "mp4")]);
    let source = FileSource::new("clip", file.clone(), false).unwrap();
    let out = run(source, &spec_once, LiveTarget::File(once.clone()));
    let live = out.live.as_ref().unwrap();
    assert_eq!(
        (live.frames, live.ended),
        (25, LiveEnd::SourceEnded),
        "{live:?}"
    );
    assert_eq!((live.repeated, live.dropped_early), (0, 0));
    let info = crate::probe_file(&once).unwrap();
    assert!(info.audio.is_some());

    let looped = dir.path().join("looped.mp4");
    let spec_loop = spec(&[
        ("codec", "vp9"),
        ("container", "mp4"),
        ("loop", "true"),
        ("duration", "2.4s"),
    ]);
    let source = FileSource::new("clip", file, spec_loop.live.repeat).unwrap();
    let out = run(source, &spec_loop, LiveTarget::File(looped.clone()));
    let live = out.live.as_ref().unwrap();
    assert_eq!(
        (live.frames, live.ended),
        (60, LiveEnd::Duration),
        "{live:?}"
    );
    let info = crate::probe_file(&looped).unwrap();
    assert!((info.duration - 2.4).abs() < 0.05, "{}", info.duration);
}

/// What a live job cannot do is refused by name, before anything runs; and
/// a file job refuses the live settings.
#[test]
fn what_a_live_job_cannot_do_is_refused_by_name() {
    let err = |kv: &[(&str, &str)], target: LiveTarget| {
        let spec = spec(kv);
        format!(
            "{:#}",
            run_live_job_blocking(
                Synthetic::new(5, u64::MAX),
                &spec,
                target,
                Arc::new(crate::fn_sink(|_| {})),
                None,
            )
            .unwrap_err()
        )
    };
    let file = LiveTarget::File(std::env::temp_dir().join("rivet-never-written.mp4"));
    let mut trimmed = TranscodeSettings {
        trim_start: Some(1.0),
        ..Default::default()
    };
    trimmed.apply_kv("codec", "vp9").unwrap();
    let trimmed = trimmed.into_spec(64, 48).unwrap();
    let e = format!("{:#}", super::check_live_spec(&trimmed, &file).unwrap_err());
    assert!(e.contains("duration"), "{e}");
    let e = err(&[("codec", "vp9"), ("decode", "gpu:0")], file.clone());
    assert!(e.contains("arrives decoded"), "{e}");
    let e = err(&[("codec", "vp9"), ("rungs", "64x48,32x24")], file);
    assert!(e.contains("give a directory"), "{e}");

    // A file job with a live setting.
    let mut s = TranscodeSettings::default();
    s.apply_kv("duration", "10s").unwrap();
    let spec = s.into_spec(64, 48).unwrap();
    let pics = (0..2u64).map(|t| crate::synth::test_pattern(64, 48, t, h26x::ChromaFormat::Yuv420));
    let coded = crate::synth::encode_h264(&crate::synth::H264::new(64, 48, 25), pics);
    let mp4 = crate::synth::mp4(&coded, 64, 48, 25, None, None);
    let e = format!(
        "{:#}",
        crate::run_job_blocking(&mp4, &spec, None, Arc::new(crate::fn_sink(|_| {}))).unwrap_err()
    );
    assert!(e.contains("are a live job's"), "{e}");
}
