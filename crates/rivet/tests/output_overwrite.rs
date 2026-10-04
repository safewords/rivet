//! The `rivet` binary never writes over its input.
//!
//! The defect this pins: `rivet transcode x.mp3 --mode audio` with no `-o`
//! named its output `<stem>.mp3` — the input's own name — and replaced the
//! source with the transcode, exiting 0 (the same for `x.flac --mode audio
//! --audio flac`). The default name now steps aside (`x.rivet.mp3`), and an
//! output that resolves to an input by any spelling — another case on a
//! file system that ignores it, a `..`, a hard link — is refused before any
//! work in every mode, the source left as it was.
//!
//! Every source here is made by this workspace's own encoders; the
//! refusals need no encoder at all (they come before the job).

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn rivet(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rivet")).args(args).output().expect("rivet runs")
}

fn os<S: AsRef<std::ffi::OsStr> + ?Sized>(s: &S) -> &std::ffi::OsStr {
    s.as_ref()
}

/// Half a second of a 1 kHz tone as a bare `.mp3`, from rivet's own encoder.
fn mp3_tone() -> Vec<u8> {
    use rivet::codec::audio::{AudioCodec, AudioEncoderConfig, AudioFrame, create_encoder};
    let mut enc = create_encoder(AudioEncoderConfig::new(AudioCodec::Mp3, 48_000, 1, 64_000)).unwrap();
    let samples =
        (0..24_000).map(|i| 0.4 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48_000.0).sin()).collect();
    let mut frames = enc.encode(&AudioFrame { samples, sample_rate: 48_000, channels: 1, pts: 0 }).unwrap();
    frames.extend(enc.flush().unwrap());
    frames.into_iter().flat_map(|p| p.data).collect()
}

/// A source file in a fresh directory.
fn source(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, bytes).unwrap();
    p
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// Refused, naming the reason, with the source as it was.
fn assert_refused(o: &Output, input: &Path, original: &[u8]) {
    assert!(!o.status.success(), "must be refused; stdout: {}", String::from_utf8_lossy(&o.stdout));
    assert!(stderr(o).contains("refusing"), "the reason is given: {}", stderr(o));
    assert_eq!(std::fs::read(input).unwrap(), original, "the source is untouched");
}

/// Whether the file system under `dir` ignores case.
fn case_insensitive(dir: &Path) -> bool {
    let probe = source(dir, "CaseProbe.tmp", b"x");
    let yes = dir.join("caseprobe.TMP").exists();
    std::fs::remove_file(probe).unwrap();
    yes
}

#[test]
fn an_mp3_in_audio_mode_with_no_output_is_written_beside_it() {
    let dir = tempfile::tempdir().unwrap();
    let mp3 = mp3_tone();
    let input = source(dir.path(), "x.mp3", &mp3);
    let o = rivet(&[os("transcode"), input.as_os_str(), os("--mode"), os("audio")]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(std::fs::read(&input).unwrap(), mp3, "the source is untouched");
    let out = dir.path().join("x.rivet.mp3");
    assert!(std::fs::metadata(&out).unwrap().len() > 0, "the output is x.rivet.mp3");
    // Nothing else (no temporary) is left in the directory.
    let mut names: Vec<_> = std::fs::read_dir(dir.path()).unwrap().map(|e| e.unwrap().file_name()).collect();
    names.sort();
    assert_eq!(names, ["x.mp3", "x.rivet.mp3"]);

    // A FLAC source to FLAC: the same.
    let flac = dir.path().join("y.flac");
    let o = rivet(&[
        os("transcode"),
        input.as_os_str(),
        os("--mode"),
        os("audio"),
        os("--audio"),
        os("flac"),
        os("-o"),
        flac.as_os_str(),
    ]);
    assert!(o.status.success(), "{}", stderr(&o));
    let flac_bytes = std::fs::read(&flac).unwrap();
    let o = rivet(&[
        os("transcode"),
        flac.as_os_str(),
        os("--mode"),
        os("audio"),
        os("--audio"),
        os("flac"),
        os("--flac-compression"),
        os("fast"),
    ]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(std::fs::read(&flac).unwrap(), flac_bytes, "the FLAC source is untouched");
    assert!(dir.path().join("y.rivet.flac").exists());
}

#[test]
fn an_output_that_is_the_input_by_any_spelling_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mp3 = mp3_tone();
    let sub = dir.path().join("sub");
    std::fs::create_dir(&sub).unwrap();
    let input = source(&sub, "Song.mp3", &mp3);
    let mut spellings = vec![input.clone(), sub.join("..").join("sub").join("Song.mp3")];
    if case_insensitive(dir.path()) {
        spellings.push(sub.join("SONG.MP3"));
        spellings.push(dir.path().join("SUB").join("song.mp3"));
    }
    let link = sub.join("link.mp3");
    if std::fs::hard_link(&input, &link).is_ok() {
        spellings.push(link);
    }
    for out in &spellings {
        let o = rivet(&[os("transcode"), input.as_os_str(), os("--mode"), os("audio"), os("-o"), out.as_os_str()]);
        assert_refused(&o, &input, &mp3);
    }
    // Single mode with the same `-o` (an audio-only input is written as audio
    // by itself), and asked for explicitly as a video codec's file.
    let o = rivet(&[os("transcode"), input.as_os_str(), os("-o"), input.as_os_str()]);
    assert_refused(&o, &input, &mp3);
}

#[test]
fn a_video_source_is_never_its_own_output() {
    let dir = tempfile::tempdir().unwrap();
    let clip = common::synth::clip(64, 64, 10, 0.5, 0, 0, false);
    let input = source(dir.path(), "clip.mp4", &clip);
    // Single file.
    let upper = dir.path().join("CLIP.mp4");
    let target = if case_insensitive(dir.path()) { &upper } else { &input };
    let o = rivet(&[os("transcode"), input.as_os_str(), os("--codec"), os("mpeg4"), os("-o"), target.as_os_str()]);
    assert_refused(&o, &input, &clip);
    // HLS: the asset root is the input file, or holds the input where the
    // package writes.
    let o = rivet(&[os("transcode"), input.as_os_str(), os("--mode"), os("hls"), os("-o"), input.as_os_str()]);
    assert_refused(&o, &input, &clip);
    let pkg = dir.path().join("pkg");
    std::fs::create_dir_all(pkg.join("video")).unwrap();
    let inside = source(&pkg.join("video"), "clip.mp4", &clip);
    let o = rivet(&[os("transcode"), inside.as_os_str(), os("--mode"), os("hls"), os("-o"), pkg.as_os_str()]);
    assert_refused(&o, &inside, &clip);
    // A directory of rungs that is the input file.
    let o = rivet(&[
        os("transcode"),
        input.as_os_str(),
        os("--rung"),
        os("64x64"),
        os("--rung"),
        os("32x32"),
        os("-o"),
        input.as_os_str(),
    ]);
    assert_refused(&o, &input, &clip);
    // A splice whose output is one of its clips.
    let second = source(dir.path(), "b.mp4", &clip);
    let o = rivet(&[
        os("splice"),
        os("-o"),
        second.as_os_str(),
        input.as_os_str(),
        second.as_os_str(),
    ]);
    assert_refused(&o, &second, &clip);
}

#[cfg(feature = "image")]
#[test]
fn an_image_job_never_writes_over_its_input() {
    let dir = tempfile::tempdir().unwrap();
    let clip = common::synth::clip(64, 64, 10, 0.5, 0, 0, false);
    let input = source(dir.path(), "clip.mp4", &clip);
    let o = rivet(&[os("image"), input.as_os_str(), os("-o"), input.as_os_str()]);
    assert_refused(&o, &input, &clip);
}

#[cfg(feature = "batch")]
#[test]
fn a_batch_job_never_writes_over_its_input() {
    let dir = tempfile::tempdir().unwrap();
    let mp3 = mp3_tone();
    let a = source(dir.path(), "a.mp3", &mp3);
    let b = source(dir.path(), "b.mp3", &mp3);
    // `a` has no output: the default `<stem>.mp3` is its own name. `b` names
    // itself (in another case where the file system ignores it).
    let b_out = if case_insensitive(dir.path()) { "B.MP3" } else { "b.mp3" };
    let manifest = source(
        dir.path(),
        "jobs.yaml",
        format!("jobs:\n  - input: a.mp3\n    mode: audio\n  - input: b.mp3\n    mode: audio\n    output: {b_out}\n")
            .as_bytes(),
    );
    let o = rivet(&[os("batch"), manifest.as_os_str()]);
    assert!(!o.status.success(), "the refused job fails the batch");
    let out = format!("{}{}", String::from_utf8_lossy(&o.stdout), stderr(&o));
    assert!(out.contains("refusing"), "{out}");
    assert_eq!(std::fs::read(&a).unwrap(), mp3);
    assert_eq!(std::fs::read(&b).unwrap(), mp3);
    assert!(dir.path().join("a.rivet.mp3").exists(), "a's output is beside it");
}
