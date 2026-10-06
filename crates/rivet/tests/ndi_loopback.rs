//! NDI end to end, through the real runtime: a sender on this machine sends
//! pictures and a 440 Hz tone, the recorder finds it by name, records three
//! seconds, and the file is checked — its length, its pictures, and audio
//! that is the tone throughout, with no dropout.
//!
//! Needs the NDI runtime (https://ndi.video/tools/). Without one the test
//! prints SKIP and passes; `RIVET_REQUIRE_NDI=1` makes a missing runtime a
//! failure, for a runner that has one.
#![cfg(feature = "ndi")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use rivet::codec::encode::EncoderBackend;
use rivet::ndi::{EndReason, NdiSource, RecordOptions, record};
use rivet::spec::VideoCodecPolicy;

#[test]
fn a_source_on_this_machine_records_in_step() {
    let runtime = match ndi::Ndi::load() {
        Ok(r) => r,
        Err(e) if std::env::var_os("RIVET_REQUIRE_NDI").is_none() => {
            eprintln!("SKIP: {e}");
            return;
        }
        Err(e) => panic!("RIVET_REQUIRE_NDI is set: {e}"),
    };
    let name = format!("rivet-test-{}", std::process::id());
    let stop = Arc::new(AtomicBool::new(false));
    let sender = {
        let (runtime, name, stop) = (runtime.clone(), name.clone(), Arc::clone(&stop));
        std::thread::spawn(move || {
            let mut sender = runtime
                .sender(&ndi::SenderOptions::new(name))
                .expect("sender");
            let (w, h) = (320usize, 180usize);
            let mut phase = 0f64;
            let mut i = 0u64;
            while !stop.load(Ordering::Relaxed) {
                // A grey ramp whose brightness steps each frame, and 1/30 s
                // of the tone; the sender clocks the video at 30 fps.
                let mut data = vec![(16 + (i % 200)) as u8; w * h];
                data.extend(vec![128u8; w * h / 2]);
                let samples: Vec<f32> = (0..1600)
                    .flat_map(|_| {
                        phase += 2.0 * std::f64::consts::PI * 440.0 / 48_000.0;
                        let s = (phase.sin() * 0.5) as f32;
                        [s, s]
                    })
                    .collect();
                sender
                    .send_audio(48_000, 2, &samples, None)
                    .expect("send audio");
                let picture = ndi::Picture {
                    layout: ndi::Layout::Yuv420p,
                    width: w as u32,
                    height: h as u32,
                    data,
                };
                sender
                    .send_picture(&picture, (30, 1), None)
                    .expect("send picture");
                i += 1;
            }
        })
    };

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("loopback.mp4");
    let mut options = RecordOptions::new(&out);
    options.video_codec = VideoCodecPolicy::H264;
    options.encoder_backend = Some(EncoderBackend::H26x);
    options.duration = Some(Duration::from_secs(3));
    let source = NdiSource::connect(
        &name,
        &ndi::FindOptions::default(),
        &ndi::ReceiverOptions::default(),
        Duration::from_secs(15),
    )
    .expect("the sender is found by name");
    let outcome = record(source, &options, |_| {});
    stop.store(true, Ordering::Relaxed);
    sender.join().unwrap();
    let outcome = outcome.expect("record");

    assert_eq!(outcome.ended, EndReason::Limit);
    assert_eq!((outcome.width, outcome.height), (320, 180));
    assert_eq!(outcome.frame_rate, (30, 1));
    assert_eq!(outcome.progress.frames, 90);
    assert_eq!(outcome.audio.as_deref(), Some("opus 2ch 48000 Hz"));

    let bytes = std::fs::read(&out).unwrap();
    let info = rivet::probe_bytes(&bytes).expect("probe");
    assert!((info.duration - 3.0).abs() < 0.01, "{}", info.duration);

    // The audio decodes to the tone throughout: every 10 ms window carries
    // it (no dropout padded with silence), and there are about three
    // seconds of it.
    let demuxer = rivet::container::streaming::demux_streaming(&bytes).expect("demux");
    let track = demuxer.audio().cloned().expect("an audio track");
    let mut decoder = rivet::codec::audio::create_decoder(
        "opus",
        Some(&track.codec_private),
        48_000,
        track.channels as u8,
    )
    .expect("opus decoder");
    let mut left = Vec::new();
    for packet in &track.samples {
        for f in decoder.decode(packet, 0).expect("decode") {
            left.extend(f.samples.as_chunks::<2>().0.iter().map(|s| s[0]));
        }
    }
    let seconds = left.len() as f64 / 48_000.0;
    assert!((2.8..3.3).contains(&seconds), "{seconds} s of audio");
    // Skip the encoder's priming and the first picture's lead-in.
    let quiet: Vec<usize> = left[4_800..left.len() - 960]
        .chunks(480)
        .enumerate()
        .filter(|(_, w)| (w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32).sqrt() < 0.2)
        .map(|(i, _)| i)
        .collect();
    assert!(quiet.is_empty(), "silent 10 ms windows at {quiet:?}");
}
