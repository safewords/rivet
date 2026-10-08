//! NDI end to end, through the real runtime, by the same path every front
//! end takes: an `ndi://` URI opened and probed, the settings built into a
//! spec against it, and the job engine's live path run on it.
//!
//! - In: a sender on this machine sends pictures and a 440 Hz tone; a job
//!   records `ndi://NAME` for three seconds, and the file's length, frame
//!   count and audio (the tone throughout, no dropout) are checked.
//! - Out: a file is played out to `ndi://NAME` by the live path, and a
//!   receiver on this machine counts what arrives.
//!
//! Needs the NDI runtime (https://ndi.video/tools/). Without one each test
//! prints SKIP and passes; `RIVET_REQUIRE_NDI=1` makes a missing runtime a
//! failure, for a runner that has one.
#![cfg(feature = "ndi")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rivet::{LiveEnd, LiveTarget, TranscodeSettings};

fn runtime() -> Option<ndi::Ndi> {
    match ndi::Ndi::load() {
        Ok(r) => Some(r),
        Err(e) if std::env::var_os("RIVET_REQUIRE_NDI").is_none() => {
            eprintln!("SKIP: {e}");
            None
        }
        Err(e) => panic!("RIVET_REQUIRE_NDI is set: {e}"),
    }
}

fn settings(kv: &[(&str, &str)]) -> TranscodeSettings {
    let mut s = TranscodeSettings::default();
    for (k, v) in kv {
        s.apply_kv(k, v).unwrap();
    }
    s
}

#[test]
fn an_ndi_source_records_in_step_through_the_spec() {
    let Some(runtime) = runtime() else { return };
    let name = format!("rivet-test-in-{}", std::process::id());
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
                let mut data = vec![(16 + (i % 200)) as u8; w * h];
                data.extend(vec![128u8; w * h / 2]);
                let samples: Vec<f32> = (0..1600)
                    .flat_map(|_| {
                        phase += 2.0 * std::f64::consts::PI * 440.0 / 48_000.0;
                        let s = (phase.sin() * 0.5) as f32;
                        [s, s]
                    })
                    .collect();
                sender.send_audio(48_000, 2, &samples, None).expect("audio");
                let picture = ndi::Picture {
                    layout: ndi::Layout::Yuv420p,
                    width: w as u32,
                    height: h as u32,
                    data,
                };
                sender
                    .send_picture(&picture, (30, 1), None)
                    .expect("picture");
                i += 1;
            }
        })
    };

    let uri = format!("ndi://{name}");
    let wait = Duration::from_secs(15);
    let source = rivet::live::open_source(&uri, wait).expect("the sender is found by name");
    let (info, source) = rivet::live::probe_source(source, wait).expect("probe");
    assert_eq!((info.width, info.height), (320, 180));
    assert!((info.frame_rate - 30.0).abs() < 1e-9);
    assert_eq!(info.audio.as_ref().map(|a| a.channels), Some(2));

    let spec = settings(&[("codec", "vp9"), ("container", "mp4"), ("duration", "3s")])
        .into_spec_for(&info)
        .expect("spec");
    let dir = tempfile::tempdir().unwrap();
    let out_path = dir.path().join("loopback.mp4");
    let out = rivet::run_live_job_blocking(
        source,
        &spec,
        LiveTarget::File(out_path.clone()),
        Arc::new(rivet::fn_sink(|_| {})),
        None,
    );
    stop.store(true, Ordering::Relaxed);
    sender.join().unwrap();
    let out = out.expect("record");
    let live = out.live.as_ref().unwrap();
    assert_eq!(live.ended, LiveEnd::Duration);
    assert_eq!((live.frames, live.frame_rate), (90, (30, 1)));
    assert_eq!(out.audio_codecs.as_deref(), Some("opus"));

    let bytes = std::fs::read(&out_path).unwrap();
    let probed = rivet::probe_bytes(&bytes).expect("probe");
    assert!((probed.duration - 3.0).abs() < 0.01, "{}", probed.duration);

    // The tone throughout: every 10 ms window carries it, and there are
    // about three seconds of it.
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
    let quiet: Vec<usize> = left[4_800..left.len() - 960]
        .chunks(480)
        .enumerate()
        .filter(|(_, w)| (w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32).sqrt() < 0.2)
        .map(|(i, _)| i)
        .collect();
    assert!(quiet.is_empty(), "silent 10 ms windows at {quiet:?}");
}

#[test]
fn a_file_played_out_to_ndi_arrives_whole() {
    let Some(runtime) = runtime() else { return };
    let name = format!("rivet-test-out-{}", std::process::id());

    // A second of 64x48 H.264 at 25 fps, made by rivet's own encoder and
    // muxer, as a file to send.
    let file = {
        use rivet::codec::frame::{EncodedPacket, VideoCodec};
        let mut enc = rivet::codec::encode::select_encoder(
            rivet::codec::encode::EncoderConfig {
                width: 64,
                height: 48,
                frame_rate: 25.0,
                keyframe_interval: 25,
                codec: VideoCodec::Vp9,
                ..Default::default()
            },
            None,
        )
        .expect("the VP9 encoder");
        let mut mux =
            rivet::container::mux::Av1Mp4Muxer::new_with_codec(64, 48, 25.0, VideoCodec::Vp9)
                .unwrap();
        let push = |p: EncodedPacket, mux: &mut rivet::container::mux::Av1Mp4Muxer| {
            mux.add_packet(p).unwrap()
        };
        for i in 0..25u64 {
            let mut data = vec![(16 + i * 8) as u8; 64 * 48];
            data.extend(vec![128u8; 64 * 48 / 2]);
            let frame = rivet::codec::frame::VideoFrame::new(
                bytes::Bytes::from(data),
                64,
                48,
                rivet::codec::frame::PixelFormat::Yuv420p,
                rivet::codec::frame::ColorSpace::Bt709,
                i,
            );
            enc.send_frame(&frame).unwrap();
            while let Some(p) = enc.receive_packet().unwrap() {
                push(p, &mut mux);
            }
        }
        enc.flush().unwrap();
        while let Some(p) = enc.receive_packet().unwrap() {
            push(p, &mut mux);
        }
        bytes::Bytes::from(mux.finalize().unwrap().to_vec())
    };

    // A receiver waiting for the stream before it is announced.
    let received = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let receiver = {
        let (runtime, name, received) = (runtime.clone(), name.clone(), Arc::clone(&received));
        std::thread::spawn(move || {
            let source = runtime
                .find_source(&name, &ndi::FindOptions::default(), Duration::from_secs(15))
                .expect("the stream is announced");
            let mut rx = runtime
                .receiver(&source, &ndi::ReceiverOptions::default())
                .expect("receiver");
            let until = Instant::now() + Duration::from_secs(20);
            while Instant::now() < until {
                if let ndi::Capture::Video(v) = rx.capture(Duration::from_millis(200)).unwrap() {
                    assert_eq!((v.width(), v.height()), (64, 48));
                    if received.fetch_add(1, Ordering::Relaxed) + 1 >= 25 {
                        return;
                    }
                }
            }
        })
    };

    let probed = rivet::probe_bytes(&file).unwrap();
    // Played twice over (loop, two seconds), so the receiver, which
    // connects a moment after the stream appears, still sees a whole second.
    let spec = settings(&[("loop", "true"), ("duration", "3s")])
        .into_spec_for(&probed)
        .unwrap();
    let source = rivet::live::FileSource::new("clip", file, spec.live.repeat).unwrap();
    let rivet::live::LiveUri::Ndi(endpoint) =
        rivet::live::LiveUri::parse(&format!("ndi://{name}")).unwrap();
    let out = rivet::run_live_job_blocking(
        source,
        &spec,
        LiveTarget::Ndi(endpoint),
        Arc::new(rivet::fn_sink(|_| {})),
        None,
    )
    .expect("send");
    receiver.join().unwrap();
    assert!(
        matches!(&out.rungs[0].artifact, rivet::RungArtifact::Ndi { source } if *source == name)
    );
    assert_eq!(out.live.as_ref().unwrap().frames, 75);
    assert!(received.load(Ordering::Relaxed) >= 25);
}

/// A sender of 320x180 pictures at 30 fps and a 440 Hz tone, named `name`,
/// until `stop` is set.
fn spawn_sender(
    runtime: &ndi::Ndi,
    name: &str,
    stop: &Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    let (runtime, name, stop) = (runtime.clone(), name.to_string(), Arc::clone(stop));
    std::thread::spawn(move || {
        let mut sender = runtime
            .sender(&ndi::SenderOptions::new(name))
            .expect("sender");
        let (w, h) = (320usize, 180usize);
        let mut phase = 0f64;
        let mut i = 0u64;
        while !stop.load(Ordering::Relaxed) {
            let mut data = vec![(16 + (i % 200)) as u8; w * h];
            data.extend(vec![128u8; w * h / 2]);
            let samples: Vec<f32> = (0..1600)
                .flat_map(|_| {
                    phase += 2.0 * std::f64::consts::PI * 440.0 / 48_000.0;
                    let s = (phase.sin() * 0.5) as f32;
                    [s, s]
                })
                .collect();
            sender.send_audio(48_000, 2, &samples, None).expect("audio");
            let picture = ndi::Picture {
                layout: ndi::Layout::Yuv420p,
                width: w as u32,
                height: h as u32,
                data,
            };
            sender
                .send_picture(&picture, (30, 1), None)
                .expect("picture");
            i += 1;
        }
    })
}

/// Several sources at once: a batch manifest naming two NDI sources records
/// both side by side (live jobs start together), each to its own file, the
/// shared `defaults` applied to both.
#[cfg(feature = "batch")]
#[test]
fn a_manifest_records_several_sources_at_once() {
    let Some(runtime) = runtime() else { return };
    let pid = std::process::id();
    let names = [format!("rivet-test-a-{pid}"), format!("rivet-test-b-{pid}")];
    let stop = Arc::new(AtomicBool::new(false));
    let senders: Vec<_> = names
        .iter()
        .map(|n| spawn_sender(&runtime, n, &stop))
        .collect();

    let dir = tempfile::tempdir().unwrap();
    let yaml = format!(
        "defaults:\n  codec: vp9\n  container: mp4\n  duration: 2s\njobs:\n  - input: \"ndi://{}\"\n    output: a.mp4\n  - input: \"ndi://{}\"\n    output: b.mp4\n    rungs: [\"160x90\"]\n",
        names[0], names[1]
    );
    let manifest = rivet::manifest::parse_manifest(&yaml, rivet::manifest::Format::Yaml).unwrap();
    let started = Instant::now();
    let report = rivet::manifest::run_manifest(&manifest, dir.path()).unwrap();
    let took = started.elapsed();
    stop.store(true, Ordering::Relaxed);
    for s in senders {
        s.join().unwrap();
    }
    assert!(report.all_ok(), "{:?}", report.outcomes);
    assert_eq!(report.outcomes.len(), 2);
    for (file, (w, h)) in [("a.mp4", (320, 180)), ("b.mp4", (160, 90))] {
        let info = rivet::probe_file(dir.path().join(file)).expect(file);
        assert_eq!((info.width, info.height), (w, h), "{file}");
        assert!(
            (info.duration - 2.0).abs() < 0.01,
            "{file}: {}",
            info.duration
        );
        assert!(info.audio.is_some(), "{file}");
    }
    // Two seconds each, side by side: well under the four they would take
    // one after the other (plus each one's connect).
    assert!(took < Duration::from_secs(6), "took {took:?}");
}

/// The HTTP API: a live job started by a JSON request with an `ndi://`
/// input, reported as running, ended by `POST /v1/jobs/{id}/stop`, and its
/// file written.
#[cfg(feature = "server")]
#[tokio::test(flavor = "multi_thread")]
async fn the_api_runs_a_live_job_until_stopped() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    let Some(runtime) = runtime() else { return };
    let name = format!("rivet-test-api-{}", std::process::id());
    let stop = Arc::new(AtomicBool::new(false));
    let sender = spawn_sender(&runtime, &name, &stop);
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("api.mp4");

    let app = rivet::server::build_router();
    let call = |method: &str, uri: String, body: Option<serde_json::Value>| {
        let app = app.clone();
        let req = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(match body {
                Some(b) => Body::from(b.to_string()),
                None => Body::empty(),
            })
            .unwrap();
        async move {
            let resp = app.oneshot(req).await.unwrap();
            let status = resp.status();
            let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
            (
                status,
                serde_json::from_slice::<serde_json::Value>(&bytes).unwrap_or_default(),
            )
        }
    };
    let (status, body) = call(
        "POST",
        "/v1/transcode".into(),
        Some(serde_json::json!({
            "input": { "path": format!("ndi://{name}") },
            "output": { "path": out.display().to_string() },
            "spec": { "codec": "vp9", "container": "mp4" }
        })),
    )
    .await;
    assert_eq!(status.as_u16(), 202, "{body}");
    let id = body["job_id"].as_str().unwrap().to_string();
    assert_eq!(body["stop"], format!("/v1/jobs/{id}/stop"));

    // Running, then stopped.
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let (_, status) = call("GET", format!("/v1/jobs/{id}"), None).await;
        if status["status"] == "running"
            && status["progress"].as_array().is_some_and(|p| !p.is_empty())
        {
            break;
        }
        assert!(Instant::now() < deadline, "never started: {status}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let (status, _) = call("POST", format!("/v1/jobs/{id}/stop"), None).await;
    assert_eq!(status.as_u16(), 202);
    let done = loop {
        let (_, status) = call("GET", format!("/v1/jobs/{id}"), None).await;
        if status["status"] != "running" && status["status"] != "queued" {
            break status;
        }
        assert!(Instant::now() < deadline, "never ended: {status}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    stop.store(true, Ordering::Relaxed);
    sender.join().unwrap();
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(done["live"]["ended"], "stopped", "{done}");
    assert!(done["live"]["frames"].as_u64().unwrap() >= 30, "{done}");
    // The server names the file as it resolved it (canonical: `\?\C:\…` on
    // Windows); the same file either way.
    let written = std::path::PathBuf::from(done["artifacts"][0]["output_path"].as_str().unwrap());
    assert!(same_file::is_same_file(&written, &out).unwrap(), "{done}");
    let info = rivet::probe_file(&out).unwrap();
    assert!(info.duration >= 1.0, "{}", info.duration);
}
