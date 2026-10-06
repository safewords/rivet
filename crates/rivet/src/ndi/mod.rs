//! NDI® in and out (opt-in `ndi` feature): record an NDI source into a
//! file, and play a file out as an NDI source.
//!
//! The NDI protocol is the [`ndi`] crate's (`rivet-ndi`): hand-rolled FFI
//! that loads the NDI runtime when first used, so a build with the feature
//! needs nothing from NDI, and a host without the runtime is told how to
//! get it the first time an NDI command runs.
//!
//! - [`record()`] — an NDI source (or any [`LiveSource`]) → the job engine's
//!   per-frame normalisation (colour, tonemap, depth, filters) → any video
//!   encoder rivet has, GPU first → MP4 / QuickTime / WebM, with the audio
//!   in Opus or AAC, kept in step by the source's timestamps.
//! - [`send_file`] — a file → decode → NDI (I420, or P216 at 10 bits),
//!   paced at the file's frame rate, its audio alongside.
//! - [`list_sources`] — the sources discovery sees.
//!
//! NDI® is a registered trademark of Vizrt NDI AB.

pub mod live;
pub mod record;
pub mod send;

use std::time::Duration;

pub use live::{LiveAudio, LiveEvent, LiveSource, LiveVideo, NdiSource, SourceLost};
pub use record::{EndReason, RecordAudio, RecordOptions, RecordOutcome, RecordProgress, record};
pub use send::{SendOptions, SendOutcome, send_file};

/// Every NDI source seen within `wait`.
pub fn list_sources(
    options: &ndi::FindOptions,
    wait: Duration,
) -> anyhow::Result<Vec<ndi::Source>> {
    Ok(ndi::Ndi::load()?.sources(options, wait)?)
}
