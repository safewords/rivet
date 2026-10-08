//! NDI® in and out (opt-in `ndi` feature).
//!
//! An NDI source is a job's input, and an NDI stream its output, by URI —
//! `ndi://NAME` wherever a path goes — and the job is the same job any input
//! gets: the [`OutputSpec`](crate::spec::OutputSpec) the settings build, run
//! by the job engine's live path ([`crate::job::run_live_job`]). This module
//! is the NDI side of that: the receiver as a [`LiveSource`](crate::live::LiveSource)
//! ([`NdiSource`]), the sender a live job's NDI rungs write to ([`NdiSink`]),
//! and discovery ([`list_sources`]).
//!
//! The protocol is the [`ndi`] crate's (`rivet-ndi`): hand-rolled FFI that
//! loads the NDI runtime when first used, so a build with the feature needs
//! nothing from NDI, and a host without the runtime is told how to get it
//! the first time an NDI job runs.
//!
//! NDI® is a registered trademark of Vizrt NDI AB.

mod sink;
mod source;

use std::time::Duration;

pub use sink::NdiSink;
pub use source::{NdiSource, ndi_color, picture_to_frame};

/// Every NDI source seen within `wait`.
pub fn list_sources(
    options: &ndi::FindOptions,
    wait: Duration,
) -> anyhow::Result<Vec<ndi::Source>> {
    Ok(ndi::Ndi::load()?.sources(options, wait)?)
}
