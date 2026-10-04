//! HTTP transcode API (`rivet serve`, behind the `server` feature).
//!
//! A small [axum] webserver so another application can **signal rivet to
//! transcode** something over the network: it POSTs media bytes plus an output
//! spec, rivet runs the job on the same configurable engine the CLI uses, and
//! reports progress + serves the output artifacts.
//!
//! Endpoints (all under `/v1`):
//! - `GET  /v1/health` — liveness + detected GPUs + build capabilities.
//! - `POST /v1/probe` — body = media bytes → JSON [`MediaInfo`](crate::probe::MediaInfo).
//! - `POST /v1/transcode` — body = media bytes, spec from query params. Returns
//!   `202 { job_id }` and runs asynchronously; pass `?sync=true` to block and
//!   get a single-file, single-rung job's file back directly (several rungs,
//!   HLS or an `output.path` get the job status JSON instead).
//! - `GET  /v1/jobs/{id}` — job status + per-rung progress + output list.
//! - `GET  /v1/jobs/{id}/artifacts/{label}` — download a single-file rung's MP4.
//! - `GET  /v1/jobs/{id}/files/{*path}` — fetch a file from an HLS job's output
//!   tree (e.g. `master.m3u8`, `video/720p/seg-00001.m4s`).
//! - `GET  /v1/hooks` — the hooks the server runs (see [`crate::hooks`]).
//!
//! **Hooks.** A server built with [`build_router_with_hooks`] /
//! [`serve_with_hooks`] runs its required hooks on every job, and the optional
//! ones a request names (`?hooks=a,b`, or `"hooks": [...]` in a JSON body).
//! Requests choose among configured hooks only; they cannot define one. Each
//! job's status carries its hook report (`hooks`), and a job a hook rejected
//! ends `rejected` with the rejection (a `?sync=true` request gets `422`).
//!
//! **Concurrency.** By default the server starts every job it accepts at
//! once: there is no limit on how many run together. A limit is the
//! operator's choice — `rivet serve --jobs N`, or [`SERVER_JOBS_ENV`] — and
//! with one set, a request accepted while N jobs run stays `queued` until one
//! ends, in arrival order. Either way every job may use every hardware
//! encoder and decoder its plan selects (all of them by default): the limit
//! counts jobs, it never assigns a job a card. The CPU is shared among the
//! jobs ([`crate::thread_budget`]): a job's thread budget — software encoders
//! and decoders, worker pools, the decode pump's filters and colour
//! conversions — is the machine divided by the jobs running when it starts
//! them, or by N when a limit is set and that is more, so the jobs together
//! do not oversubscribe the machine.
//!
//! The job registry is in-memory; completed single-file artifacts are held in
//! RAM until the process exits (fine for a sidecar/worker, not a public CDN —
//! a production deployment would offload to object storage from a `ProgressSink`
//! watching `RungStatus::Completed`).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Context, Result};
use axum::Router;
use axum::body::Bytes;
use axum::extract::DefaultBodyLimit;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::progress::{ProgressSink, RungProgress, RungStatus};

mod docs;
mod handlers;
mod spec;
#[cfg(test)]
mod tests;

// Re-export the public items so `rivet::server::X` paths resolve.
pub use docs::openapi_spec;

/// 4 GiB upload ceiling — large enough for long source files.
pub(super) const MAX_UPLOAD: usize = 4 * 1024 * 1024 * 1024;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Environment variable: the most jobs `rivet serve` runs at once, a whole
/// number of at least 1. Unset (or not such a number), there is no limit.
/// `rivet serve --jobs` takes precedence over it. Jobs beyond the limit
/// wait, `queued`, in arrival order.
pub const SERVER_JOBS_ENV: &str = "RIVET_SERVER_JOBS";

/// The server's job limit: `flag` (`rivet serve --jobs`) when given, else
/// `env` (the [`SERVER_JOBS_ENV`] setting) when it is a whole number of at
/// least one, else `None` — no limit, every accepted job starts at once.
pub(super) fn job_limit(flag: Option<usize>, env: Option<&str>) -> Option<usize> {
    flag.filter(|&n| n >= 1).or_else(|| {
        env.and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n >= 1)
    })
}

#[derive(Clone)]
pub struct AppState {
    pub(super) jobs: Arc<RwLock<HashMap<Uuid, Arc<JobHandle>>>>,
    /// The hooks every job can run ([`crate::hooks`]).
    pub(super) hooks: crate::hooks::Hooks,
    /// With a job limit ([`job_limit`]), one permit per job that may run at
    /// once; a job holds one from leaving `queued` to its end. Fair: waiting
    /// jobs start in the order they were accepted. `None`: no limit.
    pub(super) running: Option<Arc<tokio::sync::Semaphore>>,
}

impl AppState {
    /// `jobs`: the operator's `--jobs`, if given ([`SERVER_JOBS_ENV`] else).
    fn new(hooks: crate::hooks::Hooks, jobs: Option<usize>) -> Self {
        let env = std::env::var(SERVER_JOBS_ENV).ok();
        if jobs.is_none()
            && let Some(v) = env.as_deref()
            && job_limit(None, Some(v)).is_none()
        {
            tracing::warn!(
                value = v,
                "{SERVER_JOBS_ENV} is not a whole number of at least 1; no job limit"
            );
        }
        let limit = job_limit(jobs, env.as_deref());
        // With a limit, every job's share of the CPU is reckoned against it,
        // so the first of N jobs does not size its pools to the whole
        // machine before the others start. Without one, each job's share is
        // the machine divided by the jobs running when it asks.
        crate::thread_budget::reserve_jobs(limit.unwrap_or(0));
        match limit {
            Some(n) => tracing::info!(
                jobs = n,
                "rivet serve: at most {n} jobs at once; the rest wait queued"
            ),
            None => tracing::info!("rivet serve: no job limit; every accepted job starts at once"),
        }
        Self::with_limit(hooks, limit)
    }

    pub(super) fn with_limit(hooks: crate::hooks::Hooks, limit: Option<usize>) -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            hooks,
            running: limit.map(|n| Arc::new(tokio::sync::Semaphore::new(n.max(1)))),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    Queued,
    Running,
    Completed,
    Failed,
    /// A hook stopped the job.
    Rejected,
}

impl Phase {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Phase::Queued => "queued",
            Phase::Running => "running",
            Phase::Completed => "completed",
            Phase::Failed => "failed",
            Phase::Rejected => "rejected",
        }
    }
}

pub(super) struct ArtifactEntry {
    pub(super) label: String,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) frames: u64,
    pub(super) bytes: u64,
    /// In-memory MP4 bytes for a single-file rung held in RAM; `None` for an
    /// HLS rendition or when the bytes were written to `output_path`.
    pub(super) data: Option<Bytes>,
    /// Server-side path the artifact was written to (when the request supplied
    /// `output.path`); surfaced in the status JSON.
    pub(super) output_path: Option<String>,
}

pub(super) struct JobHandle {
    pub(super) id: Uuid,
    mode: String,
    pub(super) phase: Mutex<Phase>,
    progress: Mutex<Vec<RungProgress>>,
    pub(super) artifacts: Mutex<Vec<ArtifactEntry>>,
    pub(super) error: Mutex<Option<String>>,
    /// HLS output root (a temp dir), if any.
    pub(super) output_dir: Mutex<Option<PathBuf>>,
    pub(super) master_playlist: Mutex<Option<String>>,
    /// What each requested rung came out as (see [`crate::fit::FittedRung`]).
    pub(super) renditions: Mutex<Vec<crate::fit::FittedRung>>,
    /// The job's hook session; its report is read live into the status.
    pub(super) hooks: Mutex<crate::hooks::Hooks>,
}

impl JobHandle {
    pub(super) fn new(id: Uuid, mode: &str) -> Self {
        Self {
            id,
            mode: mode.to_string(),
            phase: Mutex::new(Phase::Queued),
            progress: Mutex::new(Vec::new()),
            artifacts: Mutex::new(Vec::new()),
            error: Mutex::new(None),
            output_dir: Mutex::new(None),
            master_playlist: Mutex::new(None),
            renditions: Mutex::new(Vec::new()),
            hooks: Mutex::new(crate::hooks::Hooks::default()),
        }
    }

    pub(super) fn set_phase(&self, p: Phase) {
        *self.phase.lock().unwrap() = p;
    }

    pub(super) fn status_json(&self) -> Value {
        let phase = *self.phase.lock().unwrap();
        let progress: Vec<Value> = self
            .progress
            .lock()
            .unwrap()
            .iter()
            .map(rung_progress_json)
            .collect();
        let artifacts: Vec<Value> = self
            .artifacts
            .lock()
            .unwrap()
            .iter()
            .map(|a| {
                // Download URL only when bytes are held in RAM; when written to
                // disk (`output_path`) the caller already has the path.
                let url = if a.data.is_some() {
                    Some(format!("/v1/jobs/{}/artifacts/{}", self.id, a.label))
                } else if a.output_path.is_none() {
                    Some(format!("/v1/jobs/{}/files/", self.id))
                } else {
                    None
                };
                json!({
                    "label": a.label,
                    "width": a.width,
                    "height": a.height,
                    "frames": a.frames,
                    "bytes": a.bytes,
                    "url": url,
                    "output_path": a.output_path,
                })
            })
            .collect();
        let hooks = {
            let report = self.hooks.lock().unwrap().report();
            report.job_id.is_some().then(|| report.to_json())
        };
        json!({
            "job_id": self.id.to_string(),
            "mode": self.mode,
            "status": phase.as_str(),
            "progress": progress,
            "artifacts": artifacts,
            // One per requested rung, in request order: the box asked for,
            // the size produced, and the rung it merged into when it came out
            // the same as an earlier one.
            "renditions": self.renditions.lock().unwrap().iter().map(|r| json!({
                "label": r.label,
                "requested": { "width": r.requested.0, "height": r.requested.1 },
                "output": { "width": r.output.0, "height": r.output.1 },
                "fit": r.fit.as_str(),
                "duplicate_of": r.duplicate_of,
            })).collect::<Vec<_>>(),
            "master_playlist": *self.master_playlist.lock().unwrap(),
            "error": *self.error.lock().unwrap(),
            // What the job's hooks said so far; null when it runs none.
            "hooks": hooks,
        })
    }
}

fn rung_progress_json(p: &RungProgress) -> Value {
    json!({
        "rung_index": p.rung_index,
        "label": p.label,
        "width": p.width,
        "height": p.height,
        "status": rung_status_str(p.status),
        "percent": p.percent,
        "frames_done": p.frames_done,
        // Why a failed rung failed, its whole error chain; null otherwise.
        "message": p.message,
    })
}

fn rung_status_str(s: RungStatus) -> &'static str {
    match s {
        RungStatus::Pending => "pending",
        RungStatus::Running => "running",
        RungStatus::Finalizing => "finalizing",
        RungStatus::Completed => "completed",
        RungStatus::Failed => "failed",
    }
}

/// A [`ProgressSink`] that mirrors per-rung updates into a [`JobHandle`].
pub(super) struct RegistrySink {
    pub(super) handle: Arc<JobHandle>,
}

impl ProgressSink for RegistrySink {
    fn on_rung(&self, update: RungProgress) {
        let mut prog = self.handle.progress.lock().unwrap();
        match prog.iter_mut().find(|p| p.rung_index == update.rung_index) {
            Some(slot) => *slot = update,
            None => prog.push(update),
        }
    }
}

// ---------------------------------------------------------------------------
// Response helpers (shared across handlers)
// ---------------------------------------------------------------------------

/// JSON response wrapper (so handlers can return `Json`).
pub(super) struct Json(pub(super) Value);

impl IntoResponse for Json {
    fn into_response(self) -> Response {
        (
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::to_vec(&self.0).unwrap_or_default(),
        )
            .into_response()
    }
}

/// A JSON error with an HTTP status.
pub(super) struct ApiError {
    pub(super) status: StatusCode,
    pub(super) message: String,
}

impl ApiError {
    pub(super) fn bad_request(e: anyhow::Error) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: format!("{e:#}"),
        }
    }
    pub(super) fn internal(e: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("{e:#}"),
        }
    }
    pub(super) fn not_found(what: String) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: format!("{what} not found"),
        }
    }
    pub(super) fn rejected(message: String) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            message,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::to_vec(&json!({ "error": self.message })).unwrap_or_default(),
        )
            .into_response()
    }
}

// ---------------------------------------------------------------------------
// Router entry points
// ---------------------------------------------------------------------------

/// Build the axum router (also the test entry point), with no hooks.
pub fn build_router() -> Router {
    build_router_with_hooks(crate::hooks::Hooks::default())
}

/// Build the axum router with `hooks` available to every job: the required
/// ones run on all of them, the optional ones on the jobs that name them.
pub fn build_router_with_hooks(hooks: crate::hooks::Hooks) -> Router {
    build_router_with(hooks, None)
}

/// [`build_router_with_hooks`] with a job limit: `jobs` runs at most that
/// many at once, the rest waiting `queued`; `None` leaves it to
/// [`SERVER_JOBS_ENV`], and with that unset there is no limit.
pub fn build_router_with(hooks: crate::hooks::Hooks, jobs: Option<usize>) -> Router {
    let state = AppState::new(hooks, jobs);
    Router::new()
        .route("/", get(handlers::landing))
        .route("/openapi.json", get(handlers::openapi_json))
        .route("/swagger", get(handlers::swagger_ui))
        .route("/redoc", get(handlers::redoc_ui))
        .route("/v1/health", get(handlers::health))
        .route("/v1/hooks", get(handlers::hooks))
        .route("/v1/probe", post(handlers::probe))
        .route("/v1/transcode", post(handlers::transcode))
        .route("/v1/jobs/{id}", get(handlers::job_status))
        .route("/v1/jobs/{id}/artifacts/{label}", get(handlers::artifact))
        .route("/v1/jobs/{id}/files/{*path}", get(handlers::hls_file))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD))
        .with_state(state)
}

/// Run the server, blocking until shutdown.
pub async fn serve(addr: SocketAddr) -> Result<()> {
    serve_with_hooks(addr, crate::hooks::Hooks::default()).await
}

/// [`serve`] with `hooks` available to every job — how an integration that
/// embeds the server attaches its own [`Hook`](crate::hooks::Hook)s.
pub async fn serve_with_hooks(addr: SocketAddr, hooks: crate::hooks::Hooks) -> Result<()> {
    serve_with(addr, hooks, None).await
}

/// [`serve_with_hooks`] with a job limit (`rivet serve --jobs`): see
/// [`build_router_with`]. `None` leaves it to [`SERVER_JOBS_ENV`], and with
/// that unset every accepted job starts at once.
pub async fn serve_with(
    addr: SocketAddr,
    hooks: crate::hooks::Hooks,
    jobs: Option<usize>,
) -> Result<()> {
    let app = build_router_with(hooks, jobs);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    tracing::info!(%addr, "rivet transcode API listening");
    axum::serve(listener, app).await.context("axum serve")?;
    Ok(())
}
