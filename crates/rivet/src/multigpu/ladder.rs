//! The ladder core shared by the HLS and single-file paths.
//!
//! ```text
//!   decode worker per card ──► per-rung scaler ──► SegmentChunkQueue (per rung)
//!   (pulls ranges by speed)    (one per range × rung)        │
//!                                                            ▼
//!                                       ladder worker (one per GPU, serves EVERY rung)
//! ```
//!
//! Everything about *how* the ladder is scheduled lives here, once: the
//! range-split decode, the scalers and their continuous segment numbering,
//! the byte-budgeted queues, the workers, the setup guard on the active count
//! and the "finished" rule the finalizers wait on. What differs between the
//! two output paths — what a worker does with a chunk (write a CMAF segment
//! file, or collect its packets) and what a finalizer does with a rung's
//! contributions (merge manifests, or stitch packets) — is passed in.
//!
//! # Two things distinguish this from a worker-per-rung ladder
//!
//! Both are about never letting a card sit idle while work exists.
//!
//! **Workers serve the whole ladder.** Each holds one GPU lease for the life
//! of the job and repeatedly takes the next chunk from whichever rung is
//! furthest behind ([`EncodePolicy::AllGpus`](crate::spec::EncodePolicy::AllGpus)). A per-rung worker idled the
//! moment its rung was blocked even with another rung's chunks sitting ready;
//! it also capped the rungs in flight at the GPU count, so a longer ladder fell
//! back to decoding the source once per rung — and decode is the dominant cost
//! of a transcode. Because no rung can now be left without a consumer, the pump
//! is always shared and the ladder costs exactly one decode however many rungs
//! it has. [`EncodePolicy::PerRung`](crate::spec::EncodePolicy::PerRung) keeps the pinned shape available for
//! comparison and for hosts where placement matters more than throughput.
//!
//! **The decode is split across the cards.** One decoder for the whole ladder
//! is one decoder, and the giveaway that it is the limiter is rungs of very
//! different encode cost advancing in lockstep on the same segment number.
//! [`plan_decode_ranges`](crate::decode_pump::plan_decode_ranges) cuts the
//! source at keyframes that fall on chunk boundaries — several ranges per
//! card — and each card's decode worker pulls the next range when it is free
//! and runs a pump over it into every rung's scaler; the numbering stays
//! continuous across the joins ([`DecodePolicy`](crate::spec::DecodePolicy)).
//! A source that cannot be split safely is decoded whole, on the card
//! expected to be fastest.
//!
//! **Cards of different speeds share the work by speed.** Pulling already
//! gives a fast card more ranges and more chunks; what pulling alone gets
//! wrong is the end, where the last units go to whoever asks first. Both the
//! decode workers and the ladder workers ask a finish-time gate before they
//! take a unit ([`SpeedBoard::should_take`]): a card that would finish the
//! unit after the others had finished *everything* left steps aside, so the
//! tail of the job runs on the fast cards. Speeds are measured as the job runs
//! (and kept for the process); before the first measurement a card's memory
//! and PCIe link stand in ([`speed`](super::speed)).
//!
//! One encoder per GPU is still exactly true: `capacity` workers, each holding
//! its lease for its lifetime, each running one encode at a time. That
//! invariant is load-bearing — concurrent sessions on one device deadlocked at
//! init — and nothing here widens it.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Context, Result, anyhow, bail};
use tokio::sync::{Notify, mpsc, watch};
use tokio::task::JoinSet;


use crate::decode_pump::DecodeRange;
use super::speed::{self, DeviceKey, SpeedBoard};
use crate::encoder_worker::{EncoderSessionPool, EncoderWorkerConfig, RungCodecInvariant};
use crate::frame_queue::{SegmentChunk, SegmentChunkQueue};
use crate::gpu_pool::GpuLease;
use crate::spec::Rung;

use super::{FANOUT_CHANNEL_CAPACITY, MultiGpuParams, WorkerCtx, queue_capacity_for};

/// How long a ladder worker waits when every queue it serves is empty but the
/// job is not over — the normal state of a rung whose scaler is mid-chunk.
/// Short, because the wait is on the encode critical path; not zero, because a
/// worker that never yields spins a core against the decoders.
const IDLE_POLL: std::time::Duration = std::time::Duration::from_millis(5);

/// The chunking a path asks for.
#[derive(Debug, Clone, Copy)]
pub(super) struct LadderShape {
    /// Frames per unit of work — a CMAF segment for HLS, several GOPs for
    /// single-file. Also the grid decode ranges must land on.
    pub frames_per_chunk: u32,
    /// Lead-in margin frames replayed from the previous chunk's tail (encoded
    /// to warm the encoder, then discarded). `0` for HLS, whose segments must
    /// each stand alone; one GOP for single-file chunk-and-stitch.
    ///
    /// At a decode-range boundary the first chunk of the new range has no
    /// tail to replay — that range's scaler never saw those frames — so it
    /// starts cold, exactly as the file's first chunk always does. Its first
    /// kept frame is still an IDR by the encoder's own cadence, so the stitch
    /// is correct; the seam is merely one warm-up less flat, once per range.
    pub overlap: usize,
}

/// What one unit of work produced, whatever the unit is.
pub(super) enum UnitOutcome<T> {
    /// The unit was encoded and its result recorded.
    Done(T),
    /// This worker's vendor disagrees with the rung's codec invariant on a
    /// mandatory field. The chunk comes back untouched for another card.
    Rejected { chunk: SegmentChunk, diff: String },
}

/// The per-run state every part of the ladder shares. `T` is one worker's
/// contribution to a rung — a segment's [`SegmentInfo`](container::cmaf::SegmentInfo)
/// wrapped as a `WorkerOutput`, or a chunk's packets.
pub(super) struct Ladder<T> {
    pub queues: Vec<Arc<SegmentChunkQueue>>,
    pub frames_encoded: Vec<Arc<AtomicU64>>,
    pub bytes_encoded: Vec<Arc<AtomicU64>>,
    pub rung_invariants: Vec<Arc<RwLock<Option<RungCodecInvariant>>>>,
    /// Outputs from every worker on a rung, accumulated until the rung's
    /// finalizer drains it.
    pub contributions: Arc<Vec<Mutex<Vec<T>>>>,
    /// How many workers may still take chunks of each rung. Set by
    /// [`spawn_workers`] from the workers' rung lists and decremented each
    /// time a worker strikes the rung off (codec-invariant refusal). At zero
    /// with work still queued, nobody will ever encode that work: the worker
    /// that took the count to zero fails the run rather than idling on a
    /// queue it will not serve while the scaler blocks behind it.
    pub serving_workers: Arc<Vec<AtomicUsize>>,
    /// Who is working on each rung right now: its scalers, plus a worker for
    /// as long as it holds one of the rung's chunks.
    ///
    /// **Seeded at 1, not 0** — a setup guard released by
    /// [`Self::release_setup_guard`] once every scaler has been spawned. The
    /// finalizers are spawned before the scalers, and a finalizer's first act
    /// is to break out of its wait if the count is already zero; with a 0 seed
    /// the runtime only had to schedule a finalizer before its scaler's
    /// `fetch_add` for that rung to conclude "nobody is working on me" and
    /// return empty. Load-dependent, so it hid on a two-rung three-second clip
    /// and showed up on a five-rung four-minute one.
    pub active_workers: Arc<Vec<AtomicUsize>>,
    pub rung_done: Arc<Vec<Notify>>,
    /// Set by each finalizer before its terminal report, so the periodic
    /// progress reporter stops printing `Running` for a rung that is done.
    pub finalized: Arc<Vec<AtomicBool>>,
    /// The stop signal for everything on this ladder — see [`AbortSignal`].
    pub abort: Arc<AbortSignal>,
}

/// How a run is stopped before it is finished — by a caller's cancel, or by
/// the first failure — without leaving anything behind.
///
/// Every part of the ladder that can block does so on one of two things: a
/// queue (scalers push, workers pop) or the `rung_done` notify (finalizers).
/// So stopping is: raise the flag, close and empty every queue, wake every
/// finalizer. A scaler mid-push gets `false` and returns, which drops its
/// frame receiver, which is what ends its pump. A worker sees the flag at the
/// top of its loop and returns its lease. A finalizer wakes, sees the flag and
/// returns without merging. Nothing waits on a wake-up that will not come, and
/// the queued frames — up to the whole byte budget — go with the run instead
/// of living on in a task nobody is joining.
///
/// This is not generic over the contribution type on purpose: the thing that
/// waits on the run ([`drain`]) knows the finalizer's output type but not the
/// worker's, and it is the one that has to be able to pull the plug.
pub(super) struct AbortSignal {
    flag: AtomicBool,
    queues: Vec<Arc<SegmentChunkQueue>>,
    rung_done: Arc<Vec<Notify>>,
}

impl AbortSignal {
    /// Whether the run has been told to stop.
    pub fn is_aborted(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Stop the run: flag, close and empty every queue, wake every finalizer.
    /// Idempotent.
    pub fn abort(&self) {
        self.flag.store(true, Ordering::Release);
        for q in &self.queues {
            q.close();
            while q.try_pop().is_some() {}
        }
        for n in self.rung_done.iter() {
            n.notify_waiters();
            n.notify_one();
        }
    }
}

impl<T: Send + 'static> Ladder<T> {
    /// Per-rung state for `rungs`, with each queue's depth derived from the
    /// byte budget rather than a fixed count (see `queue_capacity_for`).
    pub fn new(rungs: &[Rung], frames_per_chunk: u32) -> Self {
        let n = rungs.len();
        let queues: Vec<Arc<SegmentChunkQueue>> = rungs
            .iter()
            .map(|r| {
                let depth = queue_capacity_for(r.width, r.height, frames_per_chunk, n);
                Arc::new(SegmentChunkQueue::new(depth))
            })
            .collect();
        let rung_done: Arc<Vec<Notify>> = Arc::new((0..n).map(|_| Notify::new()).collect());
        Self {
            abort: Arc::new(AbortSignal {
                flag: AtomicBool::new(false),
                queues: queues.clone(),
                rung_done: Arc::clone(&rung_done),
            }),
            queues,
            frames_encoded: (0..n).map(|_| Arc::new(AtomicU64::new(0))).collect(),
            bytes_encoded: (0..n).map(|_| Arc::new(AtomicU64::new(0))).collect(),
            rung_invariants: (0..n).map(|_| Arc::new(RwLock::new(None))).collect(),
            contributions: Arc::new((0..n).map(|_| Mutex::new(Vec::new())).collect()),
            serving_workers: Arc::new((0..n).map(|_| AtomicUsize::new(0)).collect()),
            active_workers: Arc::new((0..n).map(|_| AtomicUsize::new(1)).collect()),
            rung_done,
            finalized: Arc::new((0..n).map(|_| AtomicBool::new(false)).collect()),
        }
    }

    /// Whether the run has been stopped early (cancelled, or failed elsewhere).
    /// A finalizer that wakes to this returns without merging.
    pub fn is_aborted(&self) -> bool {
        self.abort.is_aborted()
    }

    /// Wait until rung `idx` is finished: nothing is working on it *and*
    /// nothing can be handed out — queue closed, queue empty. Also returns,
    /// early, when the run is aborted; check [`Self::is_aborted`] after.
    ///
    /// A count of zero alone used to mean "finished", which was true when a
    /// rung had one worker for its whole life. A ladder worker takes one chunk
    /// at a time from whichever rung is furthest behind, so this rung's count
    /// legitimately returns to zero between chunks — every time the last card
    /// working on it moves to another rung. Finalising there takes whatever
    /// segments exist so far and calls the rung done, which the coverage check
    /// then rejects.
    pub async fn wait_rung_finished(&self, idx: usize) {
        loop {
            let notified = self.rung_done[idx].notified();
            if self.is_aborted() {
                return;
            }
            let queue_drained = self.queues[idx].is_closed() && self.queues[idx].depth() == 0;
            if self.active_workers[idx].load(Ordering::Acquire) == 0 && queue_drained {
                return;
            }
            notified.await;
        }
    }

    /// Everything the workers recorded for rung `idx`, leaving it empty.
    pub fn take_contributions(&self, idx: usize) -> Vec<T> {
        std::mem::take(&mut *self.contributions[idx].lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// Release the setup guard seeded into `active_workers`, once every scaler
    /// for every rung has bumped its own count. From here a zero means what
    /// the finalizer thinks it means. A rung whose scalers all finished during
    /// setup is why the notify is here too — without it that rung's finalizer
    /// would wait forever on a wake-up that already happened.
    pub fn release_setup_guard(&self) {
        for (idx, active) in self.active_workers.iter().enumerate() {
            if active.fetch_sub(1, Ordering::AcqRel) == 1 {
                self.rung_done[idx].notify_one();
            }
        }
    }

    fn worker_done_with(&self, rung_idx: usize) {
        if self.active_workers[rung_idx].fetch_sub(1, Ordering::AcqRel) == 1 {
            self.rung_done[rung_idx].notify_one();
        }
    }
}

/// Pre-flight: is there anything to encode on, and can it construct an
/// encoder for the job's codec?
///
/// Runs before a pump, a scaler or a finalizer exists — before a frame is
/// decoded — because that is the only point where failing costs nothing.
/// Once the pumps are up, a failure has to unwind blocking threads that
/// are parked on queues, and the run that first showed this (`--encode
/// family:intel` on a host with no Intel card, four chunks against a
/// two-deep queue) never did: it sat at `0/120 frames` until it was killed.
///
/// An **empty pool** — a policy that pinned silicon the host cannot serve —
/// is refused with the same error the pool's builder raises
/// ([`empty_pool_error`](super::gpu_policy::empty_pool_error)), so a caller
/// that built its own pool gets the same sentence as one that let
/// `gpu_pool_for_policy` refuse first.
///
/// A pool of cards answers by building one encoder **pinned to the pool's
/// first card** (index and vendor), exactly as a worker's lease will pin it:
/// fail fast with a clear error rather than after the orchestration is up —
/// and, on drivers that re-init a failed NVENC session badly, rather than by
/// hanging an uncancellable task. The probe used to run unpinned, which let
/// it pass on a card the policy had excluded. A software pool already *is*
/// the answer: the pool's builder handed out software slots because the
/// build has a software encoder for this codec, and constructing one just to
/// ask spins up a worker pool sized to the whole machine (32 threads on this
/// box) to encode nothing. So software is checked from the feature flags,
/// and the encoder is built once per unit of work, as it would be anyway.
pub(super) fn preflight_encoder(params: &MultiGpuParams<'_>, width: u32, height: u32) -> Result<()> {
    if params.gpu_pool.capacity() == 0 {
        return Err(super::gpu_policy::empty_pool_error(&params.host, params.encode, params.codec, params.output_pixel_format));
    }
    if params.gpu_pool.is_software() {
        if !super::gpu_policy::software_reaches_output(params.codec, params.output_pixel_format) {
            bail!(
                "the encode pool is software but this build's software {:?} encoder cannot produce \
                 {:?} (the software tier is `--features {}`)",
                params.codec,
                params.output_pixel_format,
                codec::encode::software_feature_for(params.codec)
            );
        }
        return Ok(());
    }
    let first = params.gpu_pool.snapshot_leases().into_iter().next();
    // At the output's format: a card that takes the codec only at 8 bits
    // passes an 8-bit probe and then fails the first 10-bit lease.
    let probe = codec::encode::EncoderConfig {
        width,
        height,
        frame_rate: params.frame_rate,
        gpu_index: first.as_ref().map(|slot| slot.index),
        gpu_vendor: first.as_ref().map(|slot| slot.vendor),
        codec: params.codec,
        pixel_format: params.output_pixel_format,
        ..Default::default()
    };
    codec::encode::select_encoder(probe, None).map_err(|e| {
        anyhow!(
            "no {:?} encoder could be started on the encode pool's first card ({}) at {width}x{height}: {e}",
            params.codec,
            first.map(|slot| format!("gpu {} {:?}, {}", slot.index, slot.vendor, slot.name)).unwrap_or_default(),
        )
    })?;
    Ok(())
}

/// How many decode ranges the default plan cuts per decoding card.
///
/// One range per card is an equal split, and an equal split is gated by the
/// slowest card: on devbox the A380's half of the decode took as long as the
/// A750 needed for the whole ladder. Cut finer, each card pulls the next
/// range when it is free and a fast card decodes more of the source. Finer
/// still costs a decoder construction and a demux pass to the range start per
/// range; four per card leaves the slow card's last range a small fraction of
/// the job, and the finish-time gate keeps it off the tail entirely.
pub(super) const RANGES_PER_CARD: usize = 4;

/// A card decodes in a split only if it is expected to be at least this
/// fraction as fast as the fastest decode-capable card.
///
/// Decoding is not free for the card that does it: every decoded frame comes
/// back over its PCIe link, the link its encode traffic also crosses. On
/// devbox the A380 (3.0 x2) decoding its share of the ranges slowed its own
/// encoding by more than its decode saved — the ladder took 5.8 s split
/// against 4.8 s with the A750 decoding everything. A card well behind the
/// fastest helps most by encoding only.
pub(super) const DECODE_PEER_RATIO: f64 = 0.6;

/// The decode for this job: the ranges the source is cut into, and the
/// devices that decode them. Each device runs one decode worker that pulls
/// the next range when it is free (see [`spawn_decode`]).
#[derive(Debug, Clone)]
pub(super) struct DecodePlan {
    pub ranges: Vec<DecodeRange>,
    /// One entry per decode worker: the GPU it decodes on, `None` for the
    /// software decoder (or an unpinned hardware one).
    pub devices: Vec<Option<u32>>,
}

/// The decode plan for this job under the spec's [`DecodePolicy`](crate::spec::DecodePolicy).
///
/// Only an un-spliced, untrimmed single input is split: a range is addressed
/// by demuxed sample index and its numbering assumes the source starts at
/// chunk 0, neither of which survives a trim window or a concat. Those decode
/// whole, as they always did — and so does anything `plan_decode_ranges`
/// cannot cut safely.
///
/// A whole-source decode runs on the pinned card if the policy pins one, else
/// on the decode-capable card expected to be fastest
/// ([`speed::fastest_of`]) — not the first one detected, which on devbox is
/// the slow A380.
pub(super) fn plan_decode(params: &MultiGpuParams<'_>, shape: LadderShape, capacity: usize) -> DecodePlan {
    let decode_gpus = params.decode_capable_gpus();
    let role = decode_role(&params.header.codec);
    // The slots a split decode runs on: each decode-capable card once. With
    // none, software decoders — on the CPU, several at once is real
    // parallelism — one per encode slot, as many as the split had before.
    // Software encode slots share the cores a split decode would also run
    // on, and the software encoders are far slower than the software
    // decoder, so a software pool splits nothing unless the policy names a
    // count (`ranges:N`) outright.
    let slots: Vec<Option<u32>> = if params.gpu_pool.is_software() {
        vec![None]
    } else if decode_gpus.is_empty() {
        vec![None; capacity.max(1)]
    } else {
        decode_peers(&role, &decode_gpus).into_iter().map(Some).collect()
    };
    let want = match params.decode {
        crate::spec::DecodePolicy::Auto if slots.len() > 1 => slots.len() * RANGES_PER_CARD,
        crate::spec::DecodePolicy::Auto => 1,
        other => other.ranges_for(slots.len()),
    };
    let whole = |why: Option<&str>| {
        if let Some(why) = why {
            tracing::info!("decode ranges: {why}; decoding whole");
        }
        let device = match params.decode.gpu_index() {
            Some(pinned) => Some(pinned),
            None if decode_gpus.len() > 1 => {
                let keys: Vec<DeviceKey> = decode_gpus.iter().map(|&g| DeviceKey::Gpu(g)).collect();
                let pick = speed::fastest_of(&role, &keys).map(|i| decode_gpus[i]);
                tracing::info!(
                    candidates = ?decode_gpus,
                    chosen = ?pick,
                    "whole-source decode on the decode-capable card expected to be fastest"
                );
                pick
            }
            None => decode_gpus.first().copied(),
        };
        DecodePlan { ranges: vec![DecodeRange::whole_source()], devices: vec![device] }
    };
    // A temporal filter (hqdn3d) makes each frame depend on the ones before
    // it, and a range starts with no history: split, the frames at every
    // range start would differ from a whole decode. One stream, one pump.
    // A frame-rate cap drops frames, so a sample's index no longer counts the
    // output frames before it and a range's first segment cannot be placed.
    if want > 1 && crate::decode_pump::decimation(params.header.info.frame_rate, Some(params.frame_rate)).is_some() {
        return whole(Some("the output frame rate is capped below the source's"));
    }
    if want > 1 && params.filters.is_stateful() {
        return whole(Some("the filter chain is temporal (frame history)"));
    }
    if want <= 1 || !params.spliced_clips.is_empty() {
        return whole(None);
    }
    let Some(ranges) = crate::decode_pump::plan_decode_ranges(
        &params.input,
        &params.header.codec,
        shape.frames_per_chunk,
        want,
        shape.overlap as u64,
    ) else {
        return whole(None);
    };
    // `ranges:N` keeps its meaning — N pumps at once, round-robin over the
    // cards, several on one card when N exceeds them; the default runs one
    // worker per card and lets each pull ranges.
    let devices: Vec<Option<u32>> = match params.decode {
        crate::spec::DecodePolicy::Ranges(n) => {
            (0..n.min(ranges.len()).max(1)).map(|i| params.range_decode_gpu_for(i, &decode_gpus)).collect()
        }
        _ => slots.into_iter().take(ranges.len()).collect(),
    };
    DecodePlan { ranges, devices }
}

/// The thread budget of each of `pumps` decode pumps running at once: this
/// job's share of the machine ([`crate::thread_budget::per_job`]) divided
/// among them.
fn filter_threads_per_pump(pumps: usize) -> usize {
    pump_share(crate::thread_budget::per_job(), pumps)
}

/// `job_threads` among `pumps` pumps: `0` (no narrower budget than the pump
/// sets itself, the job's share) for a lone pump, else an equal part, at
/// least one thread each.
fn pump_share(job_threads: usize, pumps: usize) -> usize {
    if pumps <= 1 {
        return 0;
    }
    (job_threads / pumps).max(1)
}

/// The decode-capable cards fit to share a split decode: those expected to be
/// at least [`DECODE_PEER_RATIO`] as fast as the fastest, in detection order.
fn decode_peers(role: &str, gpus: &[u32]) -> Vec<u32> {
    if gpus.len() < 2 {
        return gpus.to_vec();
    }
    let keys: Vec<DeviceKey> = gpus.iter().map(|&g| DeviceKey::Gpu(g)).collect();
    let board = SpeedBoard::new(speed::priors_for(role, &keys));
    peers_at(&board, gpus)
}

fn peers_at(board: &SpeedBoard, gpus: &[u32]) -> Vec<u32> {
    let best = (0..gpus.len()).map(|d| board.rate(d)).fold(0.0f64, f64::max);
    let peers: Vec<u32> =
        gpus.iter().enumerate().filter(|&(d, _)| board.rate(d) >= best * DECODE_PEER_RATIO).map(|(_, &g)| g).collect();
    if peers.len() < gpus.len() {
        tracing::info!(
            candidates = ?gpus,
            decoding = ?peers,
            "split decode: cards well behind the fastest encode only"
        );
    }
    peers
}

/// The speed record's role for decoding `codec`.
fn decode_role(codec: &str) -> String {
    format!("decode:{}", codec.to_ascii_lowercase())
}

/// What the decode workers share: the ranges, the next one to hand out, and
/// the devices' speeds.
struct RangeDispatch {
    ranges: Vec<DecodeRange>,
    /// The frames each range contributes, for the speed record and the gate.
    frames: Vec<u64>,
    state: Mutex<(usize, SpeedBoard)>,
    epoch: std::time::Instant,
    role: String,
    keys: Vec<DeviceKey>,
}

enum NextRange {
    Take(usize),
    Wait,
    Done,
}

impl RangeDispatch {
    fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }

    /// The next range for worker `w`, if it should take one now.
    fn next_for(&self, w: usize) -> NextRange {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let (next, board) = &mut *state;
        if *next >= self.ranges.len() {
            board.retire(w);
            return NextRange::Done;
        }
        let units = self.frames[*next].max(1) as f64;
        let remaining: f64 = self.frames[*next..].iter().map(|&f| f.max(1) as f64).sum();
        let now = self.now();
        if !board.should_take(w, units, remaining, now, |_| true) {
            return NextRange::Wait;
        }
        board.start(w, units, now);
        *next += 1;
        NextRange::Take(*next - 1)
    }

    fn finished(&self, w: usize, range_idx: usize, elapsed: f64) {
        let units = self.frames[range_idx].max(1) as f64;
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(rate) = state.1.finish(w, units, elapsed) {
            speed::record_rate(&self.role, self.keys[w], rate);
        }
    }

    fn retire(&self, w: usize) {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).1.retire(w);
    }
}

/// The decode: one worker per device of the plan, each pulling the next
/// range when it is free and running it — a decode pump on its device,
/// fanning out to one scaler per rung, numbering that range's chunks from its
/// first frame — until none is left. Returns the worker tasks; each reports
/// the frames it decoded.
///
/// # Who decodes what
///
/// The ranges are handed out in source order, so chunks reach the encoders
/// roughly in order, and a range goes to whichever worker asks first —
/// unless the finish-time gate ([`SpeedBoard::should_take`]) says the other
/// workers would be done with every remaining range before this one could
/// finish the next: then it waits, and near the end the slow card simply
/// stops taking ranges. The plan's boundaries are fixed before anything is
/// decoded, and each range carries its own chunk numbering and lead-in, so
/// which card decoded which range changes nothing in the output — not a
/// frame, not a timestamp, not a chunk.
///
/// # Memory
///
/// A worker runs one range at a time and waits for that range's scalers to
/// hand their last chunk to the queues before it takes another, so what is in
/// flight is bounded exactly as before: per worker, a pump's channels and one
/// chunk in the making per rung; the queues hold the rest, inside their byte
/// budget.
///
/// # Ending
///
/// A rung's queue is fed by every range's scaler and closed by whichever
/// finishes last — closing on the first exit would drain the workers while
/// other ranges were still feeding, losing every chunk after the first
/// range's end. Only the last range's scaler may mark a chunk final: a middle
/// range also finishes on a short chunk — its boundary — and marking that
/// final would end the stream mid-video.
pub(super) fn spawn_decode<T: Send + 'static>(
    params: &MultiGpuParams<'_>,
    plan: DecodePlan,
    rungs: &[Rung],
    shape: LadderShape,
    ladder: &Ladder<T>,
) -> JoinSet<Result<u64>> {
    let DecodePlan { ranges, devices } = plan;
    let multi_range = ranges.len() > 1;
    let total_frames = params.total_input_frames;
    let frames: Vec<u64> = ranges
        .iter()
        .enumerate()
        .map(|(i, r)| r.frames(ranges.get(i + 1).map(|n| n.start_frame), total_frames))
        .collect();
    let keys: Vec<DeviceKey> = devices.iter().map(|&d| DeviceKey::of_gpu(d)).collect();
    let role = decode_role(&params.header.codec);
    let dispatch = Arc::new(RangeDispatch {
        state: Mutex::new((0, SpeedBoard::new(speed::priors_for(&role, &keys)))),
        ranges: ranges.clone(),
        frames,
        epoch: std::time::Instant::now(),
        role,
        keys,
    });
    let rung_producers: Vec<Arc<AtomicUsize>> =
        (0..rungs.len()).map(|_| Arc::new(AtomicUsize::new(ranges.len()))).collect();
    let scaler_template: Vec<crate::rung_scaler::RungScalerConfig> = rungs
        .iter()
        .enumerate()
        .map(|(idx, rung)| crate::rung_scaler::RungScalerConfig {
            rung_idx: idx,
            target_width: rung.width,
            target_height: rung.height,
            placement: rung.placement,
            frames_per_chunk: shape.frames_per_chunk,
            overlap: shape.overlap,
            first_segment_idx: 0,
            is_final_range: false,
            lead_in: 0,
        })
        .collect();

    let mut tasks: JoinSet<Result<u64>> = JoinSet::new();
    let workers = devices.len();
    for (w, device) in devices.iter().copied().enumerate() {
        let clips = params.clip_sources_for(device);
        let dispatch = Arc::clone(&dispatch);
        let producers = rung_producers.clone();
        let scalers = scaler_template.clone();
        let queues = ladder.queues.clone();
        let active = Arc::clone(&ladder.active_workers);
        let rung_done = Arc::clone(&ladder.rung_done);
        let abort = Arc::clone(&ladder.abort);
        let rt = tokio::runtime::Handle::current();
        tasks.spawn(async move {
            tokio::task::spawn_blocking(move || {
                let mut decoded = 0u64;
                let outcome = loop {
                    if abort.is_aborted() {
                        break Ok(decoded);
                    }
                    let range_idx = match dispatch.next_for(w) {
                        NextRange::Done => break Ok(decoded),
                        NextRange::Wait => {
                            std::thread::sleep(DECODE_WAIT_POLL);
                            continue;
                        }
                        NextRange::Take(i) => i,
                    };
                    let range = dispatch.ranges[range_idx];
                    let started = std::time::Instant::now();
                    let job = RangeJob {
                        range_idx,
                        range,
                        is_final: range_idx + 1 == dispatch.ranges.len(),
                        multi_range,
                        frames_per_chunk: shape.frames_per_chunk,
                        concurrent: workers,
                    };
                    match run_range(&job, &clips, &scalers, &queues, &producers, &active, &rung_done, &rt) {
                        Ok(n) => decoded += n,
                        Err(e) => break Err(e.context(format!("decode range {range_idx} on {device:?}"))),
                    }
                    let elapsed = started.elapsed().as_secs_f64();
                    dispatch.finished(w, range_idx, elapsed);
                    if multi_range {
                        tracing::info!(
                            range = range_idx,
                            gpu_index = ?device,
                            frames = dispatch.frames[range_idx],
                            seconds = format!("{elapsed:.2}"),
                            "decode range done"
                        );
                    }
                };
                dispatch.retire(w);
                outcome
            })
            .await
            .map_err(|e| anyhow!("decode worker {w} join error: {e}"))
            .and_then(|r| r)
        });
    }

    if multi_range {
        tracing::info!(
            rungs = rungs.len(),
            ranges = ranges.len(),
            decoders = ?devices,
            boundaries = ?ranges.iter().map(|r| r.start_sample).collect::<Vec<_>>(),
            "range-parallel decode engaged — each card pulls the next stretch of the source when it is free",
        );
    } else {
        tracing::info!(rungs = rungs.len(), decoder = ?devices.first().copied().flatten(), "shared decode pump engaged (one decode for the whole ladder)");
    }
    tasks
}

/// How long a decode worker waits when the gate tells it another card would
/// finish the next range sooner.
const DECODE_WAIT_POLL: std::time::Duration = std::time::Duration::from_millis(10);

/// One range for one worker.
struct RangeJob {
    range_idx: usize,
    range: DecodeRange,
    is_final: bool,
    multi_range: bool,
    frames_per_chunk: u32,
    /// Decode workers running at once: they share the machine's threads,
    /// for a software decoder and for the filters alike.
    concurrent: usize,
}

/// Decode one range: its scalers (one per rung, each on a blocking thread),
/// then the pump on this thread, then wait for the scalers to hand over
/// their last chunk. The first error wins: the pump's, else a scaler's.
#[allow(clippy::too_many_arguments)]
fn run_range(
    job: &RangeJob,
    clips: &[crate::decode_pump::ClipSource],
    scalers: &[crate::rung_scaler::RungScalerConfig],
    queues: &[Arc<SegmentChunkQueue>],
    producers: &[Arc<AtomicUsize>],
    active: &Arc<Vec<AtomicUsize>>,
    rung_done: &Arc<Vec<Notify>>,
    rt: &tokio::runtime::Handle,
) -> Result<u64> {
    // `plan_decode_ranges` guarantees the boundary is a multiple of
    // `frames_per_chunk`, so this division is exact.
    let first_segment_idx = (job.range.start_frame / u64::from(job.frames_per_chunk)) as usize;
    let mut senders = Vec::with_capacity(scalers.len());
    let mut handles = Vec::with_capacity(scalers.len());
    for (idx, template) in scalers.iter().enumerate() {
        let (tx, rx) = mpsc::channel(FANOUT_CHANNEL_CAPACITY);
        senders.push(tx);
        let cfg = crate::rung_scaler::RungScalerConfig {
            first_segment_idx,
            is_final_range: job.is_final,
            lead_in: job.range.lead_in as usize,
            ..template.clone()
        };
        let queue = Arc::clone(&queues[idx]);
        let producers = Arc::clone(&producers[idx]);
        let active = Arc::clone(active);
        let rung_done = Arc::clone(rung_done);
        let rt_scaler = rt.clone();
        // Counted before the scaler exists, so the rung's finalizer cannot
        // see it idle while its queue is still being fed.
        active[idx].fetch_add(1, Ordering::AcqRel);
        handles.push(rt.spawn_blocking(move || {
            let result = crate::rung_scaler::run_rung_scaler_blocking_shared(cfg, rx, queue, rt_scaler, producers);
            if active[idx].fetch_sub(1, Ordering::AcqRel) == 1 {
                rung_done[idx].notify_one();
            }
            result.with_context(|| format!("scaler for rung {idx}"))
        }));
    }
    let mut clips = clips.to_vec();
    if job.multi_range {
        for clip in clips.iter_mut() {
            clip.cfg.sample_range = job.range.sample_range();
            clip.cfg.software_share = job.concurrent;
        }
    }
    // Several pumps at once share the machine: each one's filters (the
    // denoisers split frames over threads) get its share of it.
    let pumped = codec::filter::with_thread_budget(filter_threads_per_pump(job.concurrent), || {
        crate::decode_pump::run_spliced_decode_pump_blocking(clips, senders, rt.clone())
    });
    let mut scaler_error = None;
    for handle in handles {
        let joined = rt.block_on(handle).map_err(|e| anyhow!("scaler join error: {e}")).and_then(|r| r);
        if let Err(e) = joined
            && scaler_error.is_none()
        {
            scaler_error = Some(e);
        }
    }
    let frames = pumped.with_context(|| format!("decode pump (range {})", job.range_idx))?;
    if let Some(e) = scaler_error {
        return Err(e);
    }
    Ok(frames)
}

/// The per-rung worker config for one card: the rung's own knobs, the job's
/// output format, and this worker's lease. `output_dir` is the rung's
/// directory under the output root, keyed by label; the single-file path never
/// writes there.
fn rung_worker_config(
    ctx: &WorkerCtx,
    rung_idx: usize,
    rung: &Rung,
    lease: &GpuLease,
    rung_invariant: Arc<RwLock<Option<RungCodecInvariant>>>,
) -> EncoderWorkerConfig {
    EncoderWorkerConfig {
        overrides: rung.quality.overrides,
        rung_idx,
        codec: ctx.codec,
        width: rung.width,
        height: rung.height,
        frame_rate: ctx.frame_rate,
        quality: rung.quality.crf.unwrap_or(codec::encode::AUTO_FROM_TARGET),
        speed_preset: rung.quality.speed_preset.unwrap_or(codec::encode::AUTO_FROM_TARGET),
        target: rung.quality.target,
        tier: rung.quality.tier,
        // A software lease is a share of the CPU: its thread budget goes to
        // the encoder, and the encoder is asked for by name so a chunk never
        // re-runs the hardware probes the pool already ran. A card leaves
        // `threads` at 0 (the encoder does not run on host threads) and
        // `backend` unset (the vendor pin steers the chain).
        threads: lease.threads(),
        gpu_index: lease.gpu_index(),
        gpu_vendor: lease.vendor(),
        backend: if lease.is_software() { codec::encode::software_backend_for(ctx.codec) } else { None },
        output_color_metadata: ctx.output_color_metadata,
        output_pixel_format: ctx.output_pixel_format,
        constant_qp: ctx.constant_qp,
        timescale: ctx.timescale,
        per_frame_ticks: ctx.per_frame_ticks,
        keyframe_interval: ctx.keyframe_interval,
        segment_target_ticks: ctx.segment_target_ticks,
        base_decode_time_offset: ctx.video_delay_ticks,
        output_dir: ctx.output_root.join(format!("video/{}", rung.label)),
        rung_invariant,
    }
}

/// What a worker does with one chunk of one rung.
///
/// Called with that rung's config, the chunk, the worker's `init_written`
/// flag for the rung (only the CMAF path reads it), the worker's encoder
/// session pool (only the single-file path uses it — one pool per worker,
/// kept across every chunk and every rung it serves), the rung's shared
/// frame and byte counters, and the progress channel. `Done(T)` is recorded
/// against the rung; `Rejected` puts the chunk back and strikes the rung off
/// this worker's list.
pub(super) trait EncodeUnit<T>: Send + Sync + 'static {
    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        cfg: &EncoderWorkerConfig,
        chunk: SegmentChunk,
        init_written: &mut bool,
        sessions: &mut EncoderSessionPool,
        frames_encoded: &AtomicU64,
        bytes_encoded: &AtomicU64,
        progress_tx: &mpsc::Sender<u64>,
    ) -> Result<UnitOutcome<T>>;
}

impl<T, F> EncodeUnit<T> for F
where
    F: Fn(
            &EncoderWorkerConfig,
            SegmentChunk,
            &mut bool,
            &mut EncoderSessionPool,
            &AtomicU64,
            &AtomicU64,
            &mpsc::Sender<u64>,
        ) -> Result<UnitOutcome<T>>
        + Send
        + Sync
        + 'static,
{
    fn encode(
        &self,
        cfg: &EncoderWorkerConfig,
        chunk: SegmentChunk,
        init_written: &mut bool,
        sessions: &mut EncoderSessionPool,
        frames_encoded: &AtomicU64,
        bytes_encoded: &AtomicU64,
        progress_tx: &mpsc::Sender<u64>,
    ) -> Result<UnitOutcome<T>> {
        self(cfg, chunk, init_written, sessions, frames_encoded, bytes_encoded, progress_tx)
    }
}

/// The encode side's finish-time gate: which worker should take the next
/// chunk, given how fast each has shown itself to be and how much of the
/// ladder is left.
///
/// Work is measured in pixels encoded (frames × width × height), so a 1080p
/// chunk weighs what it costs against a 360p one and the rate of a card is one
/// number across every rung it serves. What is left is every rung's frames
/// not yet taken by a worker, chunks not decoded yet included — the gate is
/// about the tail of the whole job, not of what happens to be queued.
pub(super) struct EncodeGate {
    board: Mutex<SpeedBoard>,
    epoch: std::time::Instant,
    role: String,
    keys: Vec<DeviceKey>,
    /// Frames of each rung no worker has taken yet.
    remaining: Vec<AtomicU64>,
    /// Pixels per frame of each rung.
    pixels: Vec<f64>,
    frames_per_chunk: u64,
    /// Which rungs each worker may still take: its schedule, less the rungs
    /// it has refused on the codec invariant.
    serves: Mutex<Vec<Vec<bool>>>,
}

impl EncodeGate {
    pub(super) fn new(
        codec: codec::frame::VideoCodec,
        keys: Vec<DeviceKey>,
        rungs: &[Rung],
        serves: &[Vec<usize>],
        total_frames: u64,
        frames_per_chunk: u32,
    ) -> Self {
        let role = format!("encode:{codec:?}").to_ascii_lowercase();
        let board = SpeedBoard::new(speed::priors_for(&role, &keys));
        tracing::info!(
            devices = ?keys,
            expected_rates = ?(0..board.len()).map(|d| format!("{:.2}", board.rate(d))).collect::<Vec<_>>(),
            "encode scheduling: the slower devices step aside at the tail of the job"
        );
        Self {
            board: Mutex::new(board),
            epoch: std::time::Instant::now(),
            role,
            keys,
            remaining: rungs.iter().map(|_| AtomicU64::new(total_frames)).collect(),
            pixels: rungs.iter().map(|r| f64::from(r.width) * f64::from(r.height)).collect(),
            frames_per_chunk: u64::from(frames_per_chunk.max(1)),
            serves: Mutex::new(
                serves.iter().map(|s| (0..rungs.len()).map(|r| s.contains(&r)).collect()).collect(),
            ),
        }
    }

    fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }

    /// Should worker `slot` take a chunk of `rung` now? With the total frame
    /// count unknown (`0`) there is no tail to see, and the answer is yes.
    fn should_take(&self, slot: usize, rung: usize) -> bool {
        let left = self.remaining[rung].load(Ordering::Acquire);
        let units = self.frames_per_chunk.min(left).max(1) as f64 * self.pixels[rung];
        let remaining: f64 = self
            .remaining
            .iter()
            .zip(&self.pixels)
            .map(|(r, px)| r.load(Ordering::Acquire) as f64 * px)
            .sum();
        if remaining <= 0.0 {
            return true;
        }
        let serves = self.serves.lock().unwrap_or_else(|p| p.into_inner());
        let board = self.board.lock().unwrap_or_else(|p| p.into_inner());
        board.should_take(slot, units, remaining.max(units), self.now(), |o| serves[o][rung])
    }

    /// Worker `slot` took `chunk` of `rung`; returns its weight in work units.
    fn took(&self, slot: usize, rung: usize, chunk: &SegmentChunk) -> f64 {
        let left = &self.remaining[rung];
        let mut now = left.load(Ordering::Acquire);
        while let Err(seen) =
            left.compare_exchange_weak(now, now.saturating_sub(chunk.keep as u64), Ordering::AcqRel, Ordering::Acquire)
        {
            now = seen;
        }
        let units = chunk.frames.len().max(1) as f64 * self.pixels[rung];
        self.board.lock().unwrap_or_else(|p| p.into_inner()).start(slot, units, self.now());
        units
    }

    fn done(&self, slot: usize, units: f64, elapsed: f64) {
        if let Some(rate) = self.board.lock().unwrap_or_else(|p| p.into_inner()).finish(slot, units, elapsed) {
            speed::record_rate(&self.role, self.keys[slot], rate);
            // Whatever the codec: what a serial job asks for when it picks a
            // card (`serial_target`).
            speed::record_rate(speed::ANY_ENCODE_ROLE, self.keys[slot], rate);
        }
    }

    /// Worker `slot` handed `keep` frames of `rung` back without encoding
    /// them, and will not take that rung again.
    fn refused(&self, slot: usize, rung: usize, keep: usize) {
        self.remaining[rung].fetch_add(keep as u64, Ordering::AcqRel);
        self.board.lock().unwrap_or_else(|p| p.into_inner()).abandon(slot);
        self.serves.lock().unwrap_or_else(|p| p.into_inner())[slot][rung] = false;
    }

    fn retire(&self, slot: usize) {
        self.board.lock().unwrap_or_else(|p| p.into_inner()).retire(slot);
    }
}

/// Claim a lease per GPU and start the workers. Returns the worker tasks and
/// how many started. Fails only when the pool hands out nothing at all —
/// which [`preflight_encoder`] refuses first, before anything is spawned;
/// this is the same refusal for a caller that skipped it.
pub(super) async fn spawn_workers<T: Send + 'static>(
    params: &MultiGpuParams<'_>,
    ctx: &WorkerCtx,
    rungs: &[Rung],
    shape: LadderShape,
    ladder: &Arc<Ladder<T>>,
    encode: Arc<dyn EncodeUnit<T>>,
) -> Result<(JoinSet<(usize, Result<()>)>, usize)> {
    let capacity = params.gpu_pool.capacity().max(1);
    let mut worker_tasks: JoinSet<(usize, Result<()>)> = JoinSet::new();
    let mut leases = Vec::with_capacity(capacity);
    for slot in 0..capacity {
        match Arc::clone(&params.gpu_pool).claim().await {
            Some(l) => leases.push(l),
            None if slot == 0 => {
                // The pool is empty, and the pool's builder already decided
                // that software was not an answer here — say why, by name.
                return Err(super::gpu_policy::empty_pool_error(&params.host, params.encode, params.codec, params.output_pixel_format));
            }
            None => break,
        }
    }
    let workers = leases.len();
    let software = leases.iter().filter(|l| l.is_software()).count();
    // Which rungs each worker may take from, and so how many workers each
    // rung has — the count a refusal draws down (see `serving_workers`).
    let serves_by_slot: Vec<Vec<usize>> = (0..workers)
        .map(|slot| {
            if params.encode.pins_rungs() {
                (0..rungs.len()).filter(|idx| idx % workers == slot).collect()
            } else {
                (0..rungs.len()).collect()
            }
        })
        .collect();
    for (idx, count) in ladder.serving_workers.iter().enumerate() {
        count.store(serves_by_slot.iter().filter(|s| s.contains(&idx)).count(), Ordering::Release);
    }
    let gate = (workers > 1).then(|| {
        Arc::new(EncodeGate::new(
            params.codec,
            leases.iter().map(|l| DeviceKey::of_gpu(l.gpu_index())).collect(),
            rungs,
            &serves_by_slot,
            params.total_input_frames,
            shape.frames_per_chunk,
        ))
    });
    for ((slot, lease), serves) in leases.into_iter().enumerate().zip(serves_by_slot) {
        spawn_ladder_worker(
            ctx,
            slot,
            rungs,
            serves,
            lease,
            Arc::clone(ladder),
            Arc::clone(&encode),
            gate.clone(),
            &mut worker_tasks,
        );
    }
    if software > 0 {
        tracing::info!(
            ladder_workers = workers,
            software_leases = software,
            threads_per_lease = ?params.gpu_pool.software_threads(),
            rungs = rungs.len(),
            encode = ?params.encode,
            backend = ?codec::encode::software_backend_for(ctx.codec),
            "ladder workers started on SOFTWARE leases — no GPU can encode this codec in this build; \
             each worker runs one software encoder at a time on its thread share",
        );
    } else {
        tracing::info!(
            ladder_workers = workers,
            rungs = rungs.len(),
            encode = ?params.encode,
            "ladder workers started — each serves every rung it is scheduled for, so a card idles only when that work is done",
        );
    }
    Ok((worker_tasks, workers))
}

/// One worker, every rung it serves.
///
/// Holds a single GPU lease for its lifetime — so the one-encoder-per-GPU
/// invariant is untouched — and repeatedly takes the next chunk from
/// whichever of its rungs is furthest behind.
///
/// # Why "furthest behind" rather than "smallest rung first"
///
/// The shared decode pump stalls when *any* rung queue is full. Serving the
/// fullest queue is what keeps the decode moving: it attacks the rung closest
/// to blocking everyone. Preferring the cheapest rung would publish early
/// quality sooner and then wedge the pump behind the rung nobody was serving.
///
/// # When it stops
///
/// Only when every queue it serves is closed *and* empty. A worker that finds
/// nothing waits a beat and asks again rather than exiting, because "this rung
/// has nothing right now" is the normal state of a rung whose scaler is
/// mid-chunk; exiting on it would retire a card with work still coming.
///
/// # A rung this card cannot serve
///
/// The first packet a rung sees fixes its codec invariant, and a card of a
/// vendor whose sequence header disagrees on a mandatory field can never
/// contribute to that rung — the disagreement is a property of the silicon,
/// not of the chunk. Such a chunk goes back to the head of the queue for
/// another card, and the rung is struck off this worker's list, so it does not
/// spin re-building encoders against a rung it will be refused by every time.
/// When the worker striking it off was the **last** one serving that rung,
/// nothing will ever encode what is queued for it, and "wait a beat and ask
/// again" would be forever: the run fails there, naming the rung.
///
/// # When its encoder cannot be built
///
/// A unit that fails — the encoder would not construct, the session would
/// not reset and rebuild, a driver said no — ends this worker with the
/// error, and [`drain`] ends the run with it: the other workers are stopped
/// and the leases returned. A pool whose every lease fails to build an
/// encoder is therefore a run that fails with the first such error, not one
/// that waits.
#[allow(clippy::too_many_arguments)]
fn spawn_ladder_worker<T: Send + 'static>(
    ctx: &WorkerCtx,
    slot: usize,
    rungs: &[Rung],
    serves: Vec<usize>,
    lease: GpuLease,
    ladder: Arc<Ladder<T>>,
    encode: Arc<dyn EncodeUnit<T>>,
    gate: Option<Arc<EncodeGate>>,
    worker_tasks: &mut JoinSet<(usize, Result<()>)>,
) {
    let gpu_index = lease.gpu_index();
    let gpu_vendor = lease.vendor();
    // "gpu 0 (Nvidia)" or "software slot 3 (4 threads)": the log lines below
    // are the answer to "what is actually running this chunk", and on a
    // CPU-only host `gpu_index=None` alone would not say.
    let lease_label = lease.kind().to_string();
    let rungs_labels: Vec<String> = rungs.iter().map(|r| r.label.clone()).collect();
    let configs: Vec<EncoderWorkerConfig> = rungs
        .iter()
        .enumerate()
        .map(|(idx, rung)| rung_worker_config(ctx, idx, rung, &lease, Arc::clone(&ladder.rung_invariants[idx])))
        .collect();

    let body = async move {
        // The per-frame progress channel is a formality here: the shared
        // counters are what the reporter reads. It exists so the worker is
        // never backpressured by nobody listening.
        let (progress_tx, mut progress_rx) = mpsc::channel::<u64>(32);
        let drain = tokio::spawn(async move { while progress_rx.recv().await.is_some() {} });

        let ladder_for_worker = Arc::clone(&ladder);
        let lease_label_after = lease_label.clone();
        let blocking = tokio::task::spawn_blocking(move || -> Result<()> {
            let ladder = ladder_for_worker;
            let mut init_written: Vec<bool> = vec![false; configs.len()];
            let mut refused: HashSet<usize> = HashSet::new();
            // This worker's encoder session, kept between chunks and reset
            // rather than rebuilt while consecutive chunks share a rung. One
            // per worker, so one live session per lease — the
            // one-encoder-per-GPU invariant is exactly as true as before.
            let mut sessions = EncoderSessionPool::new();
            loop {
                // Stopped from outside — cancelled, or another part of the run
                // failed. Return the lease now rather than after draining what
                // is left in the queues.
                if ladder.is_aborted() {
                    tracing::info!(slot, gpu_index = ?gpu_index, lease = %lease_label, "ladder worker stopping: run aborted");
                    break;
                }
                // Pick the rung closest to blocking the pump.
                let mut best: Option<(usize, usize)> = None;
                for &idx in &serves {
                    if refused.contains(&idx) {
                        continue;
                    }
                    let depth = ladder.queues[idx].depth();
                    if depth == 0 {
                        continue;
                    }
                    if best.is_none_or(|(_, d)| depth > d) {
                        best = Some((idx, depth));
                    }
                }

                let Some((rung_idx, _)) = best else {
                    // Nothing anywhere. Finished only if nothing can arrive.
                    if serves.iter().all(|&idx| ladder.queues[idx].is_closed() && ladder.queues[idx].depth() == 0) {
                        break;
                    }
                    std::thread::sleep(IDLE_POLL);
                    continue;
                };

                // Near the end of the job, a slower card leaves the chunk to
                // a faster one that would be done with everything left before
                // this card could finish this one (see `EncodeGate`).
                if let Some(gate) = &gate
                    && !gate.should_take(slot, rung_idx)
                {
                    std::thread::sleep(IDLE_POLL);
                    continue;
                }

                // Counted *before* the pop and held across the encode, so this
                // rung's finalizer cannot decide the rung is finished while a
                // chunk of it is on its way to a card. Counted after the pop,
                // as it was, there was a window in which the queue was closed
                // and empty and the count still zero: a finalizer woken by
                // another worker's last chunk then merged without this one,
                // and the coverage check refused the rung ("pushed 3
                // segments, 2 came back").
                ladder.active_workers[rung_idx].fetch_add(1, Ordering::AcqRel);
                let Some(chunk) = ladder.queues[rung_idx].try_pop() else {
                    // Another worker took it between the look and the grab.
                    ladder.worker_done_with(rung_idx);
                    continue;
                };
                let segment_idx = chunk.segment_idx;
                let keep = chunk.keep;
                let units = gate.as_ref().map(|g| g.took(slot, rung_idx, &chunk));
                let started = std::time::Instant::now();
                let outcome = encode.encode(
                    &configs[rung_idx],
                    chunk,
                    &mut init_written[rung_idx],
                    &mut sessions,
                    &ladder.frames_encoded[rung_idx],
                    &ladder.bytes_encoded[rung_idx],
                    &progress_tx,
                );
                match outcome {
                    Ok(UnitOutcome::Done(contribution)) => {
                        if let (Some(gate), Some(units)) = (&gate, units) {
                            gate.done(slot, units, started.elapsed().as_secs_f64());
                        }
                        // Which card did which chunk of which rung — the line
                        // that answers "what is actually happening" on a fleet.
                        tracing::info!(rung_idx, gpu_index = ?gpu_index, lease = %lease_label, chunk = segment_idx, "rung chunk done");
                        // Recorded before the count drops, so the finalizer
                        // woken by that drop sees this contribution.
                        ladder.contributions[rung_idx]
                            .lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .push(contribution);
                        ladder.worker_done_with(rung_idx);
                    }
                    Ok(UnitOutcome::Rejected { chunk, diff }) => {
                        tracing::warn!(
                            rung_idx,
                            gpu_index = ?gpu_index,
                            lease = %lease_label,
                            gpu_vendor = ?gpu_vendor,
                            rejected_chunk = chunk.segment_idx,
                            diff = %diff,
                            "codec invariant mismatch — returning the chunk for another card \
                             and leaving this rung to them",
                        );
                        ladder.queues[rung_idx].push_front(chunk);
                        refused.insert(rung_idx);
                        if let Some(gate) = &gate {
                            gate.refused(slot, rung_idx, keep);
                        }
                        let last_server = ladder.serving_workers[rung_idx].fetch_sub(1, Ordering::AcqRel) == 1;
                        ladder.worker_done_with(rung_idx);
                        if last_server {
                            return Err(anyhow!(
                                "rung {} ({}): every ladder worker has refused it on the codec \
                                 invariant (last: {lease_label}: {diff}); nothing is left to \
                                 encode its chunks",
                                rung_idx,
                                rungs_labels[rung_idx],
                            ));
                        }
                    }
                    Err(e) => {
                        ladder.worker_done_with(rung_idx);
                        return Err(e);
                    }
                }
            }
            if let Some(gate) = &gate {
                gate.retire(slot);
            }
            // The evidence for the session pool: on a run that used to build
            // one encoder per chunk, `built + reused` is the chunk count and
            // `reused` is the saving.
            let stats = sessions.stats();
            tracing::info!(
                slot,
                gpu_index,
                built = stats.built,
                reused = stats.reused,
                evicted = stats.evicted,
                reset_unsupported = stats.reset_unsupported,
                reset_failed = stats.reset_failed,
                "ladder worker encoder sessions: built vs reused"
            );
            Ok(())
        });

        let status: Result<()> = match blocking.await {
            Ok(Ok(())) => {
                tracing::info!(slot, gpu_index = ?gpu_index, lease = %lease_label_after, "ladder worker exited cleanly");
                Ok(())
            }
            Ok(Err(e)) => Err(e),
            Err(e) => Err(anyhow!("ladder worker join error: {e}")),
        };
        // `progress_tx` moved into the blocking task and dropped with it, which
        // is what ends the drain.
        let _ = drain.await;
        drop(lease);
        (slot, status)
    };
    worker_tasks.spawn(body);
}

/// The tasks a run waits on, and how many are still running.
pub(super) struct Running<R> {
    pub pumps: JoinSet<Result<u64>>,
    pub scalers: JoinSet<(usize, Result<usize>)>,
    pub workers: JoinSet<(usize, Result<()>)>,
    pub finalizer_rx: mpsc::Receiver<(usize, Result<Option<R>>)>,
    pub finalizers_remaining: usize,
    /// The ladder's stop signal, pulled on cancel and on the first failure.
    pub abort: Arc<AbortSignal>,
    /// The caller's cancel signal, if it has one: `true` means stop.
    pub cancel: Option<watch::Receiver<bool>>,
}

/// Wait for every pump, scaler, worker and finalizer; the first error wins.
/// The caller stops its progress reporter (and awaits the finalizer handles on
/// success) around this.
///
/// # Stopping early
///
/// On the first failure, or when the caller's `cancel` signal turns true, the
/// run is [aborted](AbortSignal::abort) and this waits for the **workers** to
/// return before returning the error — they hold the GPU leases, and the next
/// job's `claim()` must not find them still held by a run that is over. That
/// wait is bounded by one unit of work: a worker checks the flag between
/// units, not inside one. Pumps and scalers stop on their own once the queues
/// are closed and are not waited for; they hold no leases and drop what they
/// were carrying as they go.
///
/// A cancel comes back as [`Cancelled`](super::Cancelled), so a caller can
/// tell "asked to stop" from "failed" without reading the message.
pub(super) async fn drain<R>(mut run: Running<R>) -> Result<Vec<Option<R>>> {
    let mut completed: Vec<Option<R>> = (0..run.finalizers_remaining).map(|_| None).collect();
    let mut pumps_remaining = run.pumps.len();
    let mut scalers_remaining = run.scalers.len();
    let mut workers_remaining = run.workers.len();
    let mut finalizers_remaining = run.finalizers_remaining;

    // A cancel signal that is already raised, or absent, is handled here so the
    // select below only has to watch for a change.
    let mut cancel = run.cancel.take();
    if cancel.as_ref().is_some_and(|c| *c.borrow()) {
        return Err(stop(&mut run, super::Cancelled.into()).await);
    }

    while pumps_remaining > 0 || scalers_remaining > 0 || workers_remaining > 0 || finalizers_remaining > 0 {
        let outcome: Result<()> = tokio::select! {
            biased;
            changed = watch_cancel(&mut cancel) => match changed {
                // The sender is gone: nobody can cancel us any more.
                Err(()) => { cancel = None; Ok(()) }
                Ok(()) => Err(super::Cancelled.into()),
            },
            p = run.pumps.join_next(), if pumps_remaining > 0 => match p {
                Some(Ok(Ok(frames))) => { pumps_remaining -= 1; tracing::info!(frames, pumps_remaining, "decode pump finished"); Ok(()) }
                Some(Ok(Err(e))) => Err(anyhow!("decode pump failed: {e:#}")),
                Some(Err(je)) => Err(anyhow!("pump join error: {je}")),
                None => { pumps_remaining = 0; Ok(()) }
            },
            s = run.scalers.join_next(), if scalers_remaining > 0 => match s {
                Some(Ok((idx, Ok(chunks)))) => { tracing::debug!(idx, chunks, "scaler finished"); scalers_remaining -= 1; Ok(()) }
                Some(Ok((idx, Err(e)))) => Err(anyhow!("scaler {idx} failed: {e:#}")),
                Some(Err(je)) => Err(anyhow!("scaler join error: {je}")),
                None => { scalers_remaining = 0; Ok(()) }
            },
            w = run.workers.join_next(), if workers_remaining > 0 => match w {
                Some(Ok((slot, Ok(())))) => { tracing::debug!(slot, "ladder worker finished"); workers_remaining -= 1; Ok(()) }
                Some(Ok((slot, Err(e)))) => Err(anyhow!("ladder worker {slot} failed: {e:#}")),
                Some(Err(je)) => Err(anyhow!("worker join error: {je}")),
                None => { workers_remaining = 0; Ok(()) }
            },
            f = run.finalizer_rx.recv(), if finalizers_remaining > 0 => match f {
                Some((idx, Ok(opt))) => { completed[idx] = opt; finalizers_remaining -= 1; Ok(()) }
                Some((idx, Err(e))) => Err(anyhow!("finalizer for rung {idx} failed: {e:#}")),
                None => { finalizers_remaining = 0; Ok(()) }
            },
        };
        if let Err(e) = outcome {
            return Err(stop(&mut run, e).await);
        }
    }
    Ok(completed)
}

/// Resolve when the cancel signal turns true; `Err(())` when its sender has
/// gone. Pending forever while there is no signal, so the select arm is
/// simply never taken.
async fn watch_cancel(cancel: &mut Option<watch::Receiver<bool>>) -> std::result::Result<(), ()> {
    match cancel {
        None => std::future::pending().await,
        Some(rx) => loop {
            if *rx.borrow_and_update() {
                return Ok(());
            }
            if rx.changed().await.is_err() {
                return Err(());
            }
        },
    }
}

/// Abort the run and wait for the workers — the lease holders — to return,
/// then hand back the error that ended it. The finalizer channel is left
/// alone: a finalizer that wakes to the abort sends nothing anyone reads.
async fn stop<R>(run: &mut Running<R>, why: anyhow::Error) -> anyhow::Error {
    if why.is::<super::Cancelled>() {
        tracing::info!("ladder run cancelled; stopping the workers");
    } else {
        tracing::warn!(error = %format!("{why:#}"), "ladder run failed; stopping the workers");
    }
    run.abort.abort();
    let mut worker_failure: Option<String> = None;
    while let Some(joined) = run.workers.join_next().await {
        match joined {
            Ok((slot, Ok(()))) => tracing::debug!(slot, "ladder worker returned its lease"),
            Ok((slot, Err(e))) => {
                tracing::debug!(slot, error = %format!("{e:#}"), "ladder worker ended with an error while stopping");
                if worker_failure.is_none() && !e.is::<super::Cancelled>() {
                    worker_failure = Some(format!("ladder worker {slot} failed: {e:#}"));
                }
            }
            Err(je) => tracing::debug!(%je, "ladder worker join error while stopping"),
        }
    }
    // A worker that fails drops its hold on the rung, and the rung's
    // finalizer, woken by that, can report the hole the failure left ("chunk
    // coverage incomplete") before the worker's own task has been joined —
    // which then read as the cause. The worker's error is the cause: say it
    // first. (devbox: a two-chunk H.265 file failed with only the coverage
    // message.)
    match worker_failure {
        Some(cause) if !why.is::<super::Cancelled>() && !format!("{why:#}").starts_with("ladder worker") => {
            anyhow!("{cause} (and then: {why:#})")
        }
        _ => why,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    /// A lone pump's filters keep the machine; several share it.
    #[test]
    fn pumps_share_the_filter_threads() {
        assert_eq!(pump_share(32, 1), 0);
        assert_eq!(pump_share(32, 2), 16);
        assert_eq!(pump_share(16, 3), 5, "a job's half of 32 cores among three pumps");
        assert_eq!(pump_share(4, 16), 1);
    }
    use codec::frame::{ColorSpace, PixelFormat, VideoFrame};
    use std::time::Duration;

    fn frame(idx: u64) -> VideoFrame {
        let mut data = vec![idx as u8; 16 * 16];
        data.extend(vec![128u8; 8 * 8]);
        data.extend(vec![128u8; 8 * 8]);
        VideoFrame::new(Bytes::from(data), 16, 16, PixelFormat::Yuv420p, ColorSpace::Bt709, idx)
    }

    fn chunk(idx: usize) -> SegmentChunk {
        SegmentChunk { segment_idx: idx, frames: vec![frame(0), frame(1)], lead_in: 0, keep: 2, is_final: false }
    }

    const SHAPE: LadderShape = LadderShape { frames_per_chunk: 2, overlap: 0 };

    fn two_rungs() -> Vec<Rung> {
        vec![Rung::new(64, 64), Rung::new(32, 32)]
    }

    /// Aborting closes and empties every queue and wakes every finalizer —
    /// nothing is left holding frames or waiting on a notify that will not
    /// come.
    #[tokio::test]
    async fn abort_closes_empties_and_wakes() {
        let ladder: Ladder<()> = Ladder::new(&two_rungs(), 2);
        assert!(ladder.queues[0].push(chunk(0)).await);
        assert!(ladder.queues[0].push(chunk(1)).await);
        assert!(ladder.queues[1].push(chunk(0)).await);
        assert_eq!(ladder.queues[0].depth(), 2);
        assert!(!ladder.is_aborted());

        // A finalizer parked on a rung nobody has finished (the setup guard is
        // still held, so `active_workers` is 1 and the queue is open).
        let ladder = Arc::new(ladder);
        let waiter = {
            let l = Arc::clone(&ladder);
            tokio::spawn(async move { l.wait_rung_finished(0).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished(), "finalizer must wait while the rung is open");

        ladder.abort.abort();

        assert!(ladder.is_aborted());
        for q in &ladder.queues {
            assert!(q.is_closed());
            assert_eq!(q.depth(), 0, "abort must drop what was queued");
        }
        // A push after the abort is refused (the scaler's exit condition).
        assert!(!ladder.queues[0].push(chunk(2)).await);
        tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .expect("finalizer must be woken by the abort")
            .unwrap();
    }

    /// A cancel signal that is already raised stops the run before it waits
    /// on anything, and the error's root cause is `Cancelled`.
    #[tokio::test]
    async fn drain_returns_cancelled_when_signal_is_already_raised() {
        let ladder: Ladder<()> = Ladder::new(&two_rungs(), 2);
        let (_tx, mut rx) = watch::channel(false);
        // Raise it through a sender we keep alive so the receiver sees `true`.
        let tx = _tx;
        tx.send(true).unwrap();
        rx.mark_unchanged();
        let (_ftx, finalizer_rx) = mpsc::channel::<(usize, Result<Option<()>>)>(2);
        let run = Running {
            pumps: JoinSet::new(),
            scalers: JoinSet::new(),
            workers: JoinSet::new(),
            finalizer_rx,
            finalizers_remaining: 2,
            abort: Arc::clone(&ladder.abort),
            cancel: Some(rx),
        };
        let err = drain(run).await.expect_err("must not complete");
        assert!(err.is::<super::super::Cancelled>(), "root cause must be Cancelled, got {err:#}");
        assert!(ladder.is_aborted(), "cancel must abort the ladder");
    }

    /// A cancel raised while the run is waiting stops it, and the workers
    /// still in flight are joined before the error comes back.
    #[tokio::test]
    async fn drain_stops_on_cancel_and_joins_workers() {
        let ladder: Ladder<()> = Ladder::new(&two_rungs(), 2);
        let (tx, rx) = watch::channel(false);
        let (_ftx, finalizer_rx) = mpsc::channel::<(usize, Result<Option<()>>)>(2);
        // A "worker" that only returns once the run has been aborted — the
        // shape of a real worker checking the flag at the top of its loop.
        let mut workers: JoinSet<(usize, Result<()>)> = JoinSet::new();
        let abort = Arc::clone(&ladder.abort);
        let worker_saw_abort = Arc::new(AtomicBool::new(false));
        let saw = Arc::clone(&worker_saw_abort);
        workers.spawn(async move {
            while !abort.is_aborted() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            saw.store(true, Ordering::Release);
            (0, Ok(()))
        });
        let run = Running {
            pumps: JoinSet::new(),
            scalers: JoinSet::new(),
            workers,
            finalizer_rx,
            finalizers_remaining: 2,
            abort: Arc::clone(&ladder.abort),
            cancel: Some(rx),
        };
        let handle = tokio::spawn(drain(run));
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!handle.is_finished(), "must be waiting on the finalizers");
        tx.send(true).unwrap();
        let err = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("cancel must end the run")
            .unwrap()
            .expect_err("must not complete");
        assert!(err.is::<super::super::Cancelled>(), "got {err:#}");
        assert!(worker_saw_abort.load(Ordering::Acquire), "the worker must have been joined after the abort");
    }

    /// The first failure aborts the ladder too, so the other parts of the run
    /// stop instead of blocking on queues nobody will drain.
    #[tokio::test]
    async fn drain_aborts_ladder_on_failure() {
        let ladder: Ladder<()> = Ladder::new(&two_rungs(), 2);
        assert!(ladder.queues[0].push(chunk(0)).await);
        let (_ftx, finalizer_rx) = mpsc::channel::<(usize, Result<Option<()>>)>(2);
        let mut scalers: JoinSet<(usize, Result<usize>)> = JoinSet::new();
        scalers.spawn(async { (0, Err(anyhow!("scaler exploded"))) });
        let run = Running {
            pumps: JoinSet::new(),
            scalers,
            workers: JoinSet::new(),
            finalizer_rx,
            finalizers_remaining: 2,
            abort: Arc::clone(&ladder.abort),
            cancel: None,
        };
        let err = drain(run).await.expect_err("must fail");
        assert!(!err.is::<super::super::Cancelled>());
        assert!(format!("{err:#}").contains("scaler exploded"));
        assert!(ladder.is_aborted());
        assert!(ladder.queues[0].is_closed());
        assert_eq!(ladder.queues[0].depth(), 0);
    }

    // ---- an empty pool, and a pool that empties mid-job ----

    use super::super::test_support::{params_with_pool, within};
    use crate::gpu_pool::GpuPool;
    use crate::spec::{EncodePolicy, GpuFamily};
    use codec::frame::VideoCodec;

    fn ctx() -> WorkerCtx {
        WorkerCtx {
            codec: VideoCodec::H264,
            frame_rate: 30.0,
            output_color_metadata: Default::default(),
            output_pixel_format: PixelFormat::Yuv420p,
            timescale: 30_000,
            per_frame_ticks: 1000,
            keyframe_interval: 30,
            segment_target_ticks: 30_000,
            output_root: std::env::temp_dir(),
            constant_qp: false,
            video_delay_ticks: 0,
        }
    }

    /// An encode unit that answers every chunk the same way.
    fn unit(answer: impl Fn(SegmentChunk) -> Result<UnitOutcome<()>> + Send + Sync + 'static) -> Arc<dyn EncodeUnit<()>> {
        Arc::new(
            move |_cfg: &EncoderWorkerConfig,
                  chunk: SegmentChunk,
                  _init: &mut bool,
                  _sessions: &mut EncoderSessionPool,
                  _frames: &AtomicU64,
                  _bytes: &AtomicU64,
                  _tx: &mpsc::Sender<u64>| answer(chunk),
        )
    }

    /// One software-leased worker serving both rungs, running `unit`, and
    /// the run's handle over it — the shape of a real ladder once the pumps
    /// and scalers are out of the picture. The pool comes back so the test
    /// can check the lease was returned.
    fn one_worker_run(ladder: &Arc<Ladder<()>>, unit: Arc<dyn EncodeUnit<()>>) -> (Running<()>, Arc<GpuPool>) {
        let pool = Arc::new(GpuPool::software(1, 1));
        let lease = pool.try_claim().expect("one slot");
        // What `spawn_workers` records: this one worker serves both rungs.
        for count in ladder.serving_workers.iter() {
            count.store(1, Ordering::Release);
        }
        let mut workers: JoinSet<(usize, Result<()>)> = JoinSet::new();
        spawn_ladder_worker(&ctx(), 0, &two_rungs(), vec![0, 1], lease, Arc::clone(ladder), unit, None, &mut workers);
        ladder.release_setup_guard();
        let (ftx, finalizer_rx) = mpsc::channel::<(usize, Result<Option<()>>)>(2);
        drop(ftx);
        let run = Running {
            pumps: JoinSet::new(),
            scalers: JoinSet::new(),
            workers,
            finalizer_rx,
            finalizers_remaining: 2,
            abort: Arc::clone(&ladder.abort),
            cancel: None,
        };
        (run, pool)
    }

    /// A pool that "becomes empty mid-job": every lease is held, and the
    /// encoder behind it cannot be built. The run ends with that reason —
    /// within a bound — and the lease comes back to the pool.
    #[test]
    fn a_worker_whose_encoder_cannot_be_built_ends_the_run_with_the_reason() {
        within(
            Duration::from_secs(10),
            "a run whose only lease cannot build an encoder waited instead of failing",
            || async {
                let ladder: Arc<Ladder<()>> = Arc::new(Ladder::new(&two_rungs(), 2));
                assert!(ladder.queues[0].push(chunk(0)).await);
                let (run, pool) =
                    one_worker_run(&ladder, unit(|_| Err(anyhow!("creating encoder for chunk: the driver said no"))));
                let err = drain(run).await.expect_err("must fail");
                let msg = format!("{err:#}");
                assert!(msg.contains("ladder worker 0 failed"), "{msg}");
                assert!(msg.contains("the driver said no"), "{msg}");
                assert!(ladder.is_aborted());
                assert!(pool.try_claim().is_some(), "the failed worker's lease must be back in the pool");
            },
        );
    }

    /// A worker's failure is the error the run reports, even when the
    /// finalizer's complaint about the chunk it never delivered arrives first.
    #[test]
    fn a_workers_failure_is_reported_ahead_of_the_hole_it_left() {
        within(Duration::from_secs(10), "a failed run did not stop", || async {
            let ladder: Arc<Ladder<()>> = Arc::new(Ladder::new(&two_rungs(), 2));
            let mut workers: JoinSet<(usize, Result<()>)> = JoinSet::new();
            workers.spawn(async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                (1, Err(anyhow!("QSV said no")))
            });
            let (ftx, finalizer_rx) = mpsc::channel::<(usize, Result<Option<()>>)>(2);
            ftx.send((0, Err(anyhow!("chunk coverage incomplete")))).await.unwrap();
            let run = Running {
                pumps: JoinSet::new(),
                scalers: JoinSet::new(),
                workers,
                finalizer_rx,
                finalizers_remaining: 2,
                abort: Arc::clone(&ladder.abort),
                cancel: None,
            };
            let msg = format!("{:#}", drain(run).await.expect_err("must fail"));
            assert!(msg.starts_with("ladder worker 1 failed: QSV said no"), "{msg}");
            assert!(msg.contains("chunk coverage incomplete"), "{msg}");
        });
    }

    /// The last worker able to serve a rung strikes it off: nothing will
    /// ever encode what that rung has queued, so waiting for it is waiting
    /// forever. The run fails naming the rung instead.
    #[test]
    fn a_rung_every_worker_has_refused_ends_the_run() {
        within(
            Duration::from_secs(10),
            "a run with a rung no worker will serve waited for it instead of failing",
            || async {
                let ladder: Arc<Ladder<()>> = Arc::new(Ladder::new(&two_rungs(), 2));
                assert!(ladder.queues[1].push(chunk(0)).await);
                let (run, pool) = one_worker_run(
                    &ladder,
                    unit(|chunk| Ok(UnitOutcome::Rejected { chunk, diff: "profile 100 vs 77".into() })),
                );
                let err = drain(run).await.expect_err("must fail");
                let msg = format!("{err:#}");
                assert!(msg.contains("rung 1 (32p): every ladder worker has refused it"), "{msg}");
                assert!(msg.contains("profile 100 vs 77"), "{msg}");
                assert!(ladder.is_aborted());
                assert!(pool.try_claim().is_some());
            },
        );
    }

    /// The preflight is the first thing the HLS and single-file runners
    /// call: an empty pool is refused there, by name, before a pump exists.
    #[test]
    fn preflight_refuses_an_empty_pool_by_name() {
        let rungs = two_rungs();
        let params = params_with_pool(&rungs, Arc::new(GpuPool::new(&[])), EncodePolicy::Family(GpuFamily::Intel), VideoCodec::H264);
        let err = preflight_encoder(&params, 64, 64).expect_err("an empty pool has nothing to preflight");
        let msg = format!("{err:#}");
        assert!(msg.contains("no encoder matches `--encode family:intel` for H.264 on this host"), "{msg}");
        assert!(msg.contains("Present: synth-0 (gpu 0, NVIDIA, encodes H.264)"), "{msg}");
    }

    /// And the lease claim says the same thing for a caller that skipped
    /// the preflight.
    #[test]
    fn spawn_workers_refuses_an_empty_pool_by_name() {
        within(
            Duration::from_secs(10),
            "claiming leases from an empty pool waited instead of refusing",
            || async {
                let rungs = two_rungs();
                let params =
                    params_with_pool(&rungs, Arc::new(GpuPool::new(&[])), EncodePolicy::SingleGpu(Some(9)), VideoCodec::H265);
                let ladder: Arc<Ladder<()>> = Arc::new(Ladder::new(&rungs, 2));
                let err = spawn_workers(&params, &ctx(), &rungs, SHAPE, &ladder, unit(|_| Ok(UnitOutcome::Done(()))))
                    .await
                    .expect_err("nothing to lease");
                let msg = format!("{err:#}");
                assert!(msg.contains("no encoder matches `--encode gpu:9` for H.265 on this host: there is no gpu 9."), "{msg}");
                // The host named is the one the params carry: the refusal
                // never waited on detecting and probing this machine's cards.
                assert!(
                    msg.contains("Present: synth-0 (gpu 0, NVIDIA, encodes H.265); synth-1 (gpu 1, AMD, cannot encode H.265 in this build)."),
                    "{msg}"
                );
            },
        );
    }

    /// `spawn_workers` records how many workers serve each rung — every
    /// worker for a ladder-scheduled plan, one for a rung-pinned one — which
    /// is the count a refusal draws down.
    #[test]
    fn spawn_workers_counts_how_many_serve_each_rung() {
        within(
            Duration::from_secs(10),
            "starting and stopping two workers on a two-slot pool did not finish",
            || async {
                let rungs = two_rungs();
                for (policy, expect) in [(EncodePolicy::AllGpus, vec![2usize, 2]), (EncodePolicy::PerRung, vec![1, 1])] {
                    let params = params_with_pool(&rungs, Arc::new(GpuPool::software(2, 1)), policy, VideoCodec::H264);
                    let ladder: Arc<Ladder<()>> = Arc::new(Ladder::new(&rungs, 2));
                    let (mut workers, started) =
                        spawn_workers(&params, &ctx(), &rungs, SHAPE, &ladder, unit(|_| Ok(UnitOutcome::Done(())))).await.unwrap();
                    assert_eq!(started, 2, "{policy:?}");
                    let counts: Vec<usize> = ladder.serving_workers.iter().map(|c| c.load(Ordering::Acquire)).collect();
                    assert_eq!(counts, expect, "{policy:?}");
                    ladder.abort.abort();
                    while workers.join_next().await.is_some() {}
                }
            },
        );
    }

    // ---- the decode: many ranges, several workers, one output ----

    /// An H.264 MP4 of `frames` test-pattern pictures, an IDR every `gop`.
    fn synth_source(frames: u64, gop: u32) -> Bytes {
        use crate::synth;
        let cfg = synth::H264 { gop, ..synth::H264::new(64, 48, 25) };
        let coded = synth::encode_h264(&cfg, (0..frames).map(|t| synth::test_pattern(64, 48, t, h26x::ChromaFormat::Yuv420)));
        Bytes::from(synth::mp4(&coded, 64, 48, 25, None, None))
    }

    /// What a run handed the encoders, by chunk: `(segment, lead-in, keep,
    /// final, [(pts, pixels)])`, in segment order.
    type Handed = Vec<(usize, usize, usize, bool, Vec<(u64, Bytes)>)>;

    /// Run the decode of `plan` and collect every chunk every rung's queue
    /// receives. `None` when this build has no H.264 decoder to run.
    fn decode_through_ladder(input: &Bytes, plan: DecodePlan, shape: LadderShape) -> Option<Vec<Handed>> {
        let header = container::streaming::demux_streaming(input).ok()?.header().clone();
        let input = input.clone();
        within(Duration::from_secs(60), "a ranged decode through the ladder did not finish", move || async move {
            let rungs = two_rungs();
            let mut params =
                params_with_pool(&rungs, Arc::new(GpuPool::software(1, 1)), EncodePolicy::AllGpus, VideoCodec::H264);
            params.input = input;
            params.total_input_frames = header.info.total_frames;
            params.header = header;
            params.frame_rate = 25.0;
            let ladder: Ladder<()> = Ladder::new(&rungs, shape.frames_per_chunk);
            let mut decode = spawn_decode(&params, plan, &rungs, shape, &ladder);
            let queues = ladder.queues.clone();
            let consumer = tokio::spawn(async move {
                let mut handed: Vec<Handed> = vec![Vec::new(); queues.len()];
                loop {
                    let mut idle = true;
                    for (r, q) in queues.iter().enumerate() {
                        while let Some(c) = q.try_pop() {
                            idle = false;
                            let frames = c.frames.iter().map(|f| (f.pts, f.data.clone())).collect();
                            handed[r].push((c.segment_idx, c.lead_in, c.keep, c.is_final, frames));
                        }
                    }
                    if idle && queues.iter().all(|q| q.is_closed() && q.depth() == 0) {
                        break;
                    }
                    if idle {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                }
                for h in &mut handed {
                    h.sort_by_key(|c| c.0);
                }
                handed
            });
            while let Some(done) = decode.join_next().await {
                if let Err(e) = done.expect("decode worker join") {
                    eprintln!("SKIP: no H.264 decoder in this build ({e:#})");
                    ladder.abort.abort();
                    let _ = consumer.await;
                    return None;
                }
            }
            Some(consumer.await.expect("consumer"))
        })
    }

    /// Equal, or a readable account of the first difference.
    #[allow(clippy::type_complexity)]
    fn assert_same(split: &[Handed], whole: &[Handed]) {
        let shape = |h: &[Handed]| -> Vec<Vec<(usize, usize, usize, bool, Vec<u64>)>> {
            h.iter()
                .map(|r| r.iter().map(|c| (c.0, c.1, c.2, c.3, c.4.iter().map(|f| f.0).collect())).collect())
                .collect()
        };
        assert_eq!(shape(split), shape(whole), "chunks (segment, lead-in, keep, final, pts) differ");
        for (r, (a, b)) in split.iter().zip(whole).enumerate() {
            for (ca, cb) in a.iter().zip(b) {
                for (i, (fa, fb)) in ca.4.iter().zip(&cb.4).enumerate() {
                    assert!(fa.1 == fb.1, "rung {r} chunk {} frame {i} (pts {}): pixels differ", ca.0, fa.0);
                }
            }
        }
    }

    fn whole_plan() -> DecodePlan {
        DecodePlan { ranges: vec![DecodeRange::whole_source()], devices: vec![None] }
    }

    /// Cut fine and pulled by three workers at once, the source reaches the
    /// encoders as exactly the chunks one whole-source decode makes — same
    /// segments, same frames, same timestamps, same final flag — for HLS's
    /// segment-sized chunks with no lead-in.
    #[test]
    fn many_ranges_on_several_workers_hand_over_the_whole_decode() {
        let input = synth_source(120, 10);
        let shape = LadderShape { frames_per_chunk: 10, overlap: 0 };
        let ranges = crate::decode_pump::plan_decode_ranges(&input, "h264", 10, 8, 0).expect("splits");
        assert!(ranges.len() >= 6, "{ranges:?}");
        let Some(whole) = decode_through_ladder(&input, whole_plan(), shape) else { return };
        let split = decode_through_ladder(&input, DecodePlan { ranges, devices: vec![None; 3] }, shape).expect("ran whole");
        assert_eq!(whole[0].len(), 12);
        assert_same(&split, &whole);
    }

    /// With a chunk lead-in (single-file chunk-and-stitch), every range
    /// after the first decodes from a keyframe a GOP early and hands its first
    /// chunk the same lead-in a whole decode gives it: the output does not
    /// depend on where the source was cut.
    #[test]
    fn ranges_carry_the_lead_in_a_whole_decode_gives_their_first_chunk() {
        let input = synth_source(160, 10);
        let shape = LadderShape { frames_per_chunk: 20, overlap: 10 };
        let ranges = crate::decode_pump::plan_decode_ranges(&input, "h264", 20, 8, 10).expect("splits");
        assert!(ranges.len() >= 4, "{ranges:?}");
        for r in &ranges[1..] {
            assert_eq!(r.lead_in, 10, "{r:?}");
            assert_eq!(r.decode_from_sample, r.start_sample - 10, "{r:?}");
        }
        let Some(whole) = decode_through_ladder(&input, whole_plan(), shape) else { return };
        let split = decode_through_ladder(&input, DecodePlan { ranges, devices: vec![None; 2] }, shape).expect("ran whole");
        assert!(whole[0][1..].iter().all(|c| c.1 == 10), "a whole decode leads every chunk after the first in");
        assert_same(&split, &whole);
    }

    /// A card expected well under the fastest is left out of a split decode;
    /// cards of one speed all decode.
    #[test]
    fn a_much_slower_card_does_not_decode_in_a_split() {
        let a380_a750 = SpeedBoard::new(speed::normalise_priors(vec![0.43, 1.0], vec![None, None]));
        assert_eq!(peers_at(&a380_a750, &[0, 1]), vec![1]);
        let alike = SpeedBoard::new(speed::normalise_priors(vec![1.0, 0.9, 1.0], vec![None, None, None]));
        assert_eq!(peers_at(&alike, &[0, 1, 2]), vec![0, 1, 2]);
    }

    /// The default plan on a multi-card host cuts several ranges per card;
    /// a pinned or whole decode is one range on one card.
    #[test]
    fn the_default_plan_cuts_several_ranges_per_card() {
        let input = synth_source(400, 10);
        let rungs = two_rungs();
        let mut params = params_with_pool(&rungs, Arc::new(GpuPool::software(2, 1)), EncodePolicy::AllGpus, VideoCodec::H264);
        params.input = input;
        params.total_input_frames = 400;
        let shape = LadderShape { frames_per_chunk: 10, overlap: 0 };
        params.decode = crate::spec::DecodePolicy::Ranges(3);
        let plan = plan_decode(&params, shape, 2);
        assert_eq!(plan.ranges.len(), 3, "{plan:?}");
        assert_eq!(plan.devices.len(), 3);
        params.decode = crate::spec::DecodePolicy::Whole;
        let plan = plan_decode(&params, shape, 2);
        assert_eq!(plan.ranges, vec![DecodeRange::whole_source()]);
        assert_eq!(plan.devices.len(), 1);
        params.decode = crate::spec::DecodePolicy::SpecificGpu(5);
        assert_eq!(plan_decode(&params, shape, 2).devices, vec![Some(5)]);
    }
}
