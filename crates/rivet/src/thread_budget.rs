//! The machine's threads, shared among the jobs this process runs at once.
//!
//! Several of the workspace's codecs size their worker pools to the whole
//! machine when not told otherwise — FLAC, ALAC, MP3 and Vorbis encode a
//! batch of frames across one thread per CPU, PNG's deflate compresses its
//! segments the same way, and the software AV1 / H.26x encoders take every
//! core at `threads: 0`. That is right for one job alone and an
//! oversubscription for a server or a batch running several: N jobs would
//! each start one worker per core. Each job holds a [`JobSlot`] while it
//! runs, and rivet hands every such encoder [`per_job`] threads: the
//! machine's parallelism divided by the jobs running now, never below one.
//! The encoders' output does not depend on their thread count, so this
//! changes only how the cores are shared.
//!
//! The share is recomputed each time a job (or a decode pump) asks for it,
//! so it follows the jobs as they start and end. `rivet serve` sets no limit
//! on its jobs by default, and each then gets the machine divided by the
//! jobs running when it asks. A server the operator limited to N jobs at
//! once (`rivet serve --jobs N`) says so with [`reserve_jobs`]: the share is
//! then reckoned against at least N jobs, so the first of N concurrent jobs
//! does not size its pools to the whole machine before the others arrive
//! (they would then oversubscribe it until it ended).

use std::sync::atomic::{AtomicUsize, Ordering};

static RUNNING: AtomicUsize = AtomicUsize::new(0);
/// [`reserve_jobs`]' count (0 and 1 alike: one job).
static RESERVED: AtomicUsize = AtomicUsize::new(0);

/// Reckon every job's share against at least `jobs` jobs from now on: the
/// number this process may run at once (0: no such number; the jobs running
/// now alone). Set once, at startup, by the server.
pub fn reserve_jobs(jobs: usize) {
    RESERVED.store(jobs, Ordering::SeqCst);
}

/// The jobs each share is reckoned against: those running now, or the
/// reserved count when that is more.
pub fn sharing_jobs(running: usize, reserved: usize) -> usize {
    running.max(reserved).max(1)
}

/// A running job's place in the budget, released when dropped.
#[must_use = "the job counts against the budget only while the slot is held"]
pub struct JobSlot(());

impl Drop for JobSlot {
    fn drop(&mut self) {
        RUNNING.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Count a job as running until the returned slot is dropped.
pub fn enter_job() -> JobSlot {
    RUNNING.fetch_add(1, Ordering::SeqCst);
    JobSlot(())
}

/// Jobs holding a [`JobSlot`] now.
pub fn running_jobs() -> usize {
    RUNNING.load(Ordering::SeqCst)
}

/// The threads the machine (or the container's CPU quota) offers.
pub fn parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// `parallelism` shared by `jobs` (none counts as one), at least one each.
pub fn share(parallelism: usize, jobs: usize) -> usize {
    (parallelism / jobs.max(1)).max(1)
}

/// The threads one job's encoder gets: the machine shared by the jobs
/// running now, or by the [reserved](reserve_jobs) count when that is more.
pub fn per_job() -> usize {
    share(
        parallelism(),
        sharing_jobs(running_jobs(), RESERVED.load(Ordering::SeqCst)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_jobs_hold_each_share_to_one_nth() {
        // A server limited to two jobs: one running alone still gets half.
        assert_eq!(share(32, sharing_jobs(1, 2)), 16);
        assert_eq!(share(32, sharing_jobs(2, 2)), 16);
        // More running than reserved (jobs outside the server): those count.
        assert_eq!(share(32, sharing_jobs(4, 2)), 8);
        // Nothing reserved: the jobs running now, as before.
        assert_eq!(share(32, sharing_jobs(1, 0)), 32);
        assert_eq!(share(32, sharing_jobs(0, 0)), 32);
        // N slots together never ask for more than the machine.
        for slots in 1..=40 {
            assert!(slots * share(32, sharing_jobs(slots, slots)) <= 32.max(slots));
        }
    }

    #[test]
    fn without_a_limit_the_share_follows_the_running_jobs() {
        // `rivet serve` with no job limit reserves nothing: each job's share
        // is the machine over the jobs running when it asks, recomputed as
        // they start and end.
        let reserved = 0;
        assert_eq!(
            share(32, sharing_jobs(1, reserved)),
            32,
            "a job alone has the machine"
        );
        assert_eq!(
            share(32, sharing_jobs(4, reserved)),
            8,
            "four at once: a quarter each"
        );
        assert_eq!(
            share(32, sharing_jobs(2, reserved)),
            16,
            "two of them ended: half each"
        );
        assert_eq!(share(32, sharing_jobs(64, reserved)), 1, "never below one");
        // However many run, those asking together fit the machine.
        for running in 1..=32 {
            assert!(running * share(32, sharing_jobs(running, reserved)) <= 32);
        }
    }

    #[test]
    fn the_machine_is_shared_and_never_below_one() {
        assert_eq!(share(32, 0), 32);
        assert_eq!(share(32, 1), 32);
        assert_eq!(share(32, 3), 10);
        assert_eq!(share(4, 8), 1);
        assert_eq!(share(1, 1), 1);
    }

    #[test]
    fn a_slot_counts_while_it_is_held() {
        // Other tests may hold slots concurrently; count relative to them.
        let a = enter_job();
        let b = enter_job();
        let with_two = running_jobs();
        assert!(with_two >= 2);
        assert!(per_job() <= share(parallelism(), 2));
        drop(b);
        drop(a);
    }
}
