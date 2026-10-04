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

use std::sync::atomic::{AtomicUsize, Ordering};

static RUNNING: AtomicUsize = AtomicUsize::new(0);

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
/// running now.
pub fn per_job() -> usize {
    share(parallelism(), running_jobs())
}

#[cfg(test)]
mod tests {
    use super::*;

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
