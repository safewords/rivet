//! A thread budget for the work this crate splits across threads on its
//! own: the picture conversions' row bands (tonemap, SDR into HDR), the
//! denoisers' bands, and the decoders that size a pool when not told.
//!
//! Each of those defaults to the machine. A caller that runs several of them
//! at once — several jobs in one server, several decode pumps in one job —
//! sets a budget on the thread that does the work, with [`with_budget`], and
//! each one then takes at most that many threads. Nothing these split
//! changes with the thread count, only how long it takes.

use std::cell::Cell;

std::thread_local! {
    /// [`with_budget`]'s limit on this thread (0: none).
    static BUDGET: Cell<usize> = const { Cell::new(0) };
}

/// Run `f` with the work it does on this thread split over at most `threads`
/// threads, the calling thread among them (0: no budget, the machine). A
/// budget set inside another narrows it, never widens it, and the outer one
/// is back when `f` returns (or unwinds).
pub fn with_budget<R>(threads: usize, f: impl FnOnce() -> R) -> R {
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            BUDGET.with(|b| b.set(self.0));
        }
    }
    let outer = budget();
    let _restore = Restore(outer);
    BUDGET.with(|b| b.set(narrow(outer, threads)));
    f()
}

/// The budget on this thread (0: none).
pub fn budget() -> usize {
    BUDGET.with(Cell::get)
}

/// `threads`, held to this thread's budget, at least one.
pub fn cap(threads: usize) -> usize {
    narrow(budget(), threads).max(1)
}

/// The tighter of two limits, 0 being none.
fn narrow(a: usize, b: usize) -> usize {
    match (a, b) {
        (0, n) | (n, 0) => n,
        (a, b) => a.min(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_narrow_and_are_put_back() {
        assert_eq!(budget(), 0);
        assert_eq!(cap(16), 16);
        with_budget(4, || {
            assert_eq!(cap(16), 4);
            assert_eq!(cap(2), 2);
            with_budget(8, || {
                assert_eq!(budget(), 4, "an inner budget never widens")
            });
            with_budget(0, || assert_eq!(budget(), 4, "0 sets nothing"));
            with_budget(1, || assert_eq!(cap(16), 1));
            assert_eq!(budget(), 4);
        });
        assert_eq!(budget(), 0);
        let _ = std::panic::catch_unwind(|| with_budget(3, || panic!("unwinds")));
        assert_eq!(budget(), 0, "put back on unwind too");
        // Per thread: another thread sees no budget.
        with_budget(2, || {
            assert_eq!(std::thread::spawn(budget).join().unwrap(), 0)
        });
    }
}
