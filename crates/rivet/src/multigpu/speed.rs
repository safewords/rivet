//! How fast each device is, and what that means for who takes the next unit
//! of work.
//!
//! The ladder hands work out by *pulling*: a decode worker takes the next
//! source range, an encode worker the next chunk, as soon as it is free. On
//! cards of one speed that is all a scheduler needs — a card that finishes
//! early simply takes more. On cards of different speeds pulling alone has one
//! failure, at the end: the last units go to whoever asks first, and a card
//! three times slower than its neighbour that takes the last chunk finishes it
//! long after the fast card has gone idle. devbox measured exactly that — an
//! Arc A750 and an A380 together no faster than the A750 alone.
//!
//! So a worker asks before it takes: [`SpeedBoard::should_take`] says no when
//! the other devices, at the speeds they have shown, would finish *all* the
//! remaining work — this unit included — before this device could finish this
//! one unit. Mid-job the remaining work is large and every device takes;
//! near the end the slow one steps aside and the tail runs on the fast one.
//! The unit is never lost: the device that would finish it earliest always
//! says yes (see `should_take`), so declining is only ever waiting for a
//! better-placed device that is busy or about to ask.
//!
//! # Where the speeds come from
//!
//! - **Measured, this job**: every finished unit updates its device's rate
//!   (work per second, an exponential average).
//! - **Measured, this process**: rates are kept per (role, device) for the
//!   life of the process ([`record_rate`]), so a long-lived server's second
//!   job starts from what the first one learnt.
//! - **Expected**: a relative weight from the device's properties — its
//!   memory size and the PCIe link to the CPU ([`codec::gpu::pcie_report`]) —
//!   bounded so a prior alone never rates one card below
//!   [`MIN_PRIOR_RATIO`] of another: a prior decides the very first units
//!   and is replaced as soon as there is a measurement.
//!
//! Software encoders are devices like any other here: a software slot has a
//! rate, measured the same way, and is gated the same way.
//!
//! The board is pure: time comes in as seconds from the caller, so the
//! scheduling rules are tested by simulation, deterministically.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// No prior ever rates a device below this fraction of the best one: the
/// prior is a guess, and a guess that says "never" would keep the device from
/// ever being measured.
pub(crate) const MIN_PRIOR_RATIO: f64 = 0.35;

/// Weight of a new measurement in a device's running rate.
const EWMA_ALPHA: f64 = 0.4;

/// A device may take a unit that would finish up to this fraction later than
/// the alternative — measured rates are noisy, and waiting has a cost of its
/// own that the model does not see (a poll interval, a cold session).
const TAKE_SLACK: f64 = 0.05;

/// Past this many units of remaining work the job is nowhere near its tail,
/// and the answer is "take" without simulating the others.
const MAX_SIMULATED_UNITS: usize = 4096;

/// The role every encode measurement is also recorded under, whatever its
/// codec: the ranking a serial job's choice of card reads.
pub(crate) const ANY_ENCODE_ROLE: &str = "encode:any";

/// A device, as far as speed is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum DeviceKey {
    /// A GPU, by its global index.
    Gpu(u32),
    /// A software slot. Slots of one pool are alike, so they share a record.
    Software,
}

impl DeviceKey {
    pub(crate) fn of_gpu(index: Option<u32>) -> Self {
        index.map_or(DeviceKey::Software, DeviceKey::Gpu)
    }
}

fn process_rates() -> &'static Mutex<HashMap<(String, DeviceKey), f64>> {
    static RATES: OnceLock<Mutex<HashMap<(String, DeviceKey), f64>>> = OnceLock::new();
    RATES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The rate this process has measured for `key` in `role` (`"decode:h264"`,
/// `"encode:av1"`), in that role's units per second.
pub(crate) fn cached_rate(role: &str, key: DeviceKey) -> Option<f64> {
    process_rates().lock().unwrap_or_else(|p| p.into_inner()).get(&(role.to_string(), key)).copied()
}

/// Fold a measurement into the process-wide record.
pub(crate) fn record_rate(role: &str, key: DeviceKey, rate: f64) {
    if !(rate.is_finite() && rate > 0.0) {
        return;
    }
    let mut rates = process_rates().lock().unwrap_or_else(|p| p.into_inner());
    let entry = rates.entry((role.to_string(), key)).or_insert(rate);
    *entry = *entry * (1.0 - EWMA_ALPHA) + rate * EWMA_ALPHA;
}

/// A device's expected speed relative to others, from what can be read off
/// it without running anything: its memory (bigger cards are faster cards,
/// across vendors, far more often than not) and how narrow its PCIe link is
/// (every frame crosses it — the A380's 3.0 x2 against the A750's 4.0 x16).
/// `1.0` for a device the host does not describe.
pub(crate) fn static_weight(key: DeviceKey) -> f64 {
    // Unit tests describe their cards themselves; they never read the
    // machine's (which differ per host and would make them flaky).
    if cfg!(test) {
        return 1.0;
    }
    match key {
        DeviceKey::Software => 1.0,
        DeviceKey::Gpu(index) => codec::gpu::detect_gpus_cached()
            .iter()
            .find(|d| d.index == index)
            .map_or(1.0, |d| weight_from_properties(d.vram_mib, codec::gpu::pcie_report(d).map(|r| r.bottleneck))),
    }
}

/// [`static_weight`] from the numbers: memory as the square root of its size
/// against 8 GiB (an integrated GPU, with none of its own, counts as 2 GiB),
/// times a penalty for a link under 4 GB/s (x2 at 3.0) or 8 GB/s (x4 at 3.0).
pub(crate) fn weight_from_properties(vram_mib: u64, link: Option<codec::gpu::PcieLink>) -> f64 {
    let gib = if vram_mib == 0 { 2.0 } else { vram_mib as f64 / 1024.0 };
    let memory = (gib / 8.0).clamp(0.25, 4.0).sqrt();
    let link = match link.map(|l| l.gbytes_per_s()) {
        Some(bw) if bw < 4.0 => 0.5,
        Some(bw) if bw < 8.0 => 0.75,
        _ => 1.0,
    };
    memory * link
}

/// What is known of one device before the job starts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Prior {
    /// Relative weight (see [`static_weight`]); only ratios matter.
    pub weight: f64,
    /// A rate measured earlier in this process, in the role's units.
    pub measured: Option<f64>,
}

/// The priors for `keys` in `role`: the process's measurements where it has
/// them, and every device's static weight, normalised to the best and
/// floored at [`MIN_PRIOR_RATIO`].
pub(crate) fn priors_for(role: &str, keys: &[DeviceKey]) -> Vec<Prior> {
    let weights: Vec<f64> = keys.iter().map(|&k| static_weight(k)).collect();
    normalise_priors(weights, keys.iter().map(|&k| cached_rate(role, k)).collect())
}

pub(crate) fn normalise_priors(weights: Vec<f64>, measured: Vec<Option<f64>>) -> Vec<Prior> {
    let best = weights.iter().copied().fold(0.0f64, f64::max);
    weights
        .into_iter()
        .zip(measured)
        .map(|(w, measured)| Prior {
            weight: if best > 0.0 { (w / best).clamp(MIN_PRIOR_RATIO, 1.0) } else { 1.0 },
            measured,
        })
        .collect()
}

/// The index into `keys` of the device expected to be fastest in `role` —
/// for the choices where only one device is used (a whole-source decode, a
/// serial encode). Ties keep the earlier device.
pub(crate) fn fastest_of(role: &str, keys: &[DeviceKey]) -> Option<usize> {
    if keys.is_empty() {
        return None;
    }
    let board = SpeedBoard::new(priors_for(role, keys));
    Some(board.fastest((0..keys.len()).collect::<Vec<_>>().as_slice()))
}

#[derive(Debug, Clone)]
struct Device {
    prior: Prior,
    /// Measured this job.
    rate: Option<f64>,
    /// The unit in hand: (started at, units).
    busy: Option<(f64, f64)>,
    /// Still asking for work. A device that has stopped is no alternative.
    alive: bool,
}

/// Per-job speeds of a set of devices and the rule for handing out work.
#[derive(Debug, Clone)]
pub(crate) struct SpeedBoard {
    devices: Vec<Device>,
}

impl SpeedBoard {
    pub(crate) fn new(priors: Vec<Prior>) -> Self {
        Self {
            devices: priors.into_iter().map(|prior| Device { prior, rate: None, busy: None, alive: true }).collect(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.devices.len()
    }

    /// Device `d`'s rate in the role's units per second: measured this job,
    /// else measured earlier in the process, else its weight scaled by how
    /// the measured devices compare with their own weights — so a weight is
    /// only ever compared with something in the same units.
    pub(crate) fn rate(&self, d: usize) -> f64 {
        let dev = &self.devices[d];
        if let Some(r) = dev.rate.or(dev.prior.measured) {
            return r;
        }
        let known: Vec<f64> = self
            .devices
            .iter()
            .filter_map(|o| o.rate.or(o.prior.measured).map(|r| r / o.prior.weight.max(1e-9)))
            .collect();
        let scale = if known.is_empty() { 1.0 } else { known.iter().sum::<f64>() / known.len() as f64 };
        dev.prior.weight * scale
    }

    /// The fastest of `among`, by [`Self::rate`]; ties keep the earlier.
    pub(crate) fn fastest(&self, among: &[usize]) -> usize {
        let mut best = among[0];
        for &d in &among[1..] {
            if self.rate(d) > self.rate(best) {
                best = d;
            }
        }
        best
    }

    /// Should device `d`, idle now, take a unit of `units` work, with
    /// `remaining` units of work not yet handed out (this one included)?
    /// `eligible(o)` says whether device `o` could take this unit at all
    /// (it serves the rung, has not refused it).
    ///
    /// Yes, unless the other eligible devices — each finishing what it has in
    /// hand, then taking units of this size in earliest-finish order — would
    /// be done with *all* of `remaining` before `d` could finish this one.
    /// The device whose finish for this unit is earliest always gets a yes:
    /// the others' schedule for the remaining work ends no earlier than the
    /// best of them finishing one unit, which is no earlier than its own. So
    /// some device always takes the unit, and a "no" is only ever a wait.
    pub(crate) fn should_take(
        &self,
        d: usize,
        units: f64,
        remaining: f64,
        now: f64,
        eligible: impl Fn(usize) -> bool,
    ) -> bool {
        let units = units.max(1e-9);
        let my_finish = now + units / self.rate(d);
        let mut free_at: Vec<(f64, f64)> = Vec::new();
        for (o, dev) in self.devices.iter().enumerate() {
            if o == d || !dev.alive || !eligible(o) {
                continue;
            }
            let rate = self.rate(o);
            let in_hand = dev.busy.map_or(0.0, |(start, u)| (start + u / rate - now).max(0.0));
            free_at.push((now + in_hand, rate));
        }
        if free_at.is_empty() {
            return true;
        }
        let pieces = (remaining.max(units) / units).ceil() as usize;
        if pieces > MAX_SIMULATED_UNITS {
            return true;
        }
        // Take when this device's finish is within the slack of the others'.
        let limit = now + (my_finish - now) / (1.0 + TAKE_SLACK);
        let mut others_done = now;
        for _ in 0..pieces {
            let (best, finish) = free_at
                .iter()
                .enumerate()
                .map(|(i, &(at, rate))| (i, at + units / rate))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("free_at is not empty");
            free_at[best].0 = finish;
            others_done = others_done.max(finish);
            if others_done >= limit {
                // The others are still busy when this device would be done.
                return true;
            }
        }
        false
    }

    /// Device `d` has taken `units` of work at `now`.
    pub(crate) fn start(&mut self, d: usize, units: f64, now: f64) {
        self.devices[d].busy = Some((now, units));
    }

    /// Device `d` finished its unit of `units` in `elapsed` seconds. Returns
    /// the measured rate, for the process record.
    pub(crate) fn finish(&mut self, d: usize, units: f64, elapsed: f64) -> Option<f64> {
        let dev = &mut self.devices[d];
        dev.busy = None;
        if !(elapsed > 0.0 && units > 0.0) {
            return None;
        }
        let observed = units / elapsed.max(1e-4);
        dev.rate = Some(match dev.rate {
            None => observed,
            Some(r) => r * (1.0 - EWMA_ALPHA) + observed * EWMA_ALPHA,
        });
        Some(observed)
    }

    /// Device `d` has put its unit back without doing it.
    pub(crate) fn abandon(&mut self, d: usize) {
        self.devices[d].busy = None;
    }

    /// Device `d` has stopped taking work.
    pub(crate) fn retire(&mut self, d: usize) {
        self.devices[d].busy = None;
        self.devices[d].alive = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn priors(weights: &[f64]) -> Vec<Prior> {
        normalise_priors(weights.to_vec(), vec![None; weights.len()])
    }

    /// One simulated device: how fast it really is, whatever the board
    /// believes.
    #[derive(Clone, Copy)]
    struct Sim {
        speed: f64,
    }

    /// What a simulated run did.
    #[derive(Debug)]
    struct Outcome {
        makespan: f64,
        /// `taken[i]` = (unit index, device), in the order units were handed
        /// out.
        taken: Vec<(usize, usize)>,
        /// Units each device did.
        per_device: Vec<usize>,
    }

    /// Discrete-event run of `units` (sizes, handed out strictly in order —
    /// the decode ranges and the queue heads are) over `devices`, pulling: a
    /// free device asks for the next unit, and under `gated` the board may
    /// tell it to wait for the next completion. Fails on a stall — every
    /// device free, work left, nobody taking it — which the board must never
    /// produce.
    #[allow(clippy::needless_range_loop)]
    fn simulate(devices: &[Sim], board_priors: Vec<Prior>, units: &[f64], gated: bool) -> Outcome {
        let mut board = SpeedBoard::new(board_priors);
        let n = devices.len();
        let mut busy_until: Vec<Option<(f64, f64, f64)>> = vec![None; n]; // (end, units, start)
        let mut now = 0.0f64;
        let mut next = 0usize;
        let mut taken = Vec::new();
        let mut per_device = vec![0usize; n];
        let mut makespan = 0.0f64;
        loop {
            // Completions at `now`.
            for d in 0..n {
                if let Some((end, u, start)) = busy_until[d]
                    && end <= now + 1e-12
                {
                    board.finish(d, u, end - start);
                    busy_until[d] = None;
                    makespan = makespan.max(end);
                }
            }
            if next == units.len() && busy_until.iter().all(Option::is_none) {
                break;
            }
            // Free devices ask, lowest index first, until none takes anything.
            loop {
                let mut handed = false;
                for d in 0..n {
                    if busy_until[d].is_some() || next == units.len() {
                        continue;
                    }
                    let remaining: f64 = units[next..].iter().sum();
                    if gated && !board.should_take(d, units[next], remaining, now, |_| true) {
                        continue;
                    }
                    let u = units[next];
                    board.start(d, u, now);
                    busy_until[d] = Some((now + u / devices[d].speed, u, now));
                    taken.push((next, d));
                    per_device[d] += 1;
                    next += 1;
                    handed = true;
                }
                if !handed {
                    break;
                }
            }
            let next_event = busy_until.iter().flatten().map(|&(end, _, _)| end).fold(f64::INFINITY, f64::min);
            assert!(
                next_event.is_finite(),
                "stall at t={now}: {} units left, every device free and none would take the next",
                units.len() - next
            );
            now = next_event;
        }
        Outcome { makespan, taken, per_device }
    }

    /// The best a schedule of identical units can do on devices of these
    /// speeds: each unit, in turn, to the device that would finish it
    /// earliest (optimal for identical jobs on uniform machines).
    fn optimal_identical(speeds: &[f64], count: usize, unit: f64) -> f64 {
        let mut free = vec![0.0f64; speeds.len()];
        let mut makespan = 0.0f64;
        for _ in 0..count {
            let (d, finish) = free
                .iter()
                .zip(speeds)
                .enumerate()
                .map(|(d, (&f, &s))| (d, f + unit / s))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            free[d] = finish;
            makespan = makespan.max(finish);
        }
        makespan
    }

    /// devbox, roughly: the A750 three times the A380. 24 chunks. Pulling
    /// without the gate leaves the slow card a chunk at the end; the gate
    /// gets within a whisker of the optimum, and both cards beat the fast
    /// one alone.
    #[test]
    fn a_slow_card_does_not_form_the_tail() {
        let devices = [Sim { speed: 1.0 }, Sim { speed: 3.0 }]; // index 0 is the slow one, as on devbox
        let units = vec![1.0; 24];
        let fast_alone = 24.0 / 3.0;
        let best = optimal_identical(&[1.0, 3.0], 24, 1.0);
        // Priors that know nothing: both cards alike until measured.
        let gated = simulate(&devices, priors(&[1.0, 1.0]), &units, true);
        let ungated = simulate(&devices, priors(&[1.0, 1.0]), &units, false);
        assert!(gated.makespan <= best + 1e-9 + 1.0 / 3.0, "gated {} vs optimum {best}", gated.makespan);
        assert!(gated.makespan < fast_alone, "two cards {} must beat the fast one alone {fast_alone}", gated.makespan);
        assert!(ungated.makespan >= gated.makespan, "ungated {} gated {}", ungated.makespan, gated.makespan);
        assert!(gated.per_device[1] > gated.per_device[0] * 2, "{:?}", gated.per_device);
    }

    /// One unit, one fast and one slow card, the slow card asking first: with
    /// a prior that says which is which, the fast card gets it.
    #[test]
    fn the_only_unit_goes_to_the_card_expected_to_be_fastest() {
        let devices = [Sim { speed: 1.0 }, Sim { speed: 3.0 }];
        let link_x2 = Some(codec::gpu::PcieLink { gts: 8.0, width: 2 });
        let link_x16 = Some(codec::gpu::PcieLink { gts: 16.0, width: 16 });
        let weights = [weight_from_properties(6144, link_x2), weight_from_properties(8192, link_x16)];
        let out = simulate(&devices, priors(&weights), &[1.0], true);
        assert_eq!(out.taken, vec![(0, 1)]);
        assert!((out.makespan - 1.0 / 3.0).abs() < 1e-9);
        // Measured earlier in the process, the same without any properties.
        let measured = normalise_priors(vec![1.0, 1.0], vec![Some(10.0), Some(30.0)]);
        assert_eq!(simulate(&devices, measured, &[1.0], true).taken, vec![(0, 1)]);
    }

    /// A prior that is wrong — it rates the slow card as the fast one — is
    /// corrected by the first measurements, and the tail still lands on the
    /// really fast card.
    #[test]
    fn a_wrong_prior_is_corrected_by_measurement() {
        let devices = [Sim { speed: 1.0 }, Sim { speed: 3.0 }];
        let units = vec![1.0; 24];
        let best = optimal_identical(&[1.0, 3.0], 24, 1.0);
        let out = simulate(&devices, priors(&[1.0, 0.5]), &units, true);
        assert!(out.makespan <= best + 1.0, "makespan {} vs optimum {best}", out.makespan);
        assert!(out.makespan < 8.0);
    }

    /// Mixed vendors and a software slot: four devices of speeds 2, 1, 0.5
    /// and 0.2 (an NVIDIA card, an Arc, an iGPU, a software encoder), units
    /// of two sizes as rungs of a ladder are. Every unit is done exactly
    /// once, and the makespan is within one fast-device unit of the lower
    /// bound of perfectly divisible work.
    #[test]
    fn mixed_devices_finish_near_the_lower_bound() {
        let speeds = [2.0, 1.0, 0.5, 0.2];
        let devices: Vec<Sim> = speeds.iter().map(|&speed| Sim { speed }).collect();
        let units: Vec<f64> = (0..60).map(|i| if i % 3 == 0 { 2.25 } else { 1.0 }).collect();
        let total: f64 = units.iter().sum();
        let lower_bound = total / speeds.iter().sum::<f64>();
        let out = simulate(&devices, priors(&[1.0; 4]), &units, true);
        assert_eq!(out.taken.len(), units.len());
        assert!(
            out.makespan <= lower_bound + 2.25 / 2.0 + 1e-9,
            "makespan {} vs lower bound {lower_bound} ({:?})",
            out.makespan,
            out.per_device
        );
        let ungated = simulate(&devices, priors(&[1.0; 4]), &units, false);
        assert!(out.makespan <= ungated.makespan + 1e-9, "gated {} ungated {}", out.makespan, ungated.makespan);
    }

    /// Whatever the speeds and priors, nothing stalls, every unit is handed
    /// out exactly once and in order — so a run reassembled by unit index is
    /// the input, bit for bit — and the makespan never exceeds the fastest
    /// device's time alone by more than the largest unit on it.
    #[test]
    fn random_fleets_never_stall_and_keep_the_order() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rand = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..300 {
            let n = 1 + (rand() * 5.0) as usize;
            let speeds: Vec<f64> = (0..n).map(|_| 0.1 + rand() * 4.0).collect();
            let weights: Vec<f64> = (0..n).map(|_| 0.1 + rand()).collect();
            let count = 1 + (rand() * 40.0) as usize;
            let units: Vec<f64> = (0..count).map(|_| 0.5 + rand() * 2.0).collect();
            let devices: Vec<Sim> = speeds.iter().map(|&speed| Sim { speed }).collect();
            let out = simulate(&devices, priors(&weights), &units, true);
            let order: Vec<usize> = out.taken.iter().map(|&(u, _)| u).collect();
            assert_eq!(order, (0..count).collect::<Vec<_>>(), "units must be handed out once each, in order");
            let fastest = speeds.iter().copied().fold(0.0, f64::max);
            let alone: f64 = units.iter().sum::<f64>() / fastest;
            let biggest = units.iter().copied().fold(0.0, f64::max);
            let slowest = speeds.iter().copied().fold(f64::INFINITY, f64::min);
            // A prior can misplace the first unit of each device before it is
            // measured: bounded by one unit on the slowest device.
            assert!(
                out.makespan <= alone + biggest / slowest + 1e-9,
                "makespan {} vs fastest alone {alone} (speeds {speeds:?})",
                out.makespan
            );
        }
    }

    /// A device that is not eligible (it refused the rung) or has stopped is
    /// no alternative: a device alone takes everything.
    #[test]
    fn only_eligible_live_devices_count_as_alternatives() {
        let mut board = SpeedBoard::new(normalise_priors(vec![1.0, 1.0], vec![Some(1.0), Some(10.0)]));
        // The fast one would do the last unit sooner: the slow one waits.
        assert!(!board.should_take(0, 1.0, 1.0, 0.0, |_| true));
        // Not if the fast one cannot take this unit,
        assert!(board.should_take(0, 1.0, 1.0, 0.0, |o| o != 1));
        // or has stopped asking.
        board.retire(1);
        assert!(board.should_take(0, 1.0, 1.0, 0.0, |_| true));
    }

    /// The fast device busy for a long while yet makes the slow one worth
    /// using even for the last unit.
    #[test]
    fn a_busy_fast_device_is_waited_for_only_when_that_is_quicker() {
        let mut board = SpeedBoard::new(normalise_priors(vec![1.0, 1.0], vec![Some(1.0), Some(3.0)]));
        board.start(1, 30.0, 0.0); // ten seconds of work in hand
        assert!(board.should_take(0, 1.0, 1.0, 0.0, |_| true), "the fast card is busy for 10 s; 1 s here is better");
        // Nearly done: waiting for it is quicker.
        assert!(!board.should_take(0, 1.0, 1.0, 9.9, |_| true));
    }

    #[test]
    fn weights_come_from_memory_and_link() {
        let x16 = Some(codec::gpu::PcieLink { gts: 16.0, width: 16 });
        let x2 = Some(codec::gpu::PcieLink { gts: 8.0, width: 2 });
        let x4 = Some(codec::gpu::PcieLink { gts: 8.0, width: 4 });
        let a750 = weight_from_properties(8192, x16);
        let a380 = weight_from_properties(6144, x2);
        assert!((a750 - 1.0).abs() < 1e-9);
        assert!(a380 < 0.5 && a380 > 0.4, "{a380}");
        assert!(weight_from_properties(8192, x4) < a750);
        assert_eq!(weight_from_properties(8192, None), a750, "an unknown link is no penalty");
        // Floored relative to the best.
        let p = normalise_priors(vec![1.0, 0.01], vec![None, None]);
        assert_eq!(p[1].weight, MIN_PRIOR_RATIO);
    }

    #[test]
    fn the_process_record_averages_and_ignores_nonsense() {
        let role = "test:the_process_record";
        record_rate(role, DeviceKey::Gpu(7), 100.0);
        assert_eq!(cached_rate(role, DeviceKey::Gpu(7)), Some(100.0));
        record_rate(role, DeviceKey::Gpu(7), 200.0);
        assert!((cached_rate(role, DeviceKey::Gpu(7)).unwrap() - 140.0).abs() < 1e-9);
        record_rate(role, DeviceKey::Gpu(7), f64::NAN);
        record_rate(role, DeviceKey::Gpu(7), 0.0);
        assert!((cached_rate(role, DeviceKey::Gpu(7)).unwrap() - 140.0).abs() < 1e-9);
        assert_eq!(cached_rate(role, DeviceKey::Software), None);
    }
}
