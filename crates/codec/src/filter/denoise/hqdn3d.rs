//! `hqdn3d` — a high-quality spatio-temporal ("3D") denoiser. Clean-room
//! design (see `docs/filters/hqdn3d.md` for provenance); the option syntax is
//! the familiar `luma_spatial:chroma_spatial:luma_tmp:chroma_tmp`.
//!
//! Every stage is the same edge-preserving first-order recursive low-pass:
//!
//! ```text
//! y = x + k(|y_prev − x|) · (y_prev − x)
//! k(d) = k_max · exp(−(d / τ)²),   k_max = S / (S + KNEE),   τ = EDGE · S
//! ```
//!
//! `x` is the incoming sample, `y_prev` the filter's running state and `S` the
//! stage's strength. A small difference is mostly noise, so the state is kept
//! (up to `k_max`); a difference well beyond `τ` is an edge or motion, so `k`
//! falls to ~0 and the sample passes through. A stronger `S` both keeps more
//! (`k_max` → 1: a longer average) and tolerates larger differences (`τ`).
//!
//! - **Spatial**: the recursion runs left→right then right→left along every
//!   row, then top→bottom then bottom→top down every column. The forward and
//!   backward sweeps cancel each other's lag, so the result is centred (no
//!   smear in one direction).
//! - **Temporal**: the spatially filtered frame is blended, sample by sample,
//!   into the previous output frame with the same recursion — a static area
//!   converges to its long-run average, a moving one passes through.
//!
//! The history is kept in `f32`, so slow convergence is not lost to rounding;
//! the output is rounded once. A frame whose samples are all equal comes out
//! unchanged (every difference is 0).

use anyhow::Result;

use super::super::{assemble, planes_8bit};
use super::for_row_bands;
use super::simd::{Simd, Tier, round_clamp_u8, tiered};
use crate::frame::VideoFrame;

/// The documented defaults: `luma_spatial = 4`; the rest derive from it.
const LUMA_SPATIAL_DEFAULT: f32 = 4.0;

/// `k_max = S / (S + KNEE)`: the strength at which a flat area keeps half of
/// its running state per step.
const KNEE: f32 = 4.0;
/// `τ = EDGE · S`: the difference (in 8-bit code values) at which the
/// retention has fallen to `k_max / e`.
const EDGE: f32 = 1.5;

/// Retention table resolution: entries per code value; differences span
/// `0..=255`.
const STEPS: f32 = 8.0;
const CURVE_LEN: usize = 256 * STEPS as usize;

/// Below this many rows per band a thread does not pay for itself.
const MIN_BAND_ROWS: usize = 64;

/// The four strengths. An omitted value (given as `0`, or negative) is
/// derived from the others, as the filter's user documentation specifies:
/// `luma_spatial` defaults to 4, `chroma_spatial` to `3·ls/4`, `luma_tmp` to
/// `6·ls/4` and `chroma_tmp` to `lt·cs/ls`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Strengths {
    pub luma_spatial: f32,
    pub chroma_spatial: f32,
    pub luma_tmp: f32,
    pub chroma_tmp: f32,
}

impl Strengths {
    /// Resolve `ls:cs:lt:ct`, deriving any value that is not positive.
    pub fn resolve(ls: f32, cs: f32, lt: f32, ct: f32) -> Strengths {
        let given = |v: f32| v > 0.0 && v.is_finite();
        let luma_spatial = if given(ls) { ls } else { LUMA_SPATIAL_DEFAULT };
        let chroma_spatial = if given(cs) { cs } else { 3.0 * luma_spatial / 4.0 };
        let luma_tmp = if given(lt) { lt } else { 6.0 * luma_spatial / 4.0 };
        let chroma_tmp = if given(ct) { ct } else { luma_tmp * chroma_spatial / luma_spatial };
        Strengths { luma_spatial, chroma_spatial, luma_tmp, chroma_tmp }
    }
}

/// The retention `k(d)` of one stage, tabulated over `d ∈ [0, 256)`.
struct Curve {
    k: Vec<f32>,
}

impl Curve {
    fn new(strength: f32) -> Self {
        if strength <= 0.0 || !strength.is_finite() {
            return Curve { k: vec![0.0; CURVE_LEN + 1] };
        }
        let k_max = strength / (strength + KNEE);
        let tau = EDGE * strength;
        // One entry past the table, 0: what an index off its end reads as
        // (`step`'s `unwrap_or`), so the vector kernels can clamp the index
        // to it and gather in bounds.
        let k = (0..CURVE_LEN)
            .map(|i| {
                let d = (i as f32 + 0.5) / STEPS;
                k_max * (-(d / tau) * (d / tau)).exp()
            })
            .chain([0.0])
            .collect();
        Curve { k }
    }

    /// One recursion step: the new state from the running `state` and the
    /// incoming sample `x`.
    #[cfg(test)]
    fn step(&self, state: f32, x: f32) -> f32 {
        step(&self.k, state, x)
    }
}

/// One recursion step with the retention table `k`.
#[inline(always)]
fn step(k: &[f32], state: f32, x: f32) -> f32 {
    let diff = state - x;
    // `as usize` saturates; |diff| ≤ 255 keeps it in the table anyway.
    let k = k.get((diff.abs() * STEPS) as usize).copied().unwrap_or(0.0);
    x + k * diff
}

/// [`step`] lane-wise, bit-exact with it: `max(d, 0 - d)` is `|d|`, an index
/// at or past the table's end is clamped to its trailing 0 (which `step`
/// reads as 0 too), and `k * d` is added to `x` after rounding, as `step`
/// does it — no fused multiply-add.
#[inline(always)]
unsafe fn step_v<S: Simd>(k: &[f32], state: S::F, x: S::F) -> S::F {
    unsafe {
        let d = S::sub_f32(state, x);
        let a = S::max_f32(d, S::sub_f32(S::set1_f32(0.0), d));
        let p = S::min_f32(S::mul_f32(a, S::set1_f32(STEPS)), S::set1_f32(CURVE_LEN as f32));
        let kv = S::lookup_f32(k, S::trunc_f32_i32(p));
        S::add_f32(x, S::mul_f32(kv, d))
    }
}

/// The per-plane curves, built once per chain ([`super::super::FilterChain`])
/// and shared by every stream's instance.
pub(crate) struct Prepared {
    spatial: [Curve; 2],
    temporal: [Curve; 2],
}

/// One stream's history: the previous output frame, unrounded.
pub(crate) struct State {
    w: usize,
    h: usize,
    planes: [Vec<f32>; 3],
}

impl Prepared {
    pub(crate) fn new(strengths: Strengths) -> Self {
        Prepared {
            spatial: [Curve::new(strengths.luma_spatial), Curve::new(strengths.chroma_spatial)],
            temporal: [Curve::new(strengths.luma_tmp), Curve::new(strengths.chroma_tmp)],
        }
    }

    /// Filter the stream's next frame against `state` (its history), updating
    /// it. No history, or history of another frame size, starts afresh.
    pub(crate) fn apply(&self, state: &mut Option<State>, frame: &VideoFrame) -> Result<VideoFrame> {
        let (yp, up, vp) = planes_8bit(frame, "hqdn3d")?;
        let (w, h) = (frame.width as usize, frame.height as usize);
        let dims = [(w, h), (w / 2, h / 2), (w / 2, h / 2)];
        if state.as_ref().is_some_and(|s| s.w != w || s.h != h) {
            *state = None;
        }
        let mut next: [Vec<f32>; 3] = Default::default();
        let mut out: [Vec<u8>; 3] = Default::default();
        for (i, src) in [yp, up, vp].iter().enumerate() {
            let (pw, ph) = dims[i];
            let c = usize::from(i > 0);
            let mut cur = spatial(src, pw, ph, &self.spatial[c]);
            if let Some(prev) = state.as_ref().map(|s| &s.planes[i]) {
                temporal(&mut cur, prev, pw, &self.temporal[c]);
            }
            let mut o = vec![0u8; cur.len()];
            to_u8(Tier::detect(), &cur, &mut o);
            out[i] = o;
            next[i] = cur;
        }
        *state = Some(State { w, h, planes: next });
        let [y, u, v] = out;
        Ok(assemble(frame, frame.width, frame.height, y, u, v))
    }
}

/// The spatial stage: both directions along rows, then both down columns.
fn spatial(src: &[u8], w: usize, h: usize, curve: &Curve) -> Vec<f32> {
    let mut buf: Vec<f32> = src[..w * h].iter().map(|&v| v as f32).collect();
    if w == 0 || h == 0 {
        return buf;
    }
    let tier = Tier::detect();
    for_row_bands(&mut buf, w, MIN_BAND_ROWS, |_, rows| sweep_rows(tier, rows, w, &curve.k));
    columns(tier, &mut buf, w, h, &curve.k);
    buf
}

/// Forward then backward recursion along one row, in place.
fn sweep_row(row: &mut [f32], k: &[f32]) {
    let mut s = row[0];
    for v in row.iter_mut() {
        s = step(k, s, *v);
        *v = s;
    }
    let mut s = *row.last().unwrap();
    for v in row.iter_mut().rev() {
        s = step(k, s, *v);
        *v = s;
    }
}

/// [`sweep_row`] over every row of `rows` (whole rows of `w`).
fn sweep_rows_scalar(rows: &mut [f32], w: usize, k: &[f32]) {
    for row in rows.chunks_exact_mut(w) {
        sweep_row(row, k);
    }
}

tiered!(fn sweep_rows(rows: &mut [f32], w: usize, k: &[f32]) => sweep_rows_body, scalar sweep_rows_scalar);

/// The row recursion is serial along a row, so the vectors run `LANES` rows
/// side by side instead: a group of rows is transposed into a scratch block
/// (sample `x` of row `r` at `x * LANES + r`, gathered), each row's
/// recursion runs as one lane of [`step_v`], and the block is transposed
/// back. Rows left over after the last full group take [`sweep_row`].
#[inline(always)]
unsafe fn sweep_rows_body<S: Simd>(rows: &mut [f32], w: usize, k: &[f32]) {
    unsafe {
        let l = S::LANES;
        let n = rows.len() / w;
        let full = n - n % l;
        if full > 0 && w > 0 {
            let mut t = vec![0f32; w * l];
            let down: Vec<i32> = (0..l).map(|r| (r * w) as i32).collect();
            let across: Vec<i32> = (0..l).map(|j| (j * l) as i32).collect();
            let (down, across) = (S::load_i32(down.as_ptr()), S::load_i32(across.as_ptr()));
            let wide = w - w % l;
            for g in (0..full).step_by(l) {
                let block = &mut rows[g * w..(g + l) * w];
                for x in 0..w {
                    S::store_f32(t.as_mut_ptr().add(x * l), S::lookup_f32(block, S::add_i32(down, S::set1_i32(x as i32))));
                }
                let tp = t.as_mut_ptr();
                let mut s = S::load_f32(tp);
                for x in 0..w {
                    s = step_v::<S>(k, s, S::load_f32(tp.add(x * l)));
                    S::store_f32(tp.add(x * l), s);
                }
                let mut s = S::load_f32(tp.add((w - 1) * l));
                for x in (0..w).rev() {
                    s = step_v::<S>(k, s, S::load_f32(tp.add(x * l)));
                    S::store_f32(tp.add(x * l), s);
                }
                for r in 0..l {
                    let row = &mut block[r * w..(r + 1) * w];
                    let mut x = 0;
                    while x < wide {
                        S::store_f32(row.as_mut_ptr().add(x), S::lookup_f32(&t, S::add_i32(across, S::set1_i32((x * l + r) as i32))));
                        x += l;
                    }
                    for x in wide..w {
                        row[x] = t[x * l + r];
                    }
                }
            }
        }
        sweep_rows_scalar(&mut rows[full * w..], w, k);
    }
}

/// Downward then upward recursion along every column, in place. The state
/// is one row wide, so the sweep walks memory row by row.
fn columns_scalar(buf: &mut [f32], w: usize, h: usize, k: &[f32]) {
    let mut s = buf[..w].to_vec();
    for y in 0..h {
        let row = &mut buf[y * w..][..w];
        for (st, v) in s.iter_mut().zip(row.iter_mut()) {
            *st = step(k, *st, *v);
            *v = *st;
        }
    }
    s.copy_from_slice(&buf[(h - 1) * w..][..w]);
    for y in (0..h).rev() {
        let row = &mut buf[y * w..][..w];
        for (st, v) in s.iter_mut().zip(row.iter_mut()) {
            *st = step(k, *st, *v);
            *v = *st;
        }
    }
}

tiered!(fn columns(buf: &mut [f32], w: usize, h: usize, k: &[f32]) => columns_body, scalar columns_scalar);

/// [`columns_scalar`] with the columns side by side in the lanes (they are
/// independent), the last `w % LANES` scalar.
#[inline(always)]
unsafe fn columns_body<S: Simd>(buf: &mut [f32], w: usize, h: usize, k: &[f32]) {
    unsafe {
        let wide = w - w % S::LANES;
        let mut s = buf[..w].to_vec();
        for y in 0..h {
            column_step::<S>(&mut s, &mut buf[y * w..][..w], wide, k);
        }
        s.copy_from_slice(&buf[(h - 1) * w..][..w]);
        for y in (0..h).rev() {
            column_step::<S>(&mut s, &mut buf[y * w..][..w], wide, k);
        }
    }
}

/// One row of [`columns_body`]: the state `s` stepped by `row`, both
/// updated. A function rather than a closure so that it inlines into the
/// `#[target_feature]` caller (a closure is compiled without the feature,
/// and its intrinsics became calls: 25 ms a plane against 1.4).
#[inline(always)]
unsafe fn column_step<S: Simd>(s: &mut [f32], row: &mut [f32], wide: usize, k: &[f32]) {
    unsafe {
        let mut x = 0;
        while x < wide {
            let v = step_v::<S>(k, S::load_f32(s.as_ptr().add(x)), S::load_f32(row.as_ptr().add(x)));
            S::store_f32(s.as_mut_ptr().add(x), v);
            S::store_f32(row.as_mut_ptr().add(x), v);
            x += S::LANES;
        }
        for x in wide..row.len() {
            s[x] = step(k, s[x], row[x]);
            row[x] = s[x];
        }
    }
}

/// The temporal stage: blend `cur` into the previous output `prev`.
fn temporal(cur: &mut [f32], prev: &[f32], w: usize, curve: &Curve) {
    if w == 0 {
        return;
    }
    let tier = Tier::detect();
    for_row_bands(cur, w, MIN_BAND_ROWS, |y0, rows| {
        temporal_rows(tier, rows, &prev[y0 * w..y0 * w + rows.len()], &curve.k);
    });
}

fn temporal_rows_scalar(cur: &mut [f32], prev: &[f32], k: &[f32]) {
    for (c, &p) in cur.iter_mut().zip(prev) {
        *c = step(k, p, *c);
    }
}

tiered!(fn temporal_rows(cur: &mut [f32], prev: &[f32], k: &[f32]) => temporal_rows_body, scalar temporal_rows_scalar);

#[inline(always)]
unsafe fn temporal_rows_body<S: Simd>(cur: &mut [f32], prev: &[f32], k: &[f32]) {
    unsafe {
        let l = S::LANES;
        let wide = cur.len() - cur.len() % l;
        let mut i = 0;
        while i < wide {
            let v = step_v::<S>(k, S::load_f32(prev.as_ptr().add(i)), S::load_f32(cur.as_ptr().add(i)));
            S::store_f32(cur.as_mut_ptr().add(i), v);
            i += l;
        }
        temporal_rows_scalar(&mut cur[wide..], &prev[wide..], k);
    }
}

/// The output: each state rounded half away from zero and clamped to a
/// byte (the states are never negative: every step lands between its two
/// inputs).
fn to_u8_scalar(src: &[f32], out: &mut [u8]) {
    for (o, &v) in out.iter_mut().zip(src) {
        *o = v.round().clamp(0.0, 255.0) as u8;
    }
}

tiered!(fn to_u8(src: &[f32], out: &mut [u8]) => to_u8_body, scalar to_u8_scalar);

#[inline(always)]
unsafe fn to_u8_body<S: Simd>(src: &[f32], out: &mut [u8]) {
    unsafe {
        let l = S::LANES;
        let wide = src.len() - src.len() % l;
        let mut i = 0;
        while i < wide {
            S::store_f32_u8(out.as_mut_ptr().add(i), round_clamp_u8::<S>(S::load_f32(src.as_ptr().add(i))));
            i += l;
        }
        to_u8_scalar(&src[wide..], &mut out[wide..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_strengths_derive_from_the_given_ones() {
        let r = Strengths::resolve;
        let s = |ls, cs, lt, ct| Strengths { luma_spatial: ls, chroma_spatial: cs, luma_tmp: lt, chroma_tmp: ct };
        assert_eq!(r(0.0, 0.0, 0.0, 0.0), s(4.0, 3.0, 6.0, 4.5));
        assert_eq!(r(8.0, 0.0, 0.0, 0.0), s(8.0, 6.0, 12.0, 9.0));
        assert_eq!(r(2.0, 0.0, 10.0, 0.0), s(2.0, 1.5, 10.0, 7.5));
        assert_eq!(r(4.0, 1.0, 0.0, 0.0), s(4.0, 1.0, 6.0, 1.5));
        assert_eq!(r(1.0, 2.0, 3.0, 4.0), s(1.0, 2.0, 3.0, 4.0));
        assert_eq!(r(-1.0, f32::NAN, 0.0, 0.0), s(4.0, 3.0, 6.0, 4.5));
    }

    #[test]
    fn retention_falls_with_the_difference_and_rises_with_the_strength() {
        for st in [1.0f32, 4.0, 10.0] {
            let c = Curve::new(st);
            assert!(c.k.windows(2).all(|p| p[1] <= p[0]), "k must not rise with d");
            assert!(c.k[0] < 1.0, "k must stay below 1 so the state cannot freeze");
            // An edge of 10·S passes essentially untouched.
            assert!(c.step(0.0, 10.0 * st) > 10.0 * st - 1e-3);
        }
        let (weak, strong) = (Curve::new(2.0), Curve::new(8.0));
        for d in [0.5f32, 2.0, 5.0, 12.0] {
            assert!(
                (strong.step(0.0, d) - d).abs() > (weak.step(0.0, d) - d).abs(),
                "a stronger setting must smooth a difference of {d} more"
            );
        }
        // Strength 0 is a pass-through.
        assert_eq!(Curve::new(0.0).step(7.0, 3.0), 3.0);
    }

    /// Every SIMD tier the host has computes the scalar reference's states
    /// bit for bit — the row sweeps (whole groups of rows and a remainder,
    /// widths with a tail), the column sweeps, the temporal blend and the
    /// output rounding — at several strengths, on noise, rails and flats.
    #[test]
    fn every_tier_matches_the_scalar_stages() {
        let mut seed = 0x4d_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        for &(w, h) in &[(1usize, 1usize), (7, 3), (16, 8), (19, 13), (33, 9), (64, 17)] {
            for kind in 0..3 {
                let src: Vec<u8> = (0..w * h)
                    .map(|_| match kind {
                        0 => next() as u8,
                        1 => [0u8, 255][(next() & 1) as usize],
                        _ => 100 + (next() % 3) as u8,
                    })
                    .collect();
                let prev: Vec<f32> = (0..w * h).map(|_| (next() % 2560) as f32 / 10.0).collect();
                for strength in [0.0f32, 1.0, 4.0, 12.0] {
                    let c = Curve::new(strength);
                    let base: Vec<f32> = src.iter().map(|&v| v as f32).collect();
                    let mut want_rows = base.clone();
                    sweep_rows_scalar(&mut want_rows, w, &c.k);
                    let mut want_cols = want_rows.clone();
                    columns_scalar(&mut want_cols, w, h, &c.k);
                    let mut want_t = want_cols.clone();
                    temporal_rows_scalar(&mut want_t, &prev, &c.k);
                    let mut want_u8 = vec![0u8; w * h];
                    to_u8_scalar(&want_t, &mut want_u8);
                    for tier in Tier::available() {
                        let mut rows = base.clone();
                        sweep_rows(tier, &mut rows, w, &c.k);
                        assert!(rows == want_rows, "rows {tier:?} {w}x{h} kind {kind} S {strength}");
                        let mut cols = want_rows.clone();
                        columns(tier, &mut cols, w, h, &c.k);
                        assert!(cols == want_cols, "columns {tier:?} {w}x{h} kind {kind} S {strength}");
                        let mut t = want_cols.clone();
                        temporal_rows(tier, &mut t, &prev, &c.k);
                        assert!(t == want_t, "temporal {tier:?} {w}x{h} kind {kind} S {strength}");
                        let mut o = vec![0u8; w * h];
                        to_u8(tier, &want_t, &mut o);
                        assert_eq!(o, want_u8, "to_u8 {tier:?} {w}x{h}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_spatial_sweeps_are_symmetric() {
        // A centred impulse spreads the same amount left and right, up and
        // down: forward and backward sweeps cancel each other's lag.
        let (w, h) = (15, 15);
        let mut src = vec![100u8; w * h];
        src[7 * w + 7] = 104;
        let out = spatial(&src, w, h, &Curve::new(6.0));
        for d in 1..7 {
            let (l, r) = (out[7 * w + 7 - d], out[7 * w + 7 + d]);
            let (u, b) = (out[(7 - d) * w + 7], out[(7 + d) * w + 7]);
            assert!((l - r).abs() < 0.05, "row asymmetry at {d}: {l} vs {r}");
            assert!((u - b).abs() < 0.05, "column asymmetry at {d}: {u} vs {b}");
        }
    }
}

#[cfg(test)]
mod stage_bench {
    use super::*;

    /// ms per 1080p luma plane for each stage at each tier.
    /// `cargo test --release -p rivet-codec hqdn3d_stage_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "timing, not a check"]
    fn hqdn3d_stage_bench() {
        let (w, h) = (1920usize, 1080usize);
        let base: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 251) as f32).collect();
        let c = Curve::new(4.0);
        for tier in Tier::available() {
            let t = std::time::Instant::now();
            let mut b = base.clone();
            for _ in 0..5 {
                sweep_rows(tier, &mut b, w, &c.k);
            }
            let rows = t.elapsed().as_secs_f64() * 200.0;
            let t = std::time::Instant::now();
            for _ in 0..5 {
                columns(tier, &mut b, w, h, &c.k);
            }
            let cols = t.elapsed().as_secs_f64() * 200.0;
            let t = std::time::Instant::now();
            for _ in 0..5 {
                temporal_rows(tier, &mut b, &base, &c.k);
            }
            let temp = t.elapsed().as_secs_f64() * 200.0;
            eprintln!("{tier:?}: rows {rows:.2} ms, columns {cols:.2} ms, temporal {temp:.2} ms");
        }
    }
}
