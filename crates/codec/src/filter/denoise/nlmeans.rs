//! **Non-local means** — clean-room implementation from the published
//! literature (see `docs/filters/nlmeans.md` for provenance).
//!
//! Every output sample is a weighted average of the samples in a square
//! *research window* around it; a candidate's weight depends on how alike the
//! *patches* (small squares) centred on the two samples are, not on how close
//! the samples are. This is the estimator of Buades, Coll & Morel, "A non-local
//! algorithm for image denoising" (CVPR 2005), with the thresholded weight form
//! of their IPOL article "Non-Local Means Denoising" (2011):
//!
//! ```text
//! d²(p, q) = mean over the patch of (I(p+u) − I(q+u))²
//! w(p, q)  = exp(−max(d² − σ², 0) / σ²)
//! out(p)   = Σ_q w(p, q)·I(q) / Σ_q w(p, q)
//! ```
//!
//! `σ` is the user's strength `s`, read as the standard deviation (in 8-bit
//! code values) of the noise to remove. Patches closer than `σ²` per sample
//! count as "the same" (weight 1); the weight decays beyond that. The IPOL
//! article uses a free zone of `2σ²` and `h = 0.4σ`; we use a free zone of `σ²`
//! and `h = σ`, a gentler roll-off measured to peak at `s ≈ noise σ` across
//! noise levels and to degrade gradually either side of it (figures in
//! `docs/filters/nlmeans.md`).
//!
//! **Speed.** The direct form costs `patch² · window²` per sample. Following
//! the published offset-major approach (Wang et al. 2006, Darbon et al. 2008),
//! the loop runs over *offsets* `o` instead: for one `o` the squared-difference
//! image `(I(x) − I(x+o))²` is box-summed over the patch with running sums, so
//! every sample's patch distance for that `o` costs O(1), independent of the
//! patch size. The distance is symmetric — `d(p, p+o) = d(p+o, p)` — so one
//! weight map serves both `+o` and `−o`, halving the work.
//!
//! **Borders.** Patches read an edge-replicated copy of the plane; candidates
//! are restricted to samples inside the plane (the research window is clipped,
//! never padded), so no replicated sample is ever averaged in.
//!
//! **Determinism.** Each output sample accumulates the same terms in the same
//! order however the plane is split into row bands across threads, and all
//! patch distances are exact integers, so the output is bit-identical for any
//! thread count.

// The vector bodies are only reached through `tiered!`'s x86 arms.
#![cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    allow(dead_code)
)]

use super::for_row_bands;
use super::simd::{Simd, Tier, tiered};

/// Fixed setting for `denoise=nlmeans[:STRENGTH]`, where the method runs at one
/// internal setting and `STRENGTH` only blends: a 3×3 patch, a 9×9 research
/// window and σ = 10 — enough to flatten light sensor noise.
const FIXED_PATCH: u32 = 3;
const FIXED_RESEARCH: u32 = 9;
const FIXED_SIGMA: f32 = 10.0;

/// Patch distances (mean squared difference per sample) up to
/// `FREE_FACTOR · σ²` weigh 1.
const FREE_FACTOR: f32 = 1.0;
/// Beyond that the weight decays as `exp(−excess / h²)`, `h = H_FACTOR · σ`.
const H_FACTOR: f32 = 1.0;

/// Weight table resolution: entries per unit of the exponent, and how many
/// units it spans. Past the end (`exp(−16)` ≈ 1e-7) the weight is 0.
const LUT_STEPS: f32 = 256.0;
const LUT_UNITS: usize = 16;
const LUT_LEN: usize = LUT_UNITS * LUT_STEPS as usize;

/// Below this many rows per band a thread does not pay for itself.
const MIN_BAND_ROWS: usize = 16;

/// `denoise=nlmeans` — the fixed setting.
pub(super) fn plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    plane_params(src, w, h, FIXED_PATCH, FIXED_RESEARCH, FIXED_SIGMA)
}

/// `nlmeans=s=..:p=..:r=..` on one plane: `patch` and `research` are window
/// sizes in samples (an even size counts as the next odd one; 0 and 1 mean a
/// single sample), `sigma` the strength.
pub(super) fn plane_params(
    src: &[u8],
    w: usize,
    h: usize,
    patch: u32,
    research: u32,
    sigma: f32,
) -> Vec<u8> {
    let Some(k) = Kernel::new(src, w, h, patch, research, sigma) else {
        return src.to_vec();
    };
    let mut out = vec![0u8; w * h];
    for_row_bands(&mut out, w, MIN_BAND_ROWS, |y0, rows| k.band(y0, rows));
    out
}

/// Radius of an odd window of `size` samples (even sizes round up to odd).
fn radius(size: u32) -> usize {
    (size / 2) as usize
}

/// The weight as a function of a patch's summed squared difference.
struct Weights {
    /// `FREE_FACTOR · σ² · n`: summed distances up to this weigh 1.
    free: f32,
    /// Converts the excess over `free` to a table index: `LUT_STEPS / (n·h²)`.
    scale: f32,
    /// `exp(−(i + ½) / LUT_STEPS)`.
    lut: Vec<f32>,
}

impl Weights {
    fn new(sigma: f32, patch_samples: usize) -> Self {
        let n = patch_samples as f32;
        let sigma = sigma.max(f32::MIN_POSITIVE);
        let hh = (H_FACTOR * sigma) * (H_FACTOR * sigma);
        Weights {
            free: FREE_FACTOR * sigma * sigma * n,
            scale: LUT_STEPS / (n * hh),
            // One entry past the table, 0: what an index off its end reads
            // as (`of`'s `unwrap_or`), so the vector form can clamp onto it.
            lut: (0..LUT_LEN)
                .map(|i| (-(i as f32 + 0.5) / LUT_STEPS).exp())
                .chain([0.0])
                .collect(),
        }
    }

    #[cfg(test)]
    fn of(&self, ssd: u32) -> f32 {
        let excess = ssd as f32 - self.free;
        if excess <= 0.0 {
            return 1.0;
        }
        // `as usize` saturates, so a huge excess lands past the table.
        self.lut
            .get((excess * self.scale) as usize)
            .copied()
            .unwrap_or(0.0)
    }
}

/// Everything one call needs, shared read-only by the band workers.
struct Kernel<'a> {
    src: &'a [u8],
    w: usize,
    h: usize,
    /// Patch radius.
    pr: usize,
    /// Research radius, clipped to the plane: horizontal, vertical.
    srx: usize,
    sry: usize,
    /// The plane edge-replicated by `pr` on every side, `pw` wide.
    pad: Vec<u8>,
    pw: usize,
    weights: Weights,
}

impl<'a> Kernel<'a> {
    /// `None` when there is nothing to average (empty plane or a research
    /// window of one sample) — the caller copies the source.
    fn new(
        src: &'a [u8],
        w: usize,
        h: usize,
        patch: u32,
        research: u32,
        sigma: f32,
    ) -> Option<Self> {
        if w == 0 || h == 0 || src.len() < w * h {
            return None;
        }
        let sr = radius(research);
        let (srx, sry) = (sr.min(w - 1), sr.min(h - 1));
        if srx == 0 && sry == 0 {
            return None;
        }
        let pr = radius(patch);
        let pw = w + 2 * pr;
        let ph = h + 2 * pr;
        let mut pad = vec![0u8; pw * ph];
        for py in 0..ph {
            let sy = py.saturating_sub(pr).min(h - 1);
            let row = &src[sy * w..][..w];
            let dst = &mut pad[py * pw..][..pw];
            dst[..pr].fill(row[0]);
            dst[pr..pr + w].copy_from_slice(row);
            dst[pr + w..].fill(row[w - 1]);
        }
        let side = 2 * pr + 1;
        Some(Kernel {
            src,
            w,
            h,
            pr,
            srx,
            sry,
            pad,
            pw,
            weights: Weights::new(sigma, side * side),
        })
    }

    /// Denoise output rows `y0 .. y0 + rows.len() / w` into `rows`.
    ///
    /// The loops are element-wise (integer squared differences, `sum +=
    /// w * c`, `wsum += w`), so the compiler vectorises them; on a CPU with
    /// AVX2 (and the denoise tier allowing it) the same code is compiled a
    /// second time with that feature and runs eight lanes wide. Element-wise
    /// IEEE operations without contraction, so both write the same bytes.
    fn band(&self, y0: usize, rows: &mut [u8]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if Tier::detect() >= Tier::Avx2 {
            #[target_feature(enable = "avx2")]
            unsafe fn avx2(k: &Kernel<'_>, y0: usize, rows: &mut [u8]) {
                k.band_impl(y0, rows)
            }
            // SAFETY: the tier is Avx2 only on a CPU with AVX2.
            return unsafe { avx2(self, y0, rows) };
        }
        self.band_impl(y0, rows)
    }

    #[inline(always)]
    fn band_impl(&self, y0: usize, rows: &mut [u8]) {
        let (w, h, src) = (self.w, self.h, self.src);
        let y1 = y0 + rows.len() / w;
        // The centre sample is its own candidate, at distance 0 ⇒ weight 1.
        let mut sum: Vec<f32> = src[y0 * w..y1 * w].iter().map(|&v| v as f32).collect();
        let mut wsum = vec![1.0f32; sum.len()];
        // Weight maps for one offset, for rows y0 − dy .. y1.
        let mut wmap = vec![0f32; (y1 - y0 + self.sry) * w];
        let mut col = vec![0u32; self.pw];

        for dy in 0..=self.sry {
            for dx in -(self.srx as isize)..=self.srx as isize {
                // Half the offsets; `−o` rides on `+o`'s weights.
                if dy == 0 && dx <= 0 {
                    continue;
                }
                // Rows q whose pair (q, q + o) has both ends in the plane and
                // matters to this band: q ∈ [y0 − dy, y1) ∩ [0, h − dy).
                let qa = y0.saturating_sub(dy);
                let qb = y1.min(h - dy);
                if qa >= qb {
                    continue;
                }
                // Columns likewise: q.x and q.x + dx both inside.
                let xa = (-dx).max(0) as usize;
                let xb = (w as isize).min(w as isize - dx);
                if xb <= xa as isize {
                    continue;
                }
                let xb = xb as usize;
                self.weight_rows(dx, dy, qa, qb, xa, xb, &mut col, &mut wmap);

                for y in y0..y1 {
                    let acc = (y - y0) * w;
                    // Candidate p + o, weighted by the map at p.
                    if y + dy < h {
                        let wr = &wmap[(y - qa) * w + xa..(y - qa) * w + xb];
                        let c0 = ((y + dy) * w) as isize + xa as isize + dx;
                        let cand = &src[c0 as usize..][..xb - xa];
                        let (s, ws) = (&mut sum[acc + xa..acc + xb], &mut wsum[acc + xa..acc + xb]);
                        for (((s, ws), &wt), &c) in
                            s.iter_mut().zip(ws.iter_mut()).zip(wr).zip(cand)
                        {
                            *s += wt * c as f32;
                            *ws += wt;
                        }
                    }
                    // Candidate p − o = q, weighted by the map at q.
                    if y >= dy {
                        let qy = y - dy;
                        let wr = &wmap[(qy - qa) * w + xa..(qy - qa) * w + xb];
                        let cand = &src[qy * w + xa..qy * w + xb];
                        let x0 = (acc as isize + xa as isize + dx) as usize;
                        let (s, ws) = (&mut sum[x0..x0 + (xb - xa)], &mut wsum[x0..x0 + (xb - xa)]);
                        for (((s, ws), &wt), &c) in
                            s.iter_mut().zip(ws.iter_mut()).zip(wr).zip(cand)
                        {
                            *s += wt * c as f32;
                            *ws += wt;
                        }
                    }
                }
            }
        }
        for ((o, &s), &ws) in rows.iter_mut().zip(&sum).zip(&wsum) {
            *o = (s / ws).round().clamp(0.0, 255.0) as u8;
        }
    }

    /// Fill `wmap` rows `q − qa` (for q in `qa..qb`), columns `xa..xb`, with
    /// the weight between the patches at q and q + (dx, dy).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn weight_rows(
        &self,
        dx: isize,
        dy: usize,
        qa: usize,
        qb: usize,
        xa: usize,
        xb: usize,
        col: &mut [u32],
        wmap: &mut [f32],
    ) {
        let (pr, pw, w) = (self.pr, self.pw, self.w);
        let side = 2 * pr + 1;
        // Padded columns the patches span: xa .. xb + 2·pr.
        let (ca, cb) = (xa, xb + 2 * pr);
        let pad = &self.pad;
        // Squared differences of padded row `py`, added to (or taken from)
        // the column sums. Padded row py + dy and column cx + dx stay inside
        // the padded plane because q + o is inside the plane.
        //
        // Written as zipped slices (the `+ dx` folded into where `b`
        // starts) with the add and the subtract as separate loops, so the
        // compiler vectorises them: integer arithmetic, the same sums.
        let diff_row = |py: usize, col: &mut [u32], add: bool| {
            let a = &pad[py * pw + ca..py * pw + cb];
            let b0 = (py + dy) * pw;
            let b = &pad[(b0 as isize + ca as isize + dx) as usize..][..cb - ca];
            let col = &mut col[ca..cb];
            let d2 = |(&a, &b): (&u8, &u8)| {
                let d = a as i32 - b as i32;
                (d * d) as u32
            };
            if add {
                for (c, d) in col.iter_mut().zip(a.iter().zip(b).map(d2)) {
                    *c += d;
                }
            } else {
                for (c, d) in col.iter_mut().zip(a.iter().zip(b).map(d2)) {
                    *c -= d;
                }
            }
        };
        col[ca..cb].fill(0);
        // Patch rows of q = qa are padded rows qa .. qa + side.
        for py in qa..qa + side {
            diff_row(py, col, true);
        }
        let tier = Tier::detect();
        let mut win = vec![0u32; xb - xa];
        for q in qa..qb {
            let out = &mut wmap[(q - qa) * w..][..w];
            // The patch sums along the row (exact integers), then their
            // weights, which vectorise once the running sum is out of the way.
            let mut s: u32 = col[xa..xa + side].iter().sum();
            win[0] = s;
            for x in xa + 1..xb {
                s = s + col[x + side - 1] - col[x - 1];
                win[x - xa] = s;
            }
            let wt = &self.weights;
            weights_row(tier, &win, &mut out[xa..xb], wt.free, wt.scale, &wt.lut);
            if q + 1 < qb {
                diff_row(q, col, false);
                diff_row(q + side, col, true);
            }
        }
    }
}

/// `Weights::of` along a row of patch sums.
fn weights_row_scalar(win: &[u32], out: &mut [f32], free: f32, scale: f32, lut: &[f32]) {
    for (o, &s) in out.iter_mut().zip(win) {
        let excess = s as f32 - free;
        *o = if excess <= 0.0 {
            1.0
        } else {
            lut.get((excess * scale) as usize).copied().unwrap_or(0.0)
        };
    }
}

tiered!(fn weights_row(win: &[u32], out: &mut [f32], free: f32, scale: f32, lut: &[f32]) => weights_row_body, scalar weights_row_scalar);

/// [`weights_row_scalar`] lane-wise, bit-exact: the sum converted as `as f32`
/// converts it (every sum is below 2^31), the excess, and where it is
/// positive the table entry at its scaled, truncated index — clamped onto
/// the table's trailing 0 past its end, which is what `unwrap_or(0.0)`
/// reads there; 1 elsewhere.
#[inline(always)]
unsafe fn weights_row_body<S: Simd>(
    win: &[u32],
    out: &mut [f32],
    free: f32,
    scale: f32,
    lut: &[f32],
) {
    unsafe {
        let l = S::LANES;
        let wide = win.len() - win.len() % l;
        let (zero, one) = (S::set1_f32(0.0), S::set1_f32(1.0));
        let (vfree, vscale, end) = (
            S::set1_f32(free),
            S::set1_f32(scale),
            S::set1_f32((lut.len() - 1) as f32),
        );
        let mut i = 0;
        while i < wide {
            let excess = S::sub_f32(
                S::i32_to_f32(S::load_i32(win.as_ptr().add(i) as *const i32)),
                vfree,
            );
            let p = S::min_f32(S::max_f32(S::mul_f32(excess, vscale), zero), end);
            let wt = S::gather_f32(lut, S::trunc_f32_i32(p));
            S::store_f32(
                out.as_mut_ptr().add(i),
                S::blend_f32(wt, one, S::cmpge_f32(zero, excess)),
            );
            i += l;
        }
        weights_row_scalar(&win[wide..], &mut out[wide..], free, scale, lut);
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{noisy, planes};
    use super::*;

    /// The weight row at every tier the host has is the scalar one bit for
    /// bit: sums of zero, at and around the free zone, mid-table, past the
    /// table's end and near the largest a 99x99 patch can reach, at several
    /// strengths, every row length up to 40 (all tails).
    #[test]
    fn every_tier_weighs_like_the_scalar_row() {
        let mut seed = 0x77_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        for (sigma, n) in [(1.0f32, 9usize), (10.0, 9), (30.0, 49), (4.0, 9801)] {
            let wt = Weights::new(sigma, n);
            let free = wt.free as u32;
            for len in 0..=40 {
                let win: Vec<u32> = (0..len)
                    .map(|_| match next() % 6 {
                        0 => 0,
                        1 => free,
                        2 => free + next() % 3,
                        3 => free.saturating_sub(next() % 3),
                        4 => next() % (free * 20 + 1),
                        _ => 637_000_000 - next() % 1000,
                    })
                    .collect();
                let mut want = vec![0f32; len];
                weights_row_scalar(&win, &mut want, wt.free, wt.scale, &wt.lut);
                for (o, &s) in want.iter().zip(&win) {
                    assert_eq!(o.to_bits(), wt.of(s).to_bits());
                }
                for tier in Tier::available() {
                    let mut got = vec![0f32; len];
                    weights_row(tier, &win, &mut got, wt.free, wt.scale, &wt.lut);
                    assert!(
                        got.iter()
                            .zip(&want)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{tier:?} sigma {sigma} len {len}"
                    );
                }
            }
        }
    }

    /// The definition, evaluated directly: every candidate, every patch
    /// sample, no running sums, no symmetry.
    fn direct(src: &[u8], w: usize, h: usize, patch: u32, research: u32, sigma: f32) -> Vec<u8> {
        let Some(k) = Kernel::new(src, w, h, patch, research, sigma) else {
            return src.to_vec();
        };
        let pr = k.pr as isize;
        let at = |x: isize, y: isize| k.pad[(y + pr) as usize * k.pw + (x + pr) as usize] as i32;
        let mut out = vec![0u8; w * h];
        for y in 0..h as isize {
            for x in 0..w as isize {
                // Same accumulation order as the kernel: centre, then for
                // each half-plane offset +o then −o.
                let mut sum = src[y as usize * w + x as usize] as f32;
                let mut wsum = 1.0f32;
                for dy in 0..=k.sry as isize {
                    for dx in -(k.srx as isize)..=k.srx as isize {
                        if dy == 0 && dx <= 0 {
                            continue;
                        }
                        for (ox, oy) in [(dx, dy), (-dx, -dy)] {
                            let (qx, qy) = (x + ox, y + oy);
                            if qx < 0 || qy < 0 || qx >= w as isize || qy >= h as isize {
                                continue;
                            }
                            let mut ssd = 0u32;
                            for uy in -pr..=pr {
                                for ux in -pr..=pr {
                                    let d = at(x + ux, y + uy) - at(qx + ux, qy + uy);
                                    ssd += (d * d) as u32;
                                }
                            }
                            let wt = k.weights.of(ssd);
                            sum += wt * src[qy as usize * w + qx as usize] as f32;
                            wsum += wt;
                        }
                    }
                }
                out[y as usize * w + x as usize] = (sum / wsum).round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    #[test]
    fn the_fast_kernel_equals_the_direct_definition() {
        for (w, h, src) in planes() {
            for (p, r, s) in [
                (3, 5, 10.0),
                (1, 3, 30.0),
                (5, 7, 4.0),
                (7, 9, 1.0),
                (4, 6, 12.0),
            ] {
                assert_eq!(
                    plane_params(&src, w, h, p, r, s),
                    direct(&src, w, h, p, r, s),
                    "{w}x{h} p={p} r={r} s={s}"
                );
            }
        }
    }

    #[test]
    fn splitting_into_bands_does_not_change_the_result() {
        let (w, h) = (37, 53);
        let src = noisy(w * h, 9);
        let k = Kernel::new(&src, w, h, 5, 11, 15.0).unwrap();
        let mut whole = vec![0u8; w * h];
        k.band(0, &mut whole);
        for rows_per_band in [1, 2, 5, 16, 52] {
            let mut split = vec![0u8; w * h];
            for (i, chunk) in split.chunks_mut(rows_per_band * w).enumerate() {
                k.band(i * rows_per_band, chunk);
            }
            assert_eq!(split, whole, "{rows_per_band} rows per band");
        }
    }

    #[test]
    fn single_row_and_single_column_planes_work() {
        let row = noisy(41, 3);
        let out = plane_params(&row, 41, 1, 3, 7, 30.0);
        assert_eq!(out, direct(&row, 41, 1, 3, 7, 30.0));
        assert_ne!(
            out, row,
            "a 1-row plane must still be filtered horizontally"
        );
        let colm = noisy(41, 4);
        assert_eq!(
            plane_params(&colm, 1, 41, 3, 7, 30.0),
            direct(&colm, 1, 41, 3, 7, 30.0)
        );
    }

    #[test]
    fn the_weight_is_one_within_the_noise_floor_and_falls_beyond_it() {
        let wt = Weights::new(10.0, 9);
        // σ² per sample, 9 samples: up to 900 counts as "the same patch".
        assert_eq!(wt.of(0), 1.0);
        assert_eq!(wt.of(900), 1.0);
        assert!(wt.of(901) < 1.0);
        let mut last = 1.0;
        for ssd in (901..9000).step_by(50) {
            let v = wt.of(ssd);
            assert!(v <= last, "weight must not rise with distance");
            last = v;
        }
        assert_eq!(wt.of(u32::MAX), 0.0);
    }
}
