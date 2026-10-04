//! Anisotropic diffusion (Perona–Malik) denoise — edge-preserving.

use super::for_row_bands;

/// Iterations, step and edge threshold of [`plane`].
const ITERS: usize = 8;
const KAPPA: f32 = 20.0;
const LAMBDA: f32 = 0.20;

/// The conduction `g(∇) = exp(−(∇/κ)²)`.
#[inline(always)]
fn g(grad: f32) -> f32 {
    let q = grad / KAPPA;
    (-(q * q)).exp()
}

/// **Anisotropic diffusion** (Perona–Malik): iterate `u += λ·Σ g(∇)·∇` over the
/// 4-neighbour gradients, where the conduction `g(∇) = exp(−(∇/κ)²)` falls to
/// ~0 at strong gradients — so the image diffuses (smooths) inside flat regions
/// but the flow stops at edges. 8 iterations, `λ = 0.20` (≤ ¼ for 4-neighbour
/// stability), `κ = 20`. Border uses edge-replicate.
///
/// Each iteration is [`plane_reference`]'s, value for value, computed with
/// half its exponentials: the flow across an edge between two samples is one
/// number, `f = g(d)·d` with `d` the difference, which one of them gains and
/// the other loses — and `g(−d)·(−d)` is exactly `−(g(d)·d)` in IEEE
/// arithmetic (negation is exact and `g` sees only `d²`), as `a − b` is
/// exactly `−(b − a)`. So the flows east and south are computed once per
/// sample, the west and north ones read as their negations (a zero flow may
/// change sign, which no sum it enters can tell), and they are added in the
/// reference's order. Rows are split into bands across threads; an
/// iteration reads only the previous one, so the bands cannot change it.
pub(super) fn plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut img: Vec<f32> = src.iter().map(|&v| v as f32).collect();
    let mut next = img.clone();
    for _ in 0..ITERS {
        let cur = &img;
        for_row_bands(&mut next, w, 32, |y0, rows| {
            let y1 = y0 + rows.len() / w;
            // flow_s[r] is the flow from row y0 - 1 + r into the row below it
            // (rows y0 - 1 ..= y1 - 1), flow_e the flow east within a row.
            let s_rows = y0.saturating_sub(1)..y1;
            let mut flow_s = vec![0f32; s_rows.len() * w];
            for (r, y) in s_rows.clone().enumerate() {
                if y + 1 < h {
                    for x in 0..w {
                        let d = cur[(y + 1) * w + x] - cur[y * w + x];
                        flow_s[r * w + x] = g(d) * d;
                    }
                }
            }
            let mut flow_e = vec![0f32; w];
            for (yy, row) in rows.chunks_exact_mut(w).enumerate() {
                let y = y0 + yy;
                let line = &cur[y * w..(y + 1) * w];
                for x in 0..w.saturating_sub(1) {
                    let d = line[x + 1] - line[x];
                    flow_e[x] = g(d) * d;
                }
                // The replicated border: the difference there is 0 and so is
                // the flow (`g(0)·0`).
                if w > 0 {
                    flow_e[w - 1] = 0.0;
                }
                let r = y - s_rows.start;
                for x in 0..w {
                    let n = if y > 0 { -flow_s[(r - 1) * w + x] } else { 0.0 };
                    let s = flow_s[r * w + x];
                    let e = flow_e[x];
                    let we = if x > 0 { -flow_e[x - 1] } else { 0.0 };
                    row[x] = line[x] + LAMBDA * (n + s + e + we);
                }
            }
        });
        std::mem::swap(&mut img, &mut next);
    }
    img.iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// The direct form: four exponentials per sample per iteration, one thread.
/// The specification [`plane`] is tested against.
#[cfg(test)]
pub(super) fn plane_reference(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    use super::clamp_idx;
    let mut img: Vec<f32> = src.iter().map(|&v| v as f32).collect();
    let mut next = img.clone();
    for _ in 0..ITERS {
        for y in 0..h {
            for x in 0..w {
                let c = img[y * w + x];
                let n = img[clamp_idx(y as isize - 1, h) * w + x] - c;
                let s = img[clamp_idx(y as isize + 1, h) * w + x] - c;
                let e = img[y * w + clamp_idx(x as isize + 1, w)] - c;
                let we = img[y * w + clamp_idx(x as isize - 1, w)] - c;
                next[y * w + x] = c + LAMBDA * (g(n) * n + g(s) * s + g(e) * e + g(we) * we);
            }
        }
        std::mem::swap(&mut img, &mut next);
    }
    img.iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shared-flow, banded form writes the direct form's bytes: noise,
    /// rails, a step edge, and single rows / columns.
    #[test]
    fn shared_flows_match_the_direct_form() {
        let mut seed = 0xa51_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        for &(w, h) in &[
            (1usize, 1usize),
            (1, 9),
            (9, 1),
            (7, 5),
            (64, 48),
            (97, 133),
        ] {
            for kind in 0..3 {
                let src: Vec<u8> = (0..w * h)
                    .map(|i| match kind {
                        0 => next() as u8,
                        1 => [0u8, 255][(next() & 1) as usize],
                        _ => {
                            if i % w < w / 2 {
                                40
                            } else {
                                200
                            }
                        }
                    })
                    .collect();
                assert_eq!(
                    plane(&src, w, h),
                    plane_reference(&src, w, h),
                    "{w}x{h} kind {kind}"
                );
            }
        }
    }
}
