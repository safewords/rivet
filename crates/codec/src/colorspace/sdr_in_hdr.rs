//! An SDR picture placed in an HDR signal (ITU-R BT.2408-7 §5.1): what an
//! HDR output policy does to an SDR source, so the picture looks on an HDR
//! display the way it looked on an SDR one. Re-tagging SDR pixels as PQ or
//! HLG instead plays them as a different, wrong picture.
//!
//! Display-referred, per pixel:
//!
//! 1. Y'CbCr (the source's matrix and range) → R'G'B', clipped to [0, 1].
//! 2. BT.1886 EOTF with a zero black level, `L = V^2.4`: linear display
//!    light, SDR reference white at 1.
//! 3. Linear source primaries → linear BT.2020 (for BT.709 this is the
//!    BT.2087 matrix; every matrix is derived from the chromaticities).
//! 4. SDR reference white placed at HDR reference white, 203 cd/m²
//!    (BT.2408 §2.2):
//!    - PQ: the ST 2084 inverse EOTF of `L · 203 / 10000`;
//!    - HLG: the BT.2100 inverse EOTF of the 1000 cd/m² reference display —
//!      the inverse OOTF (system γ 1.2, on BT.2020 luminance), then the OETF.
//! 5. R'G'B' → BT.2020 non-constant-luminance Y'CbCr, 10-bit limited range.
//!
//! SDR white lands at PQ 58 % and HLG 75 % (10-bit codes 573 and 721),
//! BT.2408's figures. The patch values the tests hold come from an
//! independent float implementation of the same steps, which ffmpeg's zscale
//! (zimg, display-referred, `npl=203`) matches within one code for PQ.
//!
//! Chroma: a 4:2:0 chroma sample is shared by the 2×2 luma block it covers,
//! every luma position is converted, and the output chroma sample is the mean
//! of its block's four converted values. Flat areas are exact.
//!
//! The two transfers are sampled into tables once per converter (the BT.1886
//! EOTF over the signal, PQ over the square root of linear light, where the
//! curve is gentle enough to interpolate) and read with linear interpolation;
//! the tests hold the tables to the formulas.

use anyhow::{Result, bail};
use bytes::Bytes;

use crate::frame::{ColorMetadata, ColorSpace, PixelFormat, TransferFn, VideoFrame};

/// HDR reference white (BT.2408 §2.2), where SDR reference white goes.
const HDR_REFERENCE_WHITE_NITS: f64 = 203.0;
/// Nominal peak of the HLG reference display (BT.2100), whose system gamma
/// is 1.2.
const HLG_DISPLAY_PEAK_NITS: f64 = 1000.0;
const HLG_SYSTEM_GAMMA: f32 = 1.2;
/// BT.1886 display gamma (zero black level).
const SDR_DISPLAY_GAMMA: f64 = 2.4;
/// BT.2020 luma coefficients.
const KR_2020: f32 = 0.2627;
const KB_2020: f32 = 0.0593;
/// Intervals in each transfer table.
const LUT_STEPS: usize = 4096;

/// D65, the white point of every primaries set this maps from.
const D65: (f64, f64) = (0.3127, 0.3290);
/// BT.2020 red, green, blue.
const BT2020_PRIMARIES: [(f64, f64); 3] = [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)];

/// The chromaticities (red, green, blue) of an H.273 `colour_primaries`
/// code, for the codes an SDR source is mapped from. 2 (unspecified) is read
/// as BT.709, as everywhere else in the pipeline.
fn primaries_xy(code: u8) -> Option<[(f64, f64); 3]> {
    Some(match code {
        1 | 2 => [(0.640, 0.330), (0.300, 0.600), (0.150, 0.060)],
        5 => [(0.640, 0.330), (0.290, 0.600), (0.150, 0.060)],
        6 | 7 => [(0.630, 0.340), (0.310, 0.595), (0.155, 0.070)],
        9 => BT2020_PRIMARIES,
        12 => [(0.680, 0.320), (0.265, 0.690), (0.150, 0.060)],
        _ => return None,
    })
}

/// The luma coefficients `(Kr, Kb)` of an H.273 `matrix_coefficients` code.
/// 2 (unspecified) is read as BT.709.
fn luma_coefficients(code: u8) -> Option<(f32, f32)> {
    Some(match code {
        1 | 2 => (0.2126, 0.0722),
        5 | 6 => (0.299, 0.114),
        7 => (0.212, 0.087),
        9 => (KR_2020, KB_2020),
        _ => return None,
    })
}

type Mat3 = [[f64; 3]; 3];

fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn mat_inv(m: &Mat3) -> Mat3 {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let c =
        |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    [
        [
            c(1, 1, 2, 2) / det,
            -c(0, 1, 2, 2) / det,
            c(0, 1, 1, 2) / det,
        ],
        [
            -c(1, 0, 2, 2) / det,
            c(0, 0, 2, 2) / det,
            -c(0, 0, 1, 2) / det,
        ],
        [
            c(1, 0, 2, 1) / det,
            -c(0, 0, 2, 1) / det,
            c(0, 0, 1, 1) / det,
        ],
    ]
}

/// Linear RGB → CIE XYZ for three primaries and a D65 white (SMPTE RP 177).
fn rgb_to_xyz(primaries: [(f64, f64); 3]) -> Mat3 {
    let xyz = |(x, y): (f64, f64)| [x / y, 1.0, (1.0 - x - y) / y];
    let [r, g, b] = primaries.map(xyz);
    let m = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
    let w = xyz(D65);
    let inv = mat_inv(&m);
    let s: [f64; 3] = std::array::from_fn(|i| (0..3).map(|k| inv[i][k] * w[k]).sum());
    std::array::from_fn(|i| std::array::from_fn(|j| m[i][j] * s[j]))
}

/// Linear RGB in `primaries` → linear RGB in BT.2020.
fn to_bt2020(primaries: [(f64, f64); 3]) -> Mat3 {
    mat_mul(
        &mat_inv(&rgb_to_xyz(BT2020_PRIMARIES)),
        &rgb_to_xyz(primaries),
    )
}

/// SMPTE ST 2084 inverse EOTF: absolute luminance / 10000 cd/m² → signal.
fn pq_inverse_eotf(y: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;
    let ym = y.max(0.0).powf(M1);
    ((C1 + C2 * ym) / (1.0 + C3 * ym)).powf(M2)
}

/// ARIB STD-B67 / BT.2100 HLG OETF: normalised scene light → signal.
fn hlg_oetf(e: f32) -> f32 {
    const A: f32 = 0.178_832_77;
    const B: f32 = 1.0 - 4.0 * A;
    const C: f32 = 0.559_910_7;
    let e = e.clamp(0.0, 1.0);
    if e <= 1.0 / 12.0 {
        (3.0 * e).sqrt()
    } else {
        A * (12.0 * e - B).ln() + C
    }
}

/// `table` sampled at `i / LUT_STEPS`, read at `x` in [0, 1] with linear
/// interpolation.
#[inline(always)]
fn lerp_table(table: &[f32], x: f32) -> f32 {
    let p = x.clamp(0.0, 1.0) * LUT_STEPS as f32;
    let i = (p as usize).min(LUT_STEPS - 1);
    let f = p - i as f32;
    table[i] + (table[i + 1] - table[i]) * f
}

/// Maps one SDR source into PQ or HLG BT.2020. Built once per stream: the
/// source's matrix, range and primaries and the target transfer are fixed.
#[derive(Debug, Clone)]
pub struct SdrToHdr {
    target: TransferFn,
    kr: f32,
    kb: f32,
    full_range: bool,
    /// Linear source RGB → linear BT.2020 RGB.
    primaries: [[f32; 3]; 3],
    /// `v^2.4` at `v = i / LUT_STEPS`.
    eotf: Vec<f32>,
    /// PQ signal of SDR-relative linear light `(i / LUT_STEPS)²`.
    pq: Vec<f32>,
}

impl SdrToHdr {
    /// A converter from `source` into `target` (`St2084` or `AribStdB67`).
    ///
    /// Refused by name, before any frame: a source that is not SDR (PQ, HLG,
    /// or linear light, which has no display EOTF to map from), and a matrix
    /// or primaries code outside those this maps (matrix 1, 2, 5, 6, 7, 9;
    /// primaries 1, 2, 5, 6, 7, 9, 12).
    pub fn new(source: &ColorMetadata, target: TransferFn) -> Result<Self> {
        let refuse = |what: String| -> Result<Self> {
            bail!(
                "cannot map this source into {target:?}: {what}. Use `--color sdr` or `--color passthrough` to keep its colour"
            )
        };
        if !matches!(target, TransferFn::St2084 | TransferFn::AribStdB67) {
            bail!("SDR into HDR maps into PQ or HLG, not {target:?}");
        }
        match source.transfer {
            TransferFn::Bt709 | TransferFn::Bt470Bg | TransferFn::Unspecified => {}
            TransferFn::St2084 | TransferFn::AribStdB67 => {
                return refuse(format!(
                    "it is already HDR ({:?}), not SDR",
                    source.transfer
                ));
            }
            TransferFn::Linear => {
                return refuse("its transfer is linear light (H.273 8), which has no SDR display EOTF to map from".into());
            }
        }
        let Some((kr, kb)) = luma_coefficients(source.matrix_coefficients) else {
            return refuse(format!(
                "its matrix_coefficients {} is not one SDR-into-HDR converts (1, 5, 6, 7, 9)",
                source.matrix_coefficients
            ));
        };
        let Some(xy) = primaries_xy(source.colour_primaries) else {
            return refuse(format!(
                "its colour_primaries {} is not one SDR-into-HDR converts (1, 5, 6, 7, 9, 12)",
                source.colour_primaries
            ));
        };
        let m = if source.colour_primaries == 9 {
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        } else {
            to_bt2020(xy)
        };
        let step = |i: usize| i as f64 / LUT_STEPS as f64;
        let eotf = (0..=LUT_STEPS)
            .map(|i| step(i).powf(SDR_DISPLAY_GAMMA) as f32)
            .collect();
        let pq = (0..=LUT_STEPS)
            .map(|i| {
                pq_inverse_eotf(step(i) * step(i) * HDR_REFERENCE_WHITE_NITS / 10_000.0) as f32
            })
            .collect();
        Ok(Self {
            target,
            kr,
            kb,
            full_range: source.full_range,
            primaries: m.map(|row| row.map(|v| v as f32)),
            eotf,
            pq,
        })
    }

    /// One pixel: normalised source Y' in [0, 1] and Cb, Cr in [-0.5, 0.5] →
    /// normalised BT.2020 PQ / HLG Y', Cb, Cr.
    #[inline(always)]
    fn pixel(&self, y: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
        let (kr, kb) = (self.kr, self.kb);
        let r = y + 2.0 * (1.0 - kr) * cr;
        let b = y + 2.0 * (1.0 - kb) * cb;
        let g = (y - kr * r - kb * b) / (1.0 - kr - kb);
        let lin = [r, g, b].map(|v| lerp_table(&self.eotf, v));
        let m = &self.primaries;
        let l: [f32; 3] = std::array::from_fn(|i| {
            (m[i][0] * lin[0] + m[i][1] * lin[1] + m[i][2] * lin[2]).clamp(0.0, 1.0)
        });
        let out = match self.target {
            TransferFn::St2084 => l.map(|v| lerp_table(&self.pq, v.sqrt())),
            _ => {
                // Display light on the 1000 cd/m² reference, then the inverse
                // OOTF: scene light = display light · Yd^((1 − γ) / γ).
                let scale = (HDR_REFERENCE_WHITE_NITS / HLG_DISPLAY_PEAK_NITS) as f32;
                let fd = l.map(|v| v * scale);
                let yd = KR_2020 * fd[0] + (1.0 - KR_2020 - KB_2020) * fd[1] + KB_2020 * fd[2];
                let k = if yd > 0.0 {
                    yd.powf((1.0 - HLG_SYSTEM_GAMMA) / HLG_SYSTEM_GAMMA)
                } else {
                    0.0
                };
                fd.map(|v| hlg_oetf(v * k))
            }
        };
        let yp = KR_2020 * out[0] + (1.0 - KR_2020 - KB_2020) * out[1] + KB_2020 * out[2];
        (
            yp,
            (out[2] - yp) / (2.0 * (1.0 - KB_2020)),
            (out[0] - yp) / (2.0 * (1.0 - KR_2020)),
        )
    }

    /// Convert a `Yuv420p` or `Yuv420p10le` frame into `Yuv420p10le` BT.2020
    /// in the target transfer, limited range.
    pub fn convert(&self, frame: &VideoFrame) -> Result<VideoFrame> {
        self.convert_at(frame, crate::simd::Level::get())
    }

    /// [`Self::convert`] at a given SIMD level. Every level writes the same
    /// bytes: the vector rows do the scalar pixel's IEEE-754 operations in
    /// its order (no fused multiply-add; the table reads are gathers of the
    /// same entries), and a chroma sample's four values are summed in the
    /// same order.
    pub(crate) fn convert_at(&self, frame: &VideoFrame, level: crate::simd::Level) -> Result<VideoFrame> {
        let depth: u32 = match frame.format {
            PixelFormat::Yuv420p => 8,
            PixelFormat::Yuv420p10le => 10,
            other => bail!("SDR into HDR takes Yuv420p or Yuv420p10le, got {other:?}"),
        };
        let bytes = if depth == 8 { 1 } else { 2 };
        let (w, h) = (frame.width as usize, frame.height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let data = &frame.data[..];
        if data.len() < (w * h + 2 * cw * ch) * bytes {
            bail!(
                "SDR into HDR: a {w}x{h} {:?} frame needs {} bytes, got {}",
                frame.format,
                (w * h + 2 * cw * ch) * bytes,
                data.len()
            );
        }
        let sample = |at: usize| -> f32 {
            if bytes == 1 {
                data[at] as f32
            } else {
                u16::from_le_bytes([data[2 * at], data[2 * at + 1]]) as f32
            }
        };
        let max = ((1u32 << depth) - 1) as f32;
        let mid = (1u32 << (depth - 1)) as f32;
        let (black, y_span, c_span) = if self.full_range {
            (0.0, max, max)
        } else {
            let s = (1u32 << (depth - 8)) as f32;
            (16.0 * s, 219.0 * s, 224.0 * s)
        };
        let (cb_at, cr_at) = (w * h, w * h + cw * ch);
        let rows = Rows { data, bytes, w, cw, cb_at, cr_at, black, y_span, mid, c_span };

        let mut y_code = vec![0u16; w * h];
        let mut cb_sum = vec![0f32; cw * ch];
        let mut cr_sum = vec![0f32; cw * ch];
        for py in 0..h {
            let (yc, cbs, crs) = (&mut y_code[py * w..(py + 1) * w], &mut cb_sum[(py / 2) * cw..], &mut cr_sum[(py / 2) * cw..]);
            #[allow(unused_mut)]
            let mut done = 0;
            #[cfg(target_arch = "x86_64")]
            // PQ only: HLG's inverse OOTF and OETF take `powf` and `ln`, which
            // have no lane-wise twin that rounds the way the C library does,
            // and calling them lane by lane from the vector rows measured
            // slower than the scalar loop (117 ms against 45 at 1080p).
            if level >= crate::simd::Level::Avx2 && self.target == TransferFn::St2084 {
                // SAFETY: the level is only ever Avx2 or above on a CPU that
                // has AVX2 (`Level::host`); `row_avx2` reads and writes only
                // inside the planes checked above and the row slices.
                done = unsafe { self.row_avx2(&rows, py, yc, cbs, crs) };
            }
            #[cfg(not(target_arch = "x86_64"))]
            let _ = level;
            for px in done..w {
                let c = (py / 2) * cw + px / 2;
                let yn = (sample(py * w + px) - black) / y_span;
                let cbn = (sample(cb_at + c) - mid) / c_span;
                let crn = (sample(cr_at + c) - mid) / c_span;
                let (yo, cbo, cro) = self.pixel(yn, cbn, crn);
                yc[px] = code(876.0 * yo + 64.0);
                cbs[px / 2] += cbo;
                crs[px / 2] += cro;
            }
        }
        let mut out = Vec::with_capacity((w * h + 2 * cw * ch) * 2);
        out.extend(y_code.iter().flat_map(|v| v.to_le_bytes()));
        for sums in [&cb_sum, &cr_sum] {
            for cy in 0..ch {
                for cx in 0..cw {
                    let n = ((w - 2 * cx).min(2) * (h - 2 * cy).min(2)) as f32;
                    out.extend_from_slice(&code(896.0 * sums[cy * cw + cx] / n + 512.0).to_le_bytes());
                }
            }
        }
        Ok(VideoFrame::new(
            Bytes::from(out),
            frame.width,
            frame.height,
            PixelFormat::Yuv420p10le,
            ColorSpace::Bt2020,
            frame.pts,
        ))
    }

    /// The first `w & !7` samples of luma row `py` of a PQ conversion,
    /// eight at a time, as the scalar loop in [`Self::convert_at`] does them: their 10-bit luma
    /// codes into `y_code`, their converted chroma added into the row's
    /// `cb_sum` / `cr_sum` (each pair in pixel order). Returns how many it
    /// did; a chunk of eight never splits a chroma pair.
    ///
    /// # Safety
    /// AVX2. `rows` describes planes that hold the whole frame (as
    /// `convert_at` checks), `y_code` is the row's `w` codes and the sums
    /// start at the row's first chroma sample.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn row_avx2(&self, rows: &Rows<'_>, py: usize, y_code: &mut [u16], cb_sum: &mut [f32], cr_sum: &mut [f32]) -> usize {
        use std::arch::x86_64::*;
        unsafe {
            let w = rows.w;
            let wide = w & !7;
            let c0 = (py / 2) * rows.cw;
            // Eight samples at index `at` of a plane, as f32 (exact).
            let load8 = |at: usize| -> __m256 {
                let v = if rows.bytes == 1 {
                    _mm256_cvtepu8_epi32(_mm_loadl_epi64(rows.data.as_ptr().add(at) as *const __m128i))
                } else {
                    _mm256_cvtepu16_epi32(_mm_loadu_si128(rows.data.as_ptr().add(2 * at) as *const __m128i))
                };
                _mm256_cvtepi32_ps(v)
            };
            // Four chroma samples at `at`, each twice: the eight pixels' own.
            let load4x2 = |at: usize| -> __m256 {
                let v = if rows.bytes == 1 {
                    _mm_cvtepu8_epi32(_mm_cvtsi32_si128((rows.data.as_ptr().add(at) as *const i32).read_unaligned()))
                } else {
                    _mm_cvtepu16_epi32(_mm_loadl_epi64(rows.data.as_ptr().add(2 * at) as *const __m128i))
                };
                let f = _mm256_castps128_ps256(_mm_cvtepi32_ps(v));
                _mm256_permutevar8x32_ps(f, _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3))
            };
            let set = _mm256_set1_ps;
            let (black, y_span, mid, c_span) = (set(rows.black), set(rows.y_span), set(rows.mid), set(rows.c_span));
            let (kr, kb) = (self.kr, self.kb);
            // The scalar pixel's constants, computed as it computes them.
            let (k_r, k_b, k_g) = (set(2.0 * (1.0 - kr)), set(2.0 * (1.0 - kb)), set(1.0 - kr - kb));
            let (zero, one) = (_mm256_setzero_ps(), set(1.0));
            let lerp = |table: &[f32], x: __m256| -> __m256 {
                let p = _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(x, zero), one), set(LUT_STEPS as f32));
                let i = _mm256_min_epi32(_mm256_cvttps_epi32(p), _mm256_set1_epi32(LUT_STEPS as i32 - 1));
                let f = _mm256_sub_ps(p, _mm256_cvtepi32_ps(i));
                let a = _mm256_i32gather_ps::<4>(table.as_ptr(), i);
                let b = _mm256_i32gather_ps::<4>(table.as_ptr().add(1), i);
                _mm256_add_ps(a, _mm256_mul_ps(_mm256_sub_ps(b, a), f))
            };
            let m = self.primaries;
            let (kr20, kg20, kb20) = (set(KR_2020), set(1.0 - KR_2020 - KB_2020), set(KB_2020));
            let (dcb, dcr) = (set(2.0 * (1.0 - KB_2020)), set(2.0 * (1.0 - KR_2020)));
            let mut x = 0;
            while x < wide {
                let c = c0 + x / 2;
                let y = _mm256_div_ps(_mm256_sub_ps(load8(py * w + x), black), y_span);
                let cb = _mm256_div_ps(_mm256_sub_ps(load4x2(rows.cb_at + c), mid), c_span);
                let cr = _mm256_div_ps(_mm256_sub_ps(load4x2(rows.cr_at + c), mid), c_span);
                // `pixel`, step by step.
                let r = _mm256_add_ps(y, _mm256_mul_ps(k_r, cr));
                let b = _mm256_add_ps(y, _mm256_mul_ps(k_b, cb));
                let g = _mm256_div_ps(_mm256_sub_ps(_mm256_sub_ps(y, _mm256_mul_ps(set(kr), r)), _mm256_mul_ps(set(kb), b)), k_g);
                let lin = [lerp(&self.eotf, r), lerp(&self.eotf, g), lerp(&self.eotf, b)];
                let l: [__m256; 3] = std::array::from_fn(|i| {
                    let s = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(set(m[i][0]), lin[0]), _mm256_mul_ps(set(m[i][1]), lin[1])),
                        _mm256_mul_ps(set(m[i][2]), lin[2]),
                    );
                    _mm256_min_ps(_mm256_max_ps(s, zero), one)
                });
                let out = l.map(|v| lerp(&self.pq, _mm256_sqrt_ps(v)));
                let yp = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(kr20, out[0]), _mm256_mul_ps(kg20, out[1])), _mm256_mul_ps(kb20, out[2]));
                let cbo = _mm256_div_ps(_mm256_sub_ps(out[2], yp), dcb);
                let cro = _mm256_div_ps(_mm256_sub_ps(out[0], yp), dcr);
                // Luma codes: `round` (half away from zero), clamp, narrow.
                let v = _mm256_add_ps(_mm256_mul_ps(set(876.0), yp), set(64.0));
                let t = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(v);
                let half = _mm256_cmp_ps::<_CMP_GE_OQ>(_mm256_andnot_ps(set(-0.0), _mm256_sub_ps(v, t)), set(0.5));
                let away = _mm256_and_ps(half, _mm256_or_ps(_mm256_and_ps(v, set(-0.0)), one));
                let q = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(t, away), zero), set(1023.0));
                let q = _mm256_cvttps_epi32(q);
                let q = _mm_packus_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256::<1>(q));
                _mm_storeu_si128(y_code.as_mut_ptr().add(x) as *mut __m128i, q);
                // Chroma: the even pixel's value first, then the odd one's.
                let (even, odd) = (_mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0), _mm256_setr_epi32(1, 3, 5, 7, 0, 0, 0, 0));
                for (sums, v) in [(&mut *cb_sum, cbo), (&mut *cr_sum, cro)] {
                    let p = sums.as_mut_ptr().add(x / 2);
                    let e = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(v, even));
                    let o = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(v, odd));
                    _mm_storeu_ps(p, _mm_add_ps(_mm_add_ps(_mm_loadu_ps(p), e), o));
                }
                x += 8;
            }
            wide
        }
    }
}

/// What the row kernels read: the frame's samples (`bytes` per sample),
/// its width, chroma width and plane offsets, and the source's range.
#[allow(dead_code)]
struct Rows<'a> {
    data: &'a [u8],
    bytes: usize,
    w: usize,
    cw: usize,
    cb_at: usize,
    cr_at: usize,
    black: f32,
    y_span: f32,
    mid: f32,
    c_span: f32,
}

/// A 10-bit code from a normalised value already scaled to codes:
/// rounded half away from zero and clamped, as `f32::round` and a clamp.
fn code(v: f32) -> u16 {
    v.round().clamp(0.0, 1023.0) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sdr(matrix: u8, primaries: u8, full_range: bool) -> ColorMetadata {
        ColorMetadata {
            matrix_coefficients: matrix,
            colour_primaries: primaries,
            full_range,
            ..Default::default()
        }
    }

    /// A 4:2:0 frame of flat 16x16 patches, one per `(Y, Cb, Cr)`.
    fn patches(values: &[(u16, u16, u16)], ten_bit: bool) -> VideoFrame {
        let w = 16 * values.len();
        let mut planes: [Vec<u16>; 3] = Default::default();
        for _ in 0..16 {
            for v in values {
                planes[0].extend(std::iter::repeat_n(v.0, 16));
            }
        }
        for _ in 0..8 {
            for v in values {
                planes[1].extend(std::iter::repeat_n(v.1, 8));
                planes[2].extend(std::iter::repeat_n(v.2, 8));
            }
        }
        let data: Vec<u8> = planes
            .iter()
            .flatten()
            .flat_map(|&s| {
                if ten_bit {
                    s.to_le_bytes().to_vec()
                } else {
                    vec![s as u8]
                }
            })
            .collect();
        let format = if ten_bit {
            PixelFormat::Yuv420p10le
        } else {
            PixelFormat::Yuv420p
        };
        VideoFrame::new(
            Bytes::from(data),
            w as u32,
            16,
            format,
            ColorSpace::Bt709,
            7,
        )
    }

    /// Each patch's centre `(Y, Cb, Cr)` of a converted patch frame.
    fn centres(frame: &VideoFrame, count: usize) -> Vec<(u16, u16, u16)> {
        let at = |i: usize| u16::from_le_bytes([frame.data[2 * i], frame.data[2 * i + 1]]);
        let w = frame.width as usize;
        let (cw, ch) = (w / 2, 8);
        (0..count)
            .map(|i| {
                (
                    at(8 * w + 16 * i + 8),
                    at(w * 16 + 4 * cw + 8 * i + 4),
                    at(w * 16 + cw * ch + 4 * cw + 8 * i + 4),
                )
            })
            .collect()
    }

    fn assert_within_one(name: &str, got: &[(u16, u16, u16)], want: &[(u16, u16, u16)]) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let d = [g.0.abs_diff(w.0), g.1.abs_diff(w.1), g.2.abs_diff(w.2)];
            assert!(
                d.iter().all(|&d| d <= 1),
                "{name} patch {i}: got {g:?}, reference {w:?}"
            );
        }
    }

    /// 8-bit limited BT.709 75 % bars plus white, grey, red and near-black —
    /// the input, and the 10-bit output of the independent reference
    /// (`bt2408_ref.py`: PQ within one code of zscale `npl=203` on every one).
    const BARS_709: [(u16, u16, u16); 12] = [
        (16, 128, 128),
        (235, 128, 128),
        (126, 128, 128),
        (180, 128, 128),
        (168, 44, 136),
        (145, 147, 44),
        (133, 63, 52),
        (63, 193, 204),
        (51, 109, 212),
        (28, 212, 120),
        (63, 102, 240),
        (27, 128, 128),
    ];
    const BARS_709_PQ: [(u16, u16, u16); 12] = [
        (64, 512, 512),
        (573, 512, 512),
        (429, 512, 512),
        (510, 512, 512),
        (498, 421, 518),
        (484, 525, 472),
        (469, 430, 475),
        (367, 586, 587),
        (342, 446, 601),
        (239, 655, 536),
        (392, 438, 608),
        (129, 512, 512),
    ];
    const BARS_709_HLG: [(u16, u16, u16); 12] = [
        (64, 512, 512),
        (721, 512, 512),
        (456, 512, 512),
        (618, 512, 512),
        (594, 327, 524),
        (564, 543, 418),
        (536, 352, 423),
        (359, 666, 666),
        (323, 418, 689),
        (193, 776, 528),
        (392, 395, 715),
        (103, 512, 512),
    ];

    /// Every SIMD level the host has writes the scalar reference's bytes:
    /// random pictures (with values past the nominal range), 8 and 10 bits,
    /// both transfers, limited and full range, BT.709 and BT.601 sources,
    /// and widths that leave a scalar tail and an odd chroma column.
    #[test]
    fn every_simd_level_writes_the_scalar_bytes() {
        let mut seed = 0x5d0_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let levels = [crate::simd::Level::Avx2, crate::simd::Level::Avx512].into_iter().filter(|&l| l <= crate::simd::Level::host());
        let mut checked = 0;
        for level in levels {
            for &(w, h) in &[(16usize, 4usize), (37, 5), (64, 3), (8, 8), (7, 3)] {
                for ten in [false, true] {
                    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
                    let n = w * h + 2 * cw * ch;
                    let data: Vec<u8> = if ten {
                        (0..n).flat_map(|_| ((next() % 1024) as u16).to_le_bytes()).collect()
                    } else {
                        (0..n).map(|_| next() as u8).collect()
                    };
                    let format = if ten { PixelFormat::Yuv420p10le } else { PixelFormat::Yuv420p };
                    let frame = VideoFrame::new(Bytes::from(data), w as u32, h as u32, format, ColorSpace::Bt709, 3);
                    for target in [TransferFn::St2084, TransferFn::AribStdB67] {
                        for (matrix, primaries, full) in [(1, 1, false), (6, 6, false), (1, 1, true), (9, 9, false)] {
                            let c = SdrToHdr::new(&sdr(matrix, primaries, full), target).unwrap();
                            let want = c.convert_at(&frame, crate::simd::Level::Scalar).unwrap();
                            let got = c.convert_at(&frame, level).unwrap();
                            assert!(got.data == want.data, "{level:?} {w}x{h} ten={ten} {target:?} matrix {matrix} full {full}");
                            checked += 1;
                        }
                    }
                }
            }
        }
        if crate::simd::Level::host() >= crate::simd::Level::Avx2 {
            assert!(checked > 0);
        }
    }

    #[test]
    fn sdr_white_lands_on_hdr_reference_white() {
        let white = patches(&[(235, 128, 128)], false);
        let pq = SdrToHdr::new(&sdr(1, 1, false), TransferFn::St2084)
            .unwrap()
            .convert(&white)
            .unwrap();
        assert_eq!(centres(&pq, 1), [(573, 512, 512)], "PQ 58 %");
        let hlg = SdrToHdr::new(&sdr(1, 1, false), TransferFn::AribStdB67)
            .unwrap()
            .convert(&white)
            .unwrap();
        assert_eq!(centres(&hlg, 1), [(721, 512, 512)], "HLG 75 %");
        assert_eq!(
            (pq.format, pq.color_space, pq.pts),
            (PixelFormat::Yuv420p10le, ColorSpace::Bt2020, 7)
        );
    }

    #[test]
    fn bars_match_the_reference() {
        let bars = patches(&BARS_709, false);
        for (target, want, name) in [
            (TransferFn::St2084, BARS_709_PQ, "PQ"),
            (TransferFn::AribStdB67, BARS_709_HLG, "HLG"),
        ] {
            let out = SdrToHdr::new(&sdr(1, 1, false), target)
                .unwrap()
                .convert(&bars)
                .unwrap();
            assert_within_one(name, &centres(&out, 12), &want);
        }
    }

    /// The source's matrix, depth and range are undone before anything else:
    /// the same colours coded BT.601, 10-bit, or full range convert to the
    /// same HDR values (reference: `bt2408_ref.py --601` / `--full`).
    #[test]
    fn matrix_depth_and_range_are_the_sources() {
        let bt601 = [
            (162, 44, 142),
            (131, 156, 44),
            (81, 90, 240),
            (235, 128, 128),
        ];
        let out = SdrToHdr::new(&sdr(6, 1, false), TransferFn::St2084)
            .unwrap()
            .convert(&patches(&bt601, false))
            .unwrap();
        assert_within_one(
            "601 PQ",
            &centres(&out, 4),
            &[
                (499, 421, 518),
                (484, 525, 472),
                (392, 438, 608),
                (573, 512, 512),
            ],
        );

        let ten: Vec<_> = BARS_709
            .iter()
            .map(|&(y, cb, cr)| (y << 2, cb << 2, cr << 2))
            .collect();
        let out = SdrToHdr::new(&sdr(1, 1, false), TransferFn::St2084)
            .unwrap()
            .convert(&patches(&ten, true))
            .unwrap();
        assert_within_one("10-bit PQ", &centres(&out, 12), &BARS_709_PQ);

        let full = [(255, 128, 128), (0, 128, 128), (191, 128, 128)];
        let out = SdrToHdr::new(&sdr(1, 1, true), TransferFn::AribStdB67)
            .unwrap()
            .convert(&patches(&full, false))
            .unwrap();
        assert_within_one(
            "full-range HLG",
            &centres(&out, 3),
            &[(721, 512, 512), (64, 512, 512), (618, 512, 512)],
        );
    }

    #[test]
    fn bt709_primaries_map_by_the_bt2087_matrix() {
        let m = to_bt2020(primaries_xy(1).unwrap());
        let bt2087 = [
            [0.6274, 0.3293, 0.0433],
            [0.0691, 0.9195, 0.0114],
            [0.0164, 0.0880, 0.8956],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m[i][j] - bt2087[i][j]).abs() < 5e-5,
                    "[{i}][{j}] {} vs {}",
                    m[i][j],
                    bt2087[i][j]
                );
            }
        }
        let id = to_bt2020(BT2020_PRIMARIES);
        for (i, row) in id.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-9);
            }
        }
    }

    /// The interpolated tables stay within half a 10-bit code of the curves
    /// they sample, over the whole domain.
    #[test]
    fn the_tables_hold_to_the_formulas() {
        let c = SdrToHdr::new(&sdr(1, 1, false), TransferFn::St2084).unwrap();
        let mut worst_pq = 0f64;
        for i in 0..=100_000 {
            let x = i as f64 / 100_000.0;
            let pq_exact = pq_inverse_eotf(x * HDR_REFERENCE_WHITE_NITS / 10_000.0);
            worst_pq = worst_pq
                .max((lerp_table(&c.pq, (x as f32).sqrt()) as f64 - pq_exact).abs() * 876.0);
        }
        assert!(worst_pq < 0.5, "PQ table off by {worst_pq} codes");
        let mut worst_eotf = 0f64;
        for i in 0..=100_000 {
            let v = i as f64 / 100_000.0;
            worst_eotf = worst_eotf.max((lerp_table(&c.eotf, v as f32) as f64 - v.powf(2.4)).abs());
        }
        assert!(worst_eotf < 1e-5, "EOTF table off by {worst_eotf}");
    }

    #[test]
    fn what_it_cannot_map_is_refused_by_name() {
        let err = |source: ColorMetadata, target| {
            format!("{:#}", SdrToHdr::new(&source, target).unwrap_err())
        };
        let pq_source = ColorMetadata {
            transfer: TransferFn::St2084,
            ..sdr(9, 9, false)
        };
        assert!(err(pq_source, TransferFn::AribStdB67).contains("already HDR"));
        let linear = ColorMetadata {
            transfer: TransferFn::Linear,
            ..sdr(1, 1, false)
        };
        assert!(err(linear, TransferFn::St2084).contains("linear light"));
        assert!(err(sdr(0, 1, false), TransferFn::St2084).contains("matrix_coefficients 0"));
        assert!(err(sdr(1, 22, false), TransferFn::St2084).contains("colour_primaries 22"));
        assert!(err(sdr(1, 1, false), TransferFn::Bt709).contains("PQ or HLG"));
        assert!(err(sdr(0, 1, false), TransferFn::St2084).contains("--color passthrough"));
    }
}
