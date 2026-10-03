//! Sample-rate conversion: rivet's own band-limited (windowed-sinc)
//! resampler, which an encoder whose coding rate is not the input's puts in
//! front of itself — 44.1 kHz into Opus's 48, 96 kHz into AAC's, AC-3's or
//! DTS's 48, and so on.
//!
//! # The filter
//!
//! Output sample `k` is the input evaluated at the input time
//! `t = k · in_rate / out_rate` through an ideal low-pass filter, truncated
//! and windowed: `y[k] = Σ x[n] · h(t − n)`, with
//! `h(τ) = 2·fc · sinc(2·fc·τ) · w(τ / W)` — `fc` the cut-off in cycles per
//! input sample, `w` a Kaiser window of half-width `W` input samples.
//!
//! - **No delay.** The kernel is centred on `t` itself (it looks `W` input
//!   samples ahead, which the streaming below buffers), so output sample `k`
//!   *is* the input at `k / out_rate` seconds: an impulse at input sample
//!   `n` comes out centred on output time `n · out_rate / in_rate` exactly,
//!   whether or not that is a whole output sample, and the output starts
//!   with the input — there is no lead-in to trim and nothing to round.
//! - **Flat pass band.** Pass band to 90 % of the lower rate's Nyquist
//!   frequency (19.8 kHz at 44.1 kHz, 21.6 kHz at 48), stop band from 100 %
//!   (nothing above the new Nyquist frequency folds back into the pass
//!   band), 120 dB of stop-band rejection (Kaiser's design formulas:
//!   β = 0.1102·(A − 8.7), length (A − 7.95) / (14.36·Δf)). A Kaiser-windowed
//!   sinc ripples by about 10^(−A/20) in both bands: some ±0.00001 dB in the
//!   pass band. Each phase of the kernel is normalised to sum to 1, so the
//!   gain at DC is exactly unity whatever the phase.
//! - **Exact phases.** For rates whose ratio reduces to `P / Q` with `Q` at
//!   most [`MAX_EXACT_PHASES`] (every pair of the usual rates: 44.1 → 48 is
//!   147 / 160), the kernel is tabulated at each of the `Q` phases an output
//!   sample can fall on; for other ratios at 1024 phases, interpolated
//!   linearly between them.
//!
//! # Length
//!
//! Exactly as many samples come out as the input's length at the output
//! rate, rounded up (`ceil(n · out / in)`): the samples whose time falls
//! inside the input. With equal rates the samples pass through untouched.

use crate::audio::{AudioError, AudioFrame};

/// Stop-band rejection, dB.
const ATTENUATION_DB: f64 = 120.0;
/// Pass-band edge, as a fraction of the lower rate's Nyquist frequency.
const PASS_EDGE: f64 = 0.90;
/// Stop-band edge, likewise.
const STOP_EDGE: f64 = 1.0;
/// The most phases tabulated exactly (the reduced ratio's denominator);
/// above it, [`INTERPOLATED_PHASES`] interpolated linearly.
pub const MAX_EXACT_PHASES: u64 = 4096;
/// Phases tabulated for a ratio with more than [`MAX_EXACT_PHASES`].
const INTERPOLATED_PHASES: usize = 1024;

/// The polyphase kernel for one pair of rates.
struct Kernel {
    /// Half-width `W`, input samples: output `k` at input time `n0 + φ`
    /// (`φ` in [0, 1)) reads inputs `n0 − W + 1 ..= n0 + W`.
    half: usize,
    /// `in / out` reduced: `step_num / step_den` input samples per output
    /// sample.
    step_num: u64,
    step_den: u64,
    /// Whether the table holds every phase (`step_den` rows) or
    /// `INTERPOLATED_PHASES + 1` rows to interpolate between.
    exact: bool,
    /// Row-major, `2 · half` taps a row.
    table: Vec<f32>,
}

impl Kernel {
    fn new(in_rate: u32, out_rate: u32) -> Self {
        let g = gcd(u64::from(in_rate), u64::from(out_rate));
        let (step_num, step_den) = (u64::from(in_rate) / g, u64::from(out_rate) / g);
        // Frequencies in cycles per input sample.
        let nyquist = 0.5 * (f64::from(out_rate) / f64::from(in_rate)).min(1.0);
        let (pass, stop) = (PASS_EDGE * nyquist, STOP_EDGE * nyquist);
        let fc = 0.5 * (pass + stop);
        let taps = ((ATTENUATION_DB - 7.95) / (14.36 * (stop - pass))).ceil() as usize;
        let half = taps.div_ceil(2).max(2);
        let beta = 0.1102 * (ATTENUATION_DB - 8.7);
        let exact = step_den <= MAX_EXACT_PHASES;
        let rows = if exact { step_den as usize } else { INTERPOLATED_PHASES + 1 };
        let width = 2 * half;
        let mut table = vec![0f32; rows * width];
        let i0_beta = bessel_i0(beta);
        let mut row = vec![0f64; width];
        for (r, out) in table.chunks_exact_mut(width).enumerate() {
            let phase = if exact { r as f64 / step_den as f64 } else { r as f64 / INTERPOLATED_PHASES as f64 };
            for (j, v) in row.iter_mut().enumerate() {
                // Tap j reads input n0 + j − (half − 1): τ = φ − that offset.
                let tau = phase - (j as f64 - (half as f64 - 1.0));
                let x = tau / half as f64;
                *v = if x.abs() >= 1.0 {
                    0.0
                } else {
                    2.0 * fc * sinc(2.0 * fc * tau) * bessel_i0(beta * (1.0 - x * x).sqrt()) / i0_beta
                };
            }
            let sum: f64 = row.iter().sum();
            for (o, v) in out.iter_mut().zip(&row) {
                *o = (v / sum) as f32;
            }
        }
        Kernel { half, step_num, step_den, exact, table }
    }

    /// Output `k`'s first input index (`n0 − W + 1`) and its taps (built in
    /// `scratch` when interpolated).
    fn taps<'a>(&'a self, k: u64, scratch: &'a mut Vec<f32>) -> (i64, &'a [f32]) {
        let pos = u128::from(k) * u128::from(self.step_num);
        let n0 = (pos / u128::from(self.step_den)) as i64;
        let rem = (pos % u128::from(self.step_den)) as u64;
        let first = n0 - self.half as i64 + 1;
        let width = 2 * self.half;
        if self.exact {
            let r = rem as usize;
            return (first, &self.table[r * width..(r + 1) * width]);
        }
        let at = rem as f64 / self.step_den as f64 * INTERPOLATED_PHASES as f64;
        let r = (at as usize).min(INTERPOLATED_PHASES - 1);
        let t = (at - r as f64) as f32;
        let a = &self.table[r * width..(r + 1) * width];
        let b = &self.table[(r + 1) * width..(r + 2) * width];
        scratch.clear();
        scratch.extend(a.iter().zip(b).map(|(&a, &b)| a + t * (b - a)));
        (first, scratch)
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// The modified Bessel function of the first kind, order 0, by its power
/// series (which converges quickly for the window's arguments, up to
/// β ≈ 12.3).
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, q) = (1.0, 1.0, x * x / 4.0);
    for k in 1..200 {
        term *= q / (k as f64 * k as f64);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// A resampler whose output lines up with its input: output sample `k` is
/// the input at time `k / out_rate`, with no delay (see the module notes),
/// and at the end exactly as many samples come out as the input's length at
/// the output rate (rounded up). What an encoder whose coding rate is not
/// the input's puts in front of itself, so the stream's priming is the
/// codec's own alone. With equal rates it passes the samples through
/// untouched.
pub struct AlignedResampler {
    kernel: Option<Kernel>,
    in_rate: u32,
    out_rate: u32,
    channels: u8,
    /// Input history per channel; `hist[c][0]` is input sample `base`
    /// (negative before the start: the silence before the input).
    hist: Vec<Vec<f32>>,
    base: i64,
    /// Input and output samples per channel so far.
    samples_in: u64,
    samples_out: u64,
    scratch: Vec<f32>,
}

impl AlignedResampler {
    /// From `in_rate` to `out_rate`, `channels` interleaved (1 to 8).
    pub fn new(in_rate: u32, out_rate: u32, channels: u8) -> Result<Self, AudioError> {
        if in_rate == 0 || out_rate == 0 {
            return Err(AudioError::Resample(format!("invalid sample rate {in_rate} -> {out_rate}")));
        }
        if channels == 0 || channels > 8 {
            return Err(AudioError::Unsupported(format!("resampler channel count {channels} (must be 1..=8)")));
        }
        let kernel = (in_rate != out_rate).then(|| Kernel::new(in_rate, out_rate));
        // The silence before the input: enough for output 0's taps.
        let lead = kernel.as_ref().map_or(0, |k| k.half);
        Ok(Self {
            kernel,
            in_rate,
            out_rate,
            channels,
            hist: vec![vec![0.0; lead]; usize::from(channels)],
            base: -(lead as i64),
            samples_in: 0,
            samples_out: 0,
            scratch: Vec::new(),
        })
    }

    /// Whether the rates differ (anything is resampled at all).
    pub fn is_active(&self) -> bool {
        self.kernel.is_some()
    }

    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }

    pub fn out_rate(&self) -> u32 {
        self.out_rate
    }

    /// The input's samples per channel so far, at the output rate, rounded up:
    /// what the output holds once [`Self::flush`] has run.
    pub fn target_len(&self) -> u64 {
        (u128::from(self.samples_in) * u128::from(self.out_rate)).div_ceil(u128::from(self.in_rate)) as u64
    }

    /// Resample `frame` (interleaved, at the input rate), appending to `out`.
    pub fn process(&mut self, frame: &AudioFrame, out: &mut Vec<f32>) -> Result<(), AudioError> {
        if frame.channels != self.channels {
            return Err(AudioError::Resample(format!(
                "channel mismatch: resampler={}, frame={}",
                self.channels, frame.channels
            )));
        }
        if frame.sample_rate != self.in_rate {
            return Err(AudioError::Resample(format!(
                "sample rate mismatch: resampler in_rate={}, frame={}",
                self.in_rate, frame.sample_rate
            )));
        }
        let ch = usize::from(self.channels);
        self.samples_in += (frame.samples.len() / ch) as u64;
        if self.kernel.is_none() {
            out.extend_from_slice(&frame.samples);
            self.samples_out = self.samples_in;
            return Ok(());
        }
        for (c, h) in self.hist.iter_mut().enumerate() {
            h.extend(frame.samples.iter().skip(c).step_by(ch));
        }
        self.run(u64::MAX, out);
        Ok(())
    }

    /// The end of the input: the outputs whose taps read past it (as
    /// silence), up to [`Self::target_len`].
    pub fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), AudioError> {
        let Some(k) = self.kernel.as_ref() else {
            return Ok(());
        };
        let pad = 2 * k.half + k.step_num.div_ceil(k.step_den) as usize + 2;
        for h in &mut self.hist {
            h.resize(h.len() + pad, 0.0);
        }
        let target = self.target_len();
        self.run(target, out);
        Ok(())
    }

    /// Every output before `limit` whose taps the history holds.
    fn run(&mut self, limit: u64, out: &mut Vec<f32>) {
        let Some(kernel) = self.kernel.as_ref() else {
            return;
        };
        let end = self.base + self.hist[0].len() as i64;
        let mut k = self.samples_out;
        while k < limit {
            let (first, taps) = kernel.taps(k, &mut self.scratch);
            if first + taps.len() as i64 > end {
                break;
            }
            let at = (first - self.base) as usize;
            for h in &self.hist {
                let x = &h[at..at + taps.len()];
                out.push(x.iter().zip(taps).map(|(&x, &t)| x * t).sum());
            }
            k += 1;
        }
        self.samples_out = k;
        // Drop the history no later output reads.
        let (first, _) = kernel.taps(k, &mut self.scratch);
        let drop = (first - self.base).clamp(0, self.hist[0].len() as i64) as usize;
        if drop > 0 {
            for h in &mut self.hist {
                h.drain(..drop);
            }
            self.base += drop as i64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resample(from: u32, to: u32, ch: u8, input: &[f32], chunk: usize) -> Vec<f32> {
        let mut r = AlignedResampler::new(from, to, ch).unwrap();
        let mut out = Vec::new();
        for c in input.chunks(chunk * usize::from(ch)) {
            let frame = AudioFrame { samples: c.to_vec(), sample_rate: from, channels: ch, pts: 0 };
            r.process(&frame, &mut out).unwrap();
        }
        r.flush(&mut out).unwrap();
        out
    }

    /// The rate pairs an encoder meets: into 48 kHz (Opus, AC-3, DTS, and
    /// AAC/MP3 from the high rates), into 44.1 and 32, and up from the low
    /// ones.
    const PAIRS: [(u32, u32); 16] = [
        (44_100, 48_000),
        (48_000, 44_100),
        (96_000, 48_000),
        (88_200, 44_100),
        (88_200, 48_000),
        (192_000, 48_000),
        (176_400, 44_100),
        (32_000, 48_000),
        (22_050, 48_000),
        (24_000, 48_000),
        (16_000, 48_000),
        (8_000, 48_000),
        (11_025, 44_100),
        (22_050, 44_100),
        (96_000, 44_100),
        (48_000, 32_000),
    ];

    #[test]
    fn rejects_bad_arguments() {
        assert!(AlignedResampler::new(0, 48000, 1).is_err());
        assert!(AlignedResampler::new(44100, 0, 1).is_err());
        assert!(AlignedResampler::new(44100, 48000, 0).is_err());
        assert!(AlignedResampler::new(44100, 48000, 9).is_err());
        let mut r = AlignedResampler::new(44100, 48000, 2).unwrap();
        let mut out = Vec::new();
        let mono = AudioFrame { samples: vec![0.0; 64], sample_rate: 44100, channels: 1, pts: 0 };
        assert!(r.process(&mono, &mut out).is_err());
        let wrong_rate = AudioFrame { samples: vec![0.0; 64], sample_rate: 22050, channels: 2, pts: 0 };
        assert!(r.process(&wrong_rate, &mut out).is_err());
    }

    #[test]
    fn equal_rates_pass_through() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.37).sin()).collect();
        assert_eq!(resample(48_000, 48_000, 2, &input, 333), input);
    }

    /// Tones across the pass band, at every pair: as long as the input at the
    /// new rate, in time with it at zero lag (matched against the same tone
    /// generated at the new rate — any delay, even a fraction of a sample,
    /// shows as error), and at unity gain to ±0.01 dB.
    #[test]
    fn tones_come_out_in_time_at_unity_gain() {
        for (from, to) in PAIRS {
            let nyquist = f64::from(from.min(to)) / 2.0;
            let n = from as usize / 2; // half a second
            for f in [100.0, 1000.0, 0.5 * nyquist, 0.85 * nyquist] {
                let tone =
                    |rate: u32, i: f64| 0.5 * (2.0 * std::f64::consts::PI * f * i / f64::from(rate) + 0.3).sin();
                let input: Vec<f32> = (0..n).map(|i| tone(from, i as f64) as f32).collect();
                let out = resample(from, to, 1, &input, 777);
                assert_eq!(out.len() as u64, (n as u64 * u64::from(to)).div_ceil(u64::from(from)), "{from} -> {to}");
                // Away from the ends (where the input starts and stops).
                let skip = to as usize / 50;
                let (mut e, mut dot, mut pow) = (0f64, 0f64, 0f64);
                for (i, &v) in out.iter().enumerate().skip(skip).take(out.len() - 2 * skip) {
                    let want = tone(to, i as f64);
                    e += (want - f64::from(v)).powi(2);
                    dot += want * f64::from(v);
                    pow += want * want;
                }
                let snr = 10.0 * (pow / e.max(1e-30)).log10();
                let gain_db = 20.0 * (dot / pow).log10();
                assert!(snr > 80.0, "{from} -> {to}, {f:.0} Hz: {snr:.1} dB against the tone at zero lag");
                assert!(gain_db.abs() < 0.01, "{from} -> {to}, {f:.0} Hz: gain {gain_db:+.4} dB");
            }
        }
    }

    /// An impulse at input sample `n` comes out centred on output time
    /// `n · out / in` (a whole output sample or not), its peak on the
    /// nearest output sample.
    #[test]
    fn impulses_come_out_where_they_went_in() {
        for (from, to) in PAIRS {
            for n in [0usize, 1, 3, 1000, 1001, 1003] {
                let mut input = vec![0.0f32; 4096];
                input[2000 + n] = 1.0;
                let out = resample(from, to, 1, &input, 500);
                let want = (2000 + n) as f64 * f64::from(to) / f64::from(from);
                // The centre of the response's energy.
                let (m, w) = out.iter().enumerate().fold((0f64, 0f64), |(m, w), (i, &v)| {
                    let e = f64::from(v) * f64::from(v);
                    (m + i as f64 * e, w + e)
                });
                let centre = m / w;
                assert!((centre - want).abs() < 1e-2, "{from} -> {to}, impulse at {n}: centre {centre:.4}, want {want:.4}");
                let peak = out
                    .iter()
                    .enumerate()
                    .fold((0, 0f32), |b, (i, &v)| if v.abs() > b.1 { (i, v.abs()) } else { b })
                    .0;
                assert!((peak as f64 - want).abs() <= 0.5 + 1e-9, "{from} -> {to}: peak at {peak}, want {want}");
            }
        }
    }

    /// Above the new Nyquist frequency a tone is rejected (it would alias).
    #[test]
    fn the_stop_band_is_rejected() {
        for (from, to) in [(96_000u32, 48_000u32), (48_000, 44_100), (192_000, 48_000)] {
            let f = f64::from(to) / 2.0 * 1.08;
            let input: Vec<f32> = (0..from as usize / 4)
                .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / f64::from(from)).sin() as f32)
                .collect();
            let out = resample(from, to, 1, &input, 1000);
            let skip = out.len() / 8;
            let body = &out[skip..out.len() - skip];
            let rms = (body.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / body.len() as f64).sqrt();
            let db = 20.0 * (rms * 2f64.sqrt()).log10();
            assert!(db < -100.0, "{from} -> {to}: {f:.0} Hz at {db:.1} dB");
        }
    }

    /// Channels are resampled independently, interleaving kept; the output
    /// does not depend on how the input is chunked.
    #[test]
    fn channels_and_chunking() {
        let n = 10_000;
        let input: Vec<f32> = (0..n).flat_map(|i| [(i as f32 * 0.01).sin(), -0.25]).collect();
        let a = resample(44_100, 48_000, 2, &input, 1);
        let b = resample(44_100, 48_000, 2, &input, 4096);
        assert_eq!(a.len(), 2 * (n * 48_000usize).div_ceil(44_100));
        assert_eq!(a, b);
        for v in a.chunks_exact(2).skip(500).take(9000) {
            assert!((v[1] + 0.25).abs() < 1e-4, "{}", v[1]);
        }
    }

    /// An odd ratio (no small reduced form) uses interpolated phases and
    /// still lines up.
    #[test]
    fn an_odd_ratio_interpolates_its_phases() {
        let (from, to) = (44_101u32, 48_000u32);
        assert!(u64::from(to) / gcd(u64::from(from), u64::from(to)) > MAX_EXACT_PHASES);
        let tone = |rate: u32, i: f64| 0.5 * (2.0 * std::f64::consts::PI * 997.0 * i / f64::from(rate)).sin();
        let input: Vec<f32> = (0..from as usize / 2).map(|i| tone(from, i as f64) as f32).collect();
        let out = resample(from, to, 1, &input, 1000);
        let (mut s, mut e) = (0f64, 0f64);
        for (i, &v) in out.iter().enumerate().skip(1000).take(out.len() - 2000) {
            let want = tone(to, i as f64);
            s += want * want;
            e += (want - f64::from(v)).powi(2);
        }
        assert!(10.0 * (s / e).log10() > 80.0);
    }
}
