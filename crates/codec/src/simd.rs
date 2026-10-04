//! Runtime SIMD dispatch for the shared pipeline's per-pixel and
//! per-sample kernels (the scaler, the SDR-to-HDR mapping, the temporal
//! denoiser, the audio resampler, ...), which do not have a ladder of their
//! own.
//!
//! The level is what the CPU advertises, capped by `RIVET_PIPE_MAX_SIMD`
//! (`avx512`, `avx2`, `none`), read once per process — so every path can be
//! exercised on one machine, and a same-binary timing control is one
//! environment variable away. Every kernel behind it keeps its scalar
//! reference, and the vector paths are bit-identical to it: integer
//! kernels trivially, float ones by doing the reference's IEEE-754
//! operations in the reference's order, with no fused multiply-add — so
//! output does not depend on which machine, or which level, ran it.

use std::sync::OnceLock;

/// The vector level the pipeline kernels run at. Ordered: a cap is a `min`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// The scalar references (and, on AArch64, where a kernel has one, its
    /// NEON form — NEON is baseline there and is not capped separately).
    Scalar,
    /// x86-64 AVX2 (256-bit; FMA is not used).
    Avx2,
    /// x86-64 AVX-512 F + BW + VL, for the kernels that have a 512-bit
    /// form; the rest run their AVX2 one.
    Avx512,
}

impl Level {
    /// What the host can run, ignoring the environment.
    pub fn host() -> Level {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                let avx512 = std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("avx512vl");
                return if avx512 { Level::Avx512 } else { Level::Avx2 };
            }
        }
        Level::Scalar
    }

    /// `host` capped by a `RIVET_PIPE_MAX_SIMD` value. An unrecognised
    /// value is ignored rather than silently downgrading.
    pub fn cap(host: Level, env: Option<&str>) -> Level {
        match env.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            Some("none") | Some("scalar") | Some("0") => Level::Scalar,
            Some("avx2") => host.min(Level::Avx2),
            _ => host,
        }
    }

    /// The level in force in this process (resolved once).
    pub fn get() -> Level {
        static LEVEL: OnceLock<Level> = OnceLock::new();
        *LEVEL.get_or_init(|| {
            let level = Level::cap(
                Level::host(),
                std::env::var("RIVET_PIPE_MAX_SIMD").ok().as_deref(),
            );
            tracing::debug!(host = ?Level::host(), ?level, "pipeline SIMD level");
            level
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cap_only_lowers() {
        assert_eq!(Level::cap(Level::Avx512, Some("avx2")), Level::Avx2);
        assert_eq!(Level::cap(Level::Avx2, Some("avx512")), Level::Avx2);
        assert_eq!(Level::cap(Level::Avx2, Some("none")), Level::Scalar);
        assert_eq!(Level::cap(Level::Scalar, Some("avx2")), Level::Scalar);
        assert_eq!(Level::cap(Level::Avx2, Some("bogus")), Level::Avx2);
        assert_eq!(Level::cap(Level::Avx512, None), Level::Avx512);
    }
}
