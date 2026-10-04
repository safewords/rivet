//! VP8 encode — this workspace's own encoder (`crates/vp8`, the rivet-vp8
//! repository), written clean-room from RFC 6386.
//!
//! The only VP8 encoder in the tree (no hardware backend here encodes VP8),
//! so [`select_encoder`](super::select_encoder) builds it directly for a VP8
//! job; see [`native`](super::native) for why that needs no feature.
//!
//! # What it takes and writes
//!
//! 8-bit 4:2:0 (`yuv420p`), one VP8 frame per picture, every one shown, in
//! display order — no reordering, so each packet carries its frame's
//! timestamp unchanged. Key frames at the configured interval and wherever
//! [`force_keyframe_next`](Encoder::force_keyframe_next) asks. The encoder
//! decodes every frame it writes with the crate's decoder and predicts from
//! that, so what it predicts from is what a decoder has.
//!
//! # Rate and quality
//!
//! A fixed quantiser (`q_index` 0-127): the rung's CRF on libvpx's 0-63
//! scale doubled, else the quality target's
//! ([`tuning::native_sw_quantizer`](super::tuning::native_sw_quantizer)).
//! No rate control — a bitrate rung is refused by name. The speed tier sets
//! the motion search range (8 / 16 / 32 samples).
//!
//! # Threads
//!
//! Each frame encodes on [`EncoderConfig::threads`] threads (0: one per
//! CPU); the stream does not depend on the count. Frames 720 lines or taller
//! are written with four token partitions, so a decoder (this crate's
//! included) can decode their macroblock rows in parallel.
//!
//! # Colour
//!
//! VP8 has one colour space (BT.601, RFC 6386 §9.2) and signals nothing
//! else, so it is 8-bit SDR only; the container's `colr` / `Colour` carry
//! the tags.

use std::collections::VecDeque;

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use super::native::{check_frame, quantizer, refuse_any_rate, threads, tier};
use super::tuning::SpeedTier;
use super::{EncodedPacket, Encoder, EncoderConfig};
use crate::frame::{PixelFormat, VideoCodec, VideoFrame};

/// A VP8 encoder behind rivet's [`Encoder`] trait.
pub struct Vp8Encoder {
    inner: vp8::Encoder,
    cfg: vp8::Config,
    ready: VecDeque<EncodedPacket>,
}

impl Vp8Encoder {
    /// Build an encoder for `config` (codec VP8, `yuv420p`).
    pub fn new(config: EncoderConfig) -> Result<Self> {
        if config.codec != VideoCodec::Vp8 {
            bail!("the VP8 encoder encodes VP8, not {}", config.codec.label());
        }
        if config.pixel_format != PixelFormat::Yuv420p {
            bail!(
                "the VP8 encoder writes 8-bit 4:2:0 only (VP8 has no other format); this rung asks for {:?}",
                config.pixel_format
            );
        }
        refuse_any_rate("VP8", &config)?;
        let cfg = vp8::Config {
            width: config.width,
            height: config.height,
            quantizer: quantizer(&config),
            keyframe_interval: config.keyframe_interval,
            search_range: match tier(&config) {
                SpeedTier::Draft => 8,
                SpeedTier::Standard => 16,
                SpeedTier::Archive => 32,
            },
            token_partitions: if config.height >= 720 { 4 } else { 1 },
            threads: threads(&config),
            ..vp8::Config::default()
        };
        let inner =
            vp8::Encoder::new(cfg.clone()).context("the VP8 encoder rejected the configuration")?;
        Ok(Self {
            inner,
            cfg,
            ready: VecDeque::new(),
        })
    }
}

impl Encoder for Vp8Encoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        let want = check_frame(
            "VP8",
            frame,
            self.cfg.width,
            self.cfg.height,
            &[PixelFormat::Yuv420p],
        )?;
        let picture =
            vp8::Frame::from_packed(self.cfg.width, self.cfg.height, frame.data[..want].to_vec())
                .context("the VP8 encoder refused the frame's layout")?;
        let data = self
            .inner
            .encode(&picture)
            .context("the VP8 encoder refused a frame")?;
        let is_keyframe = data.first().is_some_and(|tag| tag & 1 == 0);
        self.ready.push_back(EncodedPacket {
            data: Bytes::from(data),
            pts: frame.pts,
            is_keyframe,
        });
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        Ok(self.ready.pop_front())
    }

    fn force_keyframe_next(&mut self) -> Result<()> {
        self.inner.force_key_frame();
        Ok(())
    }

    /// Rebuild the encoder: no references, the next frame a key frame.
    fn reset(&mut self) -> Result<()> {
        self.inner = vp8::Encoder::new(self.cfg.clone()).context("rebuilding the VP8 encoder")?;
        self.ready.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decoder reproduces the encoder's reconstruction byte for byte, and
    /// every packet carries its frame's timestamp, key frames where asked.
    #[test]
    fn packets_decode_to_the_reconstruction() {
        let (w, h) = (64, 48);
        let config = EncoderConfig {
            width: w,
            height: h,
            codec: VideoCodec::Vp8,
            keyframe_interval: 3,
            ..Default::default()
        };
        let mut enc = Vp8Encoder::new(config).unwrap();
        let mut dec = vp8::Decoder::new();
        for n in 0..5 {
            enc.send_frame(&super::super::native::test_picture(w, h, n))
                .unwrap();
            let p = enc.receive_packet().unwrap().expect("a packet per frame");
            assert_eq!(p.pts, n);
            assert_eq!(p.is_keyframe, n % 3 == 0, "frame {n}");
            let shown = dec.decode(&p.data).unwrap().expect("a shown frame");
            assert_eq!(shown.packed(), enc.inner.reconstruction().unwrap().packed());
        }
        enc.force_keyframe_next().unwrap();
        enc.send_frame(&super::super::native::test_picture(w, h, 5))
            .unwrap();
        assert!(enc.receive_packet().unwrap().unwrap().is_keyframe);
    }

    /// The rung's thread budget reaches the encoder; zero is the machine's.
    #[test]
    fn the_rung_thread_budget_reaches_the_encoder() {
        let base = EncoderConfig {
            width: 64,
            height: 48,
            frame_rate: 25.0,
            codec: VideoCodec::Vp8,
            ..Default::default()
        };
        let all = std::thread::available_parallelism().map_or(1, |n| n.get());
        for (asked, want) in [(3, 3), (0, all)] {
            let enc = Vp8Encoder::new(EncoderConfig {
                threads: asked,
                ..base.clone()
            })
            .unwrap();
            assert_eq!(enc.cfg.threads, want, "threads {asked}");
        }
    }

    #[test]
    fn ten_bit_and_rates_are_refused() {
        let base = EncoderConfig {
            width: 64,
            height: 48,
            codec: VideoCodec::Vp8,
            ..Default::default()
        };
        let ten = EncoderConfig {
            pixel_format: PixelFormat::Yuv420p10le,
            ..base.clone()
        };
        assert!(Vp8Encoder::new(ten).is_err());
        let mut rate = base;
        rate.overrides.bitrate = Some(1_000_000);
        assert!(
            Vp8Encoder::new(rate)
                .err()
                .expect("refused")
                .to_string()
                .contains("fixed quantiser")
        );
    }
}
