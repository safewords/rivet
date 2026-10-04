//! Blu-ray LPCM (BDAV / HDMV stream_type 0x80) read into the little-endian
//! interleaved PCM rivet's PCM decoder takes.
//!
//! Each PES packet of the stream (private_stream_1, `0xBD`) carries one
//! audio frame: a 4-byte header, then the samples, big-endian, interleaved.
//! The header (Blu-ray Disc Read-Only Format, Part 3, the LPCM audio
//! header), most significant bit first:
//!
//! | bits | field                       |
//! |------|-----------------------------|
//! | 16   | `audio_data_payload_size`: the bytes of samples that follow |
//! | 4    | `channel_assignment`        |
//! | 4    | `sampling_frequency`: 1 = 48 kHz, 4 = 96 kHz, 5 = 192 kHz |
//! | 2    | `bits_per_sample`: 1 = 16, 2 = 20, 3 = 24 |
//! | 1    | `start_flag`                |
//! | 5    | reserved                    |
//!
//! The channels are stored in an even count — a layout with an odd number
//! of channels carries one more, empty, which is dropped here — and in the
//! format's own order, which for the layouts with surrounds is not WAVE's:
//! 3/2+LFE is stored L R C Ls Rs LFE, and 3/4+LFE L R C Ls Lrs Rrs Rs LFE.
//! They are put here in the WAVE order for their count that every PCM track
//! in rivet is read in ([`crate::raw_audio`]): L R C LFE Ls Rs, and L R C
//! LFE Lrs Rrs Ls Rs. The layouts whose channel count WAVE order would read
//! as another layout (2/1, 3/1, and 3/4 without LFE) are refused by name
//! rather than played from the wrong speakers. 20-bit samples are stored in
//! 24 bits and read as 24-bit.

use anyhow::{Result, bail};

use crate::demux::AudioTrack;

/// One frame's header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BdLpcmHeader {
    /// Bytes of samples after the header.
    pub(crate) payload: usize,
    pub(crate) channel_assignment: u8,
    pub(crate) sample_rate: u32,
    /// Bytes per stored sample: 2 (16-bit) or 3 (20- and 24-bit).
    pub(crate) sample_bytes: usize,
}

impl BdLpcmHeader {
    pub(crate) fn parse(h: &[u8]) -> Option<Self> {
        let h: [u8; 4] = h.get(..4)?.try_into().ok()?;
        let sample_rate = match h[2] & 0x0F {
            1 => 48_000,
            4 => 96_000,
            5 => 192_000,
            _ => return None,
        };
        let sample_bytes = match h[3] >> 6 {
            1 => 2,
            2 | 3 => 3,
            _ => return None,
        };
        Some(Self {
            payload: usize::from(u16::from_be_bytes([h[0], h[1]])),
            channel_assignment: h[2] >> 4,
            sample_rate,
            sample_bytes,
        })
    }

    /// The layout's stored channels (even), and for each output channel in
    /// WAVE order the stored channel it is. `Err` names a layout refused.
    pub(crate) fn channel_map(self) -> Result<(usize, &'static [usize])> {
        Ok(match self.channel_assignment {
            1 => (2, &[0]),                       // mono (+1 empty)
            3 => (2, &[0, 1]),                    // L R
            4 => (4, &[0, 1, 2]),                 // L R C (+1 empty)
            7 => (4, &[0, 1, 2, 3]),              // L R Ls Rs
            8 => (6, &[0, 1, 2, 3, 4]),           // L R C Ls Rs (+1 empty)
            9 => (6, &[0, 1, 2, 5, 3, 4]),        // L R C Ls Rs LFE
            11 => (8, &[0, 1, 2, 7, 4, 5, 3, 6]), // L R C Ls Lrs Rrs Rs LFE
            5 => bail!("Blu-ray LPCM 2/1 (L R S) has no WAVE-order layout in rivet"),
            6 => bail!("Blu-ray LPCM 3/1 (L R C S) has no WAVE-order layout in rivet"),
            10 => bail!("Blu-ray LPCM 3/4 without LFE has no WAVE-order layout in rivet"),
            other => bail!("Blu-ray LPCM channel_assignment {other} is reserved"),
        })
    }
}

/// A Blu-ray LPCM track from its PES packets: `es` is the packets' payloads
/// end to end and `pes_starts` where each begins in it (each is one frame).
/// One packet per frame, in `pcm_s16le` (16-bit) or `pcm_s24le` (20- and
/// 24-bit); beside it, where each kept frame starts in `es`. `None` when no
/// packet holds a frame. The first frame sets the format; a frame that
/// changes it ends the track (a stream that changes format part-way is two
/// streams).
pub(crate) fn bd_lpcm_from_pes(
    es: &[u8],
    pes_starts: &[usize],
) -> Result<Option<(AudioTrack, Vec<usize>)>> {
    let mut first: Option<(BdLpcmHeader, usize, &'static [usize])> = None;
    let mut samples = Vec::new();
    let mut durations = Vec::new();
    let mut starts = Vec::new();
    for (i, &start) in pes_starts.iter().enumerate() {
        let end = pes_starts
            .get(i + 1)
            .copied()
            .unwrap_or(es.len())
            .min(es.len());
        let Some(frame) = es.get(start..end) else {
            continue;
        };
        let Some(header) = BdLpcmHeader::parse(frame) else {
            continue;
        };
        let (stored, map) = match first {
            Some((f, stored, map)) => {
                if (f.channel_assignment, f.sample_rate, f.sample_bytes)
                    != (
                        header.channel_assignment,
                        header.sample_rate,
                        header.sample_bytes,
                    )
                {
                    break;
                }
                (stored, map)
            }
            None => {
                let (stored, map) = header.channel_map()?;
                first = Some((header, stored, map));
                (stored, map)
            }
        };
        let body = &frame[4..frame.len().min(4 + header.payload)];
        let width = header.sample_bytes;
        let frames = body.len() / (stored * width);
        if frames == 0 {
            continue;
        }
        let mut out = Vec::with_capacity(frames * map.len() * width);
        for f in body.chunks_exact(stored * width).take(frames) {
            for &c in map {
                // Big-endian to little-endian.
                out.extend(f[c * width..(c + 1) * width].iter().rev());
            }
        }
        samples.push(out);
        durations.push(frames as u32);
        starts.push(start);
    }
    let Some((header, _, map)) = first else {
        return Ok(None);
    };
    if samples.is_empty() {
        return Ok(None);
    }
    Ok(Some((
        AudioTrack {
            codec: if header.sample_bytes == 2 {
                "pcm_s16le"
            } else {
                "pcm_s24le"
            }
            .into(),
            samples,
            sample_rate: header.sample_rate,
            channels: map.len() as u16,
            asc: Vec::new(),
            codec_private: Vec::new(),
            timescale: header.sample_rate,
            durations,
        },
        starts,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One frame: the header for `assignment` / 48 kHz / 16-bit, then
    /// `frames` frames of `stored` channels, channel `c` holding `c + 1`.
    fn frame16(assignment: u8, stored: usize, frames: usize) -> Vec<u8> {
        let payload = frames * stored * 2;
        let mut f = vec![
            (payload >> 8) as u8,
            payload as u8,
            (assignment << 4) | 1,
            1 << 6,
        ];
        for _ in 0..frames {
            for c in 0..stored {
                f.extend_from_slice(&(c as i16 + 1).to_be_bytes());
            }
        }
        f
    }

    #[test]
    fn the_header_reads_as_the_format_lays_it_out() {
        // 960 bytes, 3/2+LFE, 96 kHz, 24-bit, start_flag.
        let h = BdLpcmHeader::parse(&[0x03, 0xC0, 0x94, 0xE0]).unwrap();
        assert_eq!(
            h,
            BdLpcmHeader {
                payload: 960,
                channel_assignment: 9,
                sample_rate: 96_000,
                sample_bytes: 3
            }
        );
        assert!(
            BdLpcmHeader::parse(&[0, 0, 0x32, 0x40]).is_none(),
            "sampling_frequency 2 is reserved"
        );
        assert!(
            BdLpcmHeader::parse(&[0, 0, 0x31, 0x00]).is_none(),
            "bits_per_sample 0 is reserved"
        );
    }

    #[test]
    fn surround_channels_come_out_in_wave_order() {
        // 3/2+LFE stored L R C Ls Rs LFE (1..6) → L R C LFE Ls Rs.
        let f = frame16(9, 6, 2);
        let (track, starts) = bd_lpcm_from_pes(&f, &[0]).unwrap().unwrap();
        assert_eq!(
            (track.codec.as_str(), track.channels, track.sample_rate),
            ("pcm_s16le", 6, 48_000)
        );
        let values: Vec<i16> = track.samples[0]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(values, [1, 2, 3, 6, 4, 5, 1, 2, 3, 6, 4, 5]);
        assert_eq!(
            (track.durations.as_slice(), starts.as_slice()),
            (&[2u32][..], &[0usize][..])
        );

        // 3/4+LFE stored L R C Ls Lrs Rrs Rs LFE → L R C LFE Lrs Rrs Ls Rs.
        let f = frame16(11, 8, 1);
        let (track, _) = bd_lpcm_from_pes(&f, &[0]).unwrap().unwrap();
        let values: Vec<i16> = track.samples[0]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(values, [1, 2, 3, 8, 5, 6, 4, 7]);
    }

    #[test]
    fn the_empty_channel_of_an_odd_layout_is_dropped() {
        let mut es = frame16(1, 2, 3);
        let second = es.len();
        es.extend(frame16(1, 2, 2));
        let (track, starts) = bd_lpcm_from_pes(&es, &[0, second]).unwrap().unwrap();
        assert_eq!(track.channels, 1);
        assert_eq!(track.samples[0], [1, 0, 1, 0, 1, 0]);
        assert_eq!(track.durations, [3, 2]);
        assert_eq!(starts, [0, second]);
    }

    #[test]
    fn twenty_four_bit_samples_are_byte_swapped_whole() {
        let mut f = vec![0x00, 0x06, 0x31, 0xC0];
        f.extend_from_slice(&[0x12, 0x34, 0x56, 0xFE, 0xDC, 0xBA]);
        let (track, _) = bd_lpcm_from_pes(&f, &[0]).unwrap().unwrap();
        assert_eq!(track.codec, "pcm_s24le");
        assert_eq!(track.samples[0], [0x56, 0x34, 0x12, 0xBA, 0xDC, 0xFE]);
    }

    #[test]
    fn layouts_wave_order_cannot_say_are_refused_by_name() {
        for assignment in [5u8, 6, 10] {
            let f = frame16(assignment, 4, 1);
            let e = bd_lpcm_from_pes(&f, &[0]).unwrap_err().to_string();
            assert!(e.contains("Blu-ray LPCM"), "{e}");
        }
    }
}
