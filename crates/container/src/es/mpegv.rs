//! MPEG-1 / MPEG-2 video elementary streams (`.m2v`, `.mpv`, `.m1v`): the
//! start-code units of ISO/IEC 11172-2 / 13818-2 with nothing around them,
//! read exactly as the program-stream reader reads its video once the packs
//! are gone — one sample per coded frame (a field pair joined), the frame
//! rate the sequence header's `frame_rate_code` names, MPEG-2 told from
//! MPEG-1 by its sequence extension.

use anyhow::{Result, bail};

use super::{EsSample, Indexed};
use crate::mpeg_es::{MPEG2_GOP, MPEG2_PICTURE, MPEG2_SEQUENCE_HEADER, start_codes};

/// A sequence header first whose sizes, aspect code and frame-rate code are
/// legal (ISO/IEC 13818-2 §6.3.3), then a start code of what may follow it
/// (an extension, user data, a GOP or a picture).
pub(super) fn sniff(data: &[u8]) -> bool {
    if data.len() < 16 || data[..4] != [0, 0, 1, MPEG2_SEQUENCE_HEADER] {
        return false;
    }
    let width = (u32::from(data[4]) << 4) | u32::from(data[5] >> 4);
    let height = (u32::from(data[5] & 0x0f) << 8) | u32::from(data[6]);
    let aspect = data[7] >> 4;
    let rate = data[7] & 0x0f;
    // marker_bit after the 18-bit bit_rate_value.
    let marker = (data[10] >> 5) & 1;
    if width == 0 || height == 0 || !(1..=14).contains(&aspect) || !(1..=8).contains(&rate) || marker != 1 {
        return false;
    }
    start_codes(&data[4..data.len().min(4096)])
        .first()
        .is_some_and(|&(_, c)| matches!(c, 0xb2 | 0xb5 | MPEG2_GOP | MPEG2_PICTURE))
}

pub(super) fn index(data: &[u8]) -> Result<Indexed> {
    let codes = start_codes(data);
    let mpeg2 = codes
        .iter()
        .take_while(|(_, c)| !matches!(*c, MPEG2_PICTURE | MPEG2_GOP))
        .any(|&(o, c)| c == 0xb5 && data.get(o + 4).is_some_and(|b| b >> 4 == 0x1));
    let frames = crate::ps::coded_frames(data);
    if frames.is_empty() {
        bail!("m2v: the stream has no picture");
    }
    Ok(Indexed {
        codec: if mpeg2 { "mpeg2" } else { "mpeg1" },
        frames: frames.len() as u64,
        samples: frames.into_iter().map(EsSample::Owned).collect(),
        pts: None,
        frame_rate: crate::ps::sequence_frame_rate(data),
        dims: None,
        label: "m2v",
    })
}
