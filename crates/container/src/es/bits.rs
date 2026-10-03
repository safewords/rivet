//! An MSB-first bit reader with Exp-Golomb codes (H.264 §9.1) and AV1's
//! `leb128()` / `uvlc()` (AV1 §4.10), for the few header fields the
//! elementary-stream readers need.

pub(super) struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Bits<'a> {
    pub(super) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(super) fn bit(&mut self) -> Option<u32> {
        let byte = *self.data.get(self.pos / 8)?;
        let b = (byte >> (7 - self.pos % 8)) & 1;
        self.pos += 1;
        Some(u32::from(b))
    }

    pub(super) fn bits(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Some(v)
    }

    /// `ue(v)`.
    pub(super) fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        Some(((1u64 << zeros) - 1 + u64::from(self.bits(zeros)?)) as u32)
    }

    /// AV1 `uvlc()` (§4.10.3).
    pub(super) fn uvlc(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros >= 32 {
                return Some(u32::MAX);
            }
        }
        Some(((1u64 << zeros) - 1 + u64::from(self.bits(zeros)?)) as u32)
    }
}

/// AV1 `leb128()` (§4.10.5) at the head of `data`: the value and its length
/// in bytes. At most eight bytes, and the value must fit 32 bits.
pub(super) fn leb128(data: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    for i in 0..8 {
        let b = *data.get(i)?;
        value |= u64::from(b & 0x7f) << (7 * i);
        if b & 0x80 == 0 {
            return (value <= u64::from(u32::MAX)).then_some((value, i + 1));
        }
    }
    None
}

/// `value` as `leb128()`, in as few bytes as it takes.
pub(super) fn write_leb128(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let b = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}
