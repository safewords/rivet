//! JPEG XL input, through rivet-jpegxl (`jpegxl`), the typed wrapper over
//! jxl-rs — the JPEG XL project's own pure-Rust decoder. Decode only: rivet
//! writes no JPEG XL.
//!
//! The picture comes out upright (the codestream's orientation applied by
//! the decoder), as 8-bit RGBA, tagged with the ICC profile of the colour
//! space its pixels are in; the colour step makes that sRGB as it does any
//! tagged source. A 16-bit or float (HDR) file is brought to 8 bits by the
//! decoder; HDR stills are not tone-mapped (decisions §28).

use anyhow::{Context, Result, anyhow};

use super::colour::Profile;
use super::decode::{MAX_SOURCE_PIXELS, Picture};
use super::raster::RgbaImage;

fn options() -> jpegxl::DecodeOptions {
    jpegxl::DecodeOptions {
        sample_type: jpegxl::SampleType::U8,
        gray_to_rgb: true,
        threads: crate::thread_budget::per_job(),
        limits: jpegxl::Limits {
            max_pixels: MAX_SOURCE_PIXELS,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// The picture's size as stored, its orientation, and its samples as
/// probe names them (`Rgb8`, `Rgba16`, `L8`, `Rgb32F`, …).
pub(crate) fn read_header(data: &[u8]) -> Result<(u32, u32, u16, String)> {
    let info = jpegxl::probe(data).map_err(|e| anyhow!("{e}"))?;
    let (w, h) = info.stored_dims();
    let channels = match (info.gray, info.has_alpha) {
        (true, false) => "L",
        (true, true) => "La",
        (false, false) => "Rgb",
        (false, true) => "Rgba",
    };
    let depth = if info.float {
        "32F".to_string()
    } else if info.bits_per_sample > 8 {
        "16".to_string()
    } else {
        "8".to_string()
    };
    Ok((
        w,
        h,
        u16::from(info.orientation),
        format!("{channels}{depth}"),
    ))
}

/// Decode the still (an animation's first frame), upright.
pub(crate) fn decode(data: &[u8]) -> Result<Picture> {
    let image = jpegxl::decode_with(data, &options()).map_err(|e| anyhow!("{e}"))?;
    let jpegxl::Pixels::U8(samples) = image.pixels else {
        return Err(anyhow!(
            "the JPEG XL decoder returned other than 8-bit samples"
        ));
    };
    let rgba = match image.channels {
        jpegxl::Channels::Rgba => RgbaImage::from_raw(image.width, image.height, samples),
        jpegxl::Channels::Rgb => RgbaImage::from_rgb(image.width, image.height, &samples),
        other => return Err(anyhow!("the JPEG XL decoder returned {other:?} pixels")),
    }
    .context("the JPEG XL decoder returned the wrong number of pixels")?;
    let icc = image.icc_profile.filter(|p| !p.is_empty());
    Ok(Picture::new(rgba, icc.map(Profile::Icc)))
}
