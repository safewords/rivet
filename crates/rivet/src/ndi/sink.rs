//! An NDI output: one announced source, sent each rung's pictures and the
//! job's sound.

use anyhow::{Result, bail};
use codec::audio::AudioFrame;
use codec::frame::{PixelFormat, VideoFrame};

use crate::live::NdiEndpoint;

/// One announced NDI source.
pub struct NdiSink {
    sender: ndi::Sender,
}

impl NdiSink {
    /// Announce `stream_name` (in the endpoint's groups). With `clock_video`
    /// the runtime paces the pictures at their frame rate — what a file
    /// played out needs; a live source relayed is paced by its own arrival.
    pub fn new(endpoint: &NdiEndpoint, stream_name: &str, clock_video: bool) -> Result<Self> {
        let runtime = ndi::Ndi::load()?;
        let mut options = ndi::SenderOptions::new(stream_name);
        options.groups = endpoint.groups.clone();
        options.clock_video = clock_video;
        let sender = runtime.sender(&options)?;
        tracing::info!(name = stream_name, runtime = %runtime.version(), "NDI source announced");
        Ok(Self { sender })
    }

    /// The stream name receivers see after the machine's.
    pub fn name(&self) -> &str {
        self.sender.name()
    }

    /// One picture, 8-bit as I420 or 10-bit as P216.
    pub fn send_frame(&mut self, frame: &VideoFrame, frame_rate: (u32, u32)) -> Result<()> {
        let layout = match frame.format {
            PixelFormat::Yuv420p => ndi::Layout::Yuv420p,
            PixelFormat::Yuv420p10le => ndi::Layout::Yuv420p10le,
            other => bail!("an NDI output takes 4:2:0 pictures, got {other:?}"),
        };
        let picture = ndi::Picture {
            layout,
            width: frame.width,
            height: frame.height,
            data: frame.data.to_vec(),
        };
        self.sender.send_picture(&picture, frame_rate, None)?;
        Ok(())
    }

    /// Interleaved float audio.
    pub fn send_audio(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.samples.is_empty() {
            return Ok(());
        }
        self.sender.send_audio(
            frame.sample_rate,
            usize::from(frame.channels.max(1)),
            &frame.samples,
            None,
        )?;
        Ok(())
    }
}
