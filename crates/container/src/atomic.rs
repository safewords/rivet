//! Output files that appear whole or not at all.
//!
//! Every file the muxers write into an output (an HLS package's segments,
//! init segments and playlists, a WebVTT rendition's documents) is written to
//! a temporary file beside its target and renamed over the target once it is
//! complete. A job that fails or is killed part way never leaves a truncated
//! file under a name a player or a later step would trust: what is at the
//! target is either the previous file, untouched, or the complete new one.
//! The temporary files are hidden (`.<name>.<random>.part`), removed when a
//! write is abandoned, and never named by a playlist.
//!
//! The rename also means a hard link at the target is replaced, not written
//! through: whatever the target shared its data with keeps its contents.

use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use tempfile::NamedTempFile;

/// A file being written to `target` by way of a temporary file in the same
/// directory. Write to it, then [`AtomicFile::commit`]; dropping it without
/// committing removes the temporary file and leaves `target` as it was.
pub struct AtomicFile {
    tmp: BufWriter<NamedTempFile>,
    target: PathBuf,
}

impl AtomicFile {
    /// Start writing `target`. Its directory must exist.
    pub fn create(target: &Path) -> io::Result<Self> {
        let dir = match target.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let prefix = target
            .file_name()
            .map(|n| format!(".{}.", n.to_string_lossy()))
            .unwrap_or_else(|| ".rivet-out.".into());
        let tmp = tempfile::Builder::new()
            .prefix(&prefix)
            .suffix(".part")
            .tempfile_in(dir)?;
        Ok(Self {
            tmp: BufWriter::new(tmp),
            target: target.to_path_buf(),
        })
    }

    /// The path this file becomes on [`Self::commit`].
    pub fn target(&self) -> &Path {
        &self.target
    }

    /// Flush, sync and rename the complete file over the target.
    pub fn commit(self) -> io::Result<()> {
        let tmp = self.tmp.into_inner().map_err(|e| e.into_error())?;
        tmp.as_file().sync_all()?;
        tmp.persist(&self.target).map_err(|e| e.error)?;
        Ok(())
    }
}

impl Write for AtomicFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.tmp.write(buf)
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.tmp.write_all(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.tmp.flush()
    }
}

/// Write `bytes` to `path` whole or not at all (see [`AtomicFile`]).
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut f = AtomicFile::create(path)?;
    f.write_all(bytes)?;
    f.commit()
}

/// The temporary files [`AtomicFile`] leaves while a write is in flight are
/// named `.<target>.<random>.part`: whether `name` is one.
pub fn is_temporary_name(name: &str) -> bool {
    name.starts_with('.') && name.ends_with(".part")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    #[test]
    fn commit_writes_the_whole_file_and_leaves_no_temporary() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg-00001.m4s");
        let mut f = AtomicFile::create(&path).unwrap();
        f.write_all(b"moof").unwrap();
        f.write_all(b"mdat").unwrap();
        // Nothing at the target until the commit.
        assert!(!path.exists());
        f.commit().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"moofmdat");
        assert_eq!(names(dir.path()), ["seg-00001.m4s"]);
    }

    #[test]
    fn an_abandoned_write_leaves_the_old_file_and_no_temporary() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("playlist.m3u8");
        std::fs::write(&path, b"#EXTM3U\nold\n").unwrap();
        {
            let mut f = AtomicFile::create(&path).unwrap();
            f.write_all(b"#EXTM3U\nhalf a new pl").unwrap();
            f.flush().unwrap();
            // A temporary file is in flight beside the target.
            assert!(names(dir.path()).iter().any(|n| is_temporary_name(n)));
            // Dropped without commit: the job failed here.
        }
        assert_eq!(std::fs::read(&path).unwrap(), b"#EXTM3U\nold\n");
        assert_eq!(names(dir.path()), ["playlist.m3u8"]);
    }

    #[test]
    fn replaces_an_existing_file_whole() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("init.mp4");
        std::fs::write(&path, b"an old init segment, longer than the new one").unwrap();
        write_atomic(&path, b"new").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
    }

    #[test]
    fn a_hard_link_at_the_target_is_replaced_not_written_through() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.mp4");
        std::fs::write(&source, b"source").unwrap();
        let link = dir.path().join("seg-00001.m4s");
        if std::fs::hard_link(&source, &link).is_err() {
            return; // a file system without hard links
        }
        write_atomic(&link, b"segment").unwrap();
        assert_eq!(std::fs::read(&source).unwrap(), b"source");
        assert_eq!(std::fs::read(&link).unwrap(), b"segment");
    }

    #[test]
    fn a_missing_directory_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(write_atomic(&dir.path().join("missing/seg.m4s"), b"x").is_err());
    }
}
