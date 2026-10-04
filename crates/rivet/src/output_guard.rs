//! Outputs that can never be the input.
//!
//! A job reads its whole source into memory and writes its output after, so
//! an output path that names the source file does not fail: it silently
//! replaces the source with the transcode. `rivet transcode x.mp3 --mode
//! audio` once did exactly that, because the default name of an `.mp3`
//! output, `<stem>.mp3`, is the input's own name. Every place rivet writes a
//! file therefore goes through this module:
//!
//! - [`default_beside_input`] picks a default output name that is not the
//!   input: `<stem>.<ext>`, and `<stem>.rivet.<ext>` when that is the input
//!   itself.
//! - [`refuse_input_as_output`] refuses a single output file that resolves
//!   to an input, and [`refuse_input_in_dir`] an output directory that is an
//!   input or holds one where the job writes. "Resolves to" is decided by the
//!   file system, not by spelling: the same file reached through another
//!   case (Windows and macOS file systems ignore case), a `..`, a symbolic
//!   link or a hard link is the same file.
//! - [`write_atomic`] writes a finished output to a temporary file beside the
//!   target and renames it into place, so an existing file at the target is
//!   only ever replaced whole, after the job has succeeded — never truncated
//!   first.

use std::ffi::OsString;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

/// Whether `a` and `b` are the same file on disk (the same file-system
/// object: device and inode on Unix, volume and file index on Windows).
/// Either path missing is `false` — a path that does not exist yet cannot be
/// an existing input.
pub fn is_same_file(a: &Path, b: &Path) -> bool {
    same_file::is_same_file(a, b).unwrap_or(false)
}

/// The default output path beside `input`: `candidate` itself, unless it is
/// the input (an `.mp3` made from an `.mp3`, a `.flac` from a `.flac`, an
/// HLS directory `<stem>.hls` from a file called that), in which case
/// `.rivet` goes before its extension (`x.mp3` → `x.rivet.mp3`), as many
/// times as it takes to name something that is not the input.
pub fn default_beside_input(candidate: PathBuf, input: &Path) -> PathBuf {
    let mut out = candidate;
    // Bounded: each round makes the name longer, and only one file is the
    // input; the bound is a guard, not a limit anything reaches.
    for _ in 0..8 {
        if !aliases(&out, input) {
            return out;
        }
        out = with_rivet_infix(&out);
    }
    out
}

/// Whether `out` names `input`: the same file on disk, or — for a path that
/// cannot be opened — the same spelling.
fn aliases(out: &Path, input: &Path) -> bool {
    is_same_file(out, input) || out == input
}

/// `dir/name.ext` → `dir/name.rivet.ext`; `dir/name` → `dir/name.rivet`.
fn with_rivet_infix(path: &Path) -> PathBuf {
    let stem = path.file_stem().map(|s| s.to_os_string()).unwrap_or_else(|| OsString::from("output"));
    let mut name = stem;
    name.push(".rivet");
    if let Some(ext) = path.extension() {
        name.push(".");
        name.push(ext);
    }
    path.with_file_name(name)
}

/// Refuse to write the output file `output` when it resolves to any of
/// `inputs`.
pub fn refuse_input_as_output(output: &Path, inputs: &[&Path]) -> Result<()> {
    for input in inputs {
        if aliases(output, input) {
            bail!(
                "refusing to write {}: it is the input file {} — the output would replace the source; \
                 choose another output path",
                output.display(),
                input.display()
            );
        }
    }
    Ok(())
}

/// Refuse the output directory `dir` when it is one of `inputs`, or when an
/// input lies inside it where the job may write: `writes_at` says, for an
/// input's path relative to `dir`, whether the job writes there (an HLS
/// package writes `master.m3u8` and everything in its subdirectories). A
/// directory that does not exist yet holds nothing and passes.
pub fn refuse_input_in_dir(dir: &Path, inputs: &[&Path], writes_at: impl Fn(&Path) -> bool) -> Result<()> {
    for input in inputs {
        if aliases(dir, input) {
            bail!(
                "refusing to use {} as the output directory: it is the input file {}",
                dir.display(),
                input.display()
            );
        }
        if !dir.is_dir() {
            continue;
        }
        // Walk up from the input: an ancestor that *is* `dir` (by identity,
        // so a case variant, `..` or a link to it counts) puts the input
        // inside the output directory.
        let Ok(canonical) = std::fs::canonicalize(input) else {
            continue;
        };
        let mut rel = Vec::new();
        let mut cur = canonical.as_path();
        while let Some(parent) = cur.parent() {
            if let Some(name) = cur.file_name() {
                rel.push(name.to_os_string());
            }
            if is_same_file(parent, dir) {
                let rel: PathBuf = rel.iter().rev().collect();
                if writes_at(&rel) {
                    bail!(
                        "refusing to write into {}: the input file {} is inside it, where the job writes \
                         ({}) — choose another output directory",
                        dir.display(),
                        input.display(),
                        rel.display()
                    );
                }
                break;
            }
            cur = parent;
        }
    }
    Ok(())
}

/// Where an HLS package writes, relative to its root: `master.m3u8`, and
/// everything below the root (`video/`, `audio/`, `subs/`).
pub fn hls_package_writes_at(rel: &Path) -> bool {
    rel.components().count() > 1 || rel.as_os_str().eq_ignore_ascii_case("master.m3u8")
}

/// Write `bytes` to `path` by way of a temporary file in the same directory,
/// renamed over `path` once it is complete. A file already at `path` is
/// replaced whole or not at all: a failed write leaves it untouched, and a
/// hard link at `path` is replaced by the new file rather than written
/// through (the file it shared its data with keeps its contents).
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let prefix = path
        .file_name()
        .map(|n| format!(".{}.", n.to_string_lossy()))
        .unwrap_or_else(|| ".rivet-out.".into());
    let mut tmp = tempfile::Builder::new().prefix(&prefix).suffix(".part").tempfile_in(dir)?;
    tmp.write_all(bytes)?;
    tmp.as_file().sync_all()?;
    tmp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether the file system under `dir` ignores case (NTFS and APFS by
    /// default; most Linux file systems do not).
    fn case_insensitive(dir: &Path) -> bool {
        let probe = dir.join("CaseProbe.tmp");
        std::fs::write(&probe, b"x").unwrap();
        let other = dir.join("caseprobe.TMP");
        let yes = other.exists();
        std::fs::remove_file(&probe).unwrap();
        yes
    }

    #[test]
    fn a_default_name_that_is_the_input_gets_rivet_before_its_extension() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("x.mp3");
        std::fs::write(&input, b"source").unwrap();
        assert_eq!(default_beside_input(input.clone(), &input), dir.path().join("x.rivet.mp3"));
        // A name that is not the input is kept.
        let flac = dir.path().join("x.flac");
        assert_eq!(default_beside_input(flac.clone(), &input), flac);
        // An input already called `.rivet.` gets another.
        let again = dir.path().join("x.rivet.mp3");
        std::fs::write(&again, b"source").unwrap();
        assert_eq!(default_beside_input(again.clone(), &again), dir.path().join("x.rivet.rivet.mp3"));
        // A directory default with the input's name (`clip.hls` the file).
        let hls_file = dir.path().join("clip.hls");
        std::fs::write(&hls_file, b"source").unwrap();
        assert_eq!(default_beside_input(hls_file.clone(), &hls_file), dir.path().join("clip.rivet.hls"));
    }

    #[test]
    fn the_input_by_any_spelling_is_refused_as_the_output() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        let input = sub.join("Song.mp3");
        std::fs::write(&input, b"source").unwrap();
        // Verbatim, and through `..`.
        assert!(refuse_input_as_output(&input, &[&input]).is_err());
        let dotted = sub.join("..").join("sub").join("Song.mp3");
        assert!(refuse_input_as_output(&dotted, &[&input]).is_err());
        // Another file, and one not there yet, pass.
        assert!(refuse_input_as_output(&sub.join("other.mp3"), &[&input]).is_ok());
        // Any of several inputs (a splice).
        let second = sub.join("b.mp4");
        std::fs::write(&second, b"source").unwrap();
        assert!(refuse_input_as_output(&second, &[&input, &second]).is_err());
        // A case variant: the same file where the file system ignores case,
        // another (absent) file where it does not.
        let upper = sub.join("SONG.MP3");
        let lower = dir.path().join("SUB").join("song.mp3");
        if case_insensitive(dir.path()) {
            assert!(refuse_input_as_output(&upper, &[&input]).is_err(), "case variant");
            assert!(refuse_input_as_output(&lower, &[&input]).is_err(), "case variant of the directory");
            assert_eq!(default_beside_input(upper, &input), sub.join("SONG.rivet.MP3"));
        } else {
            assert!(refuse_input_as_output(&upper, &[&input]).is_ok());
        }
    }

    #[test]
    fn a_hard_link_to_the_input_is_the_input() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("a.flac");
        std::fs::write(&input, b"source").unwrap();
        let link = dir.path().join("b.flac");
        if std::fs::hard_link(&input, &link).is_err() {
            return; // a file system without hard links
        }
        assert!(refuse_input_as_output(&link, &[&input]).is_err());
        // Written atomically, the link is replaced and the input survives.
        write_atomic(&link, b"new").unwrap();
        assert_eq!(std::fs::read(&input).unwrap(), b"source");
        assert_eq!(std::fs::read(&link).unwrap(), b"new");
    }

    #[cfg(unix)]
    #[test]
    fn a_symbolic_link_to_the_input_is_the_input() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("a.mp4");
        std::fs::write(&input, b"source").unwrap();
        let link = dir.path().join("l.mp4");
        std::os::unix::fs::symlink(&input, &link).unwrap();
        assert!(refuse_input_as_output(&link, &[&input]).is_err());
    }

    #[test]
    fn an_input_inside_an_hls_package_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("pkg");
        std::fs::create_dir_all(root.join("video/720p")).unwrap();
        let top = root.join("clip.mp4");
        std::fs::write(&top, b"source").unwrap();
        let deep = root.join("video/720p/seg-00001.m4s");
        std::fs::write(&deep, b"source").unwrap();
        let master = root.join("master.m3u8");
        std::fs::write(&master, b"source").unwrap();
        // Beside the package at the top level: not written there.
        assert!(refuse_input_in_dir(&root, &[&top], hls_package_writes_at).is_ok());
        // Where the package writes.
        assert!(refuse_input_in_dir(&root, &[&deep], hls_package_writes_at).is_err());
        assert!(refuse_input_in_dir(&root, &[&master], hls_package_writes_at).is_err());
        // The directory is the input file itself.
        assert!(refuse_input_in_dir(&top, &[&top], hls_package_writes_at).is_err());
        // A directory not made yet holds nothing.
        assert!(refuse_input_in_dir(&dir.path().join("new"), &[&deep], hls_package_writes_at).is_ok());
        if case_insensitive(dir.path()) {
            let upper = dir.path().join("PKG");
            assert!(refuse_input_in_dir(&upper, &[&deep], hls_package_writes_at).is_err(), "case variant");
        }
    }

    #[test]
    fn an_atomic_write_replaces_whole_and_leaves_no_temporary() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.mp4");
        std::fs::write(&path, b"old contents, longer than the new").unwrap();
        write_atomic(&path, b"new").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        let names: Vec<_> = std::fs::read_dir(dir.path()).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(names, vec![OsString::from("out.mp4")], "no .part left behind");
        // A target in a directory that is not there fails, writing nothing.
        assert!(write_atomic(&dir.path().join("missing/out.mp4"), b"x").is_err());
    }
}
