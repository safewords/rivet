//! Live URIs: what stands in for a path where a job's input or output is
//! live.
//!
//! ```text
//! ndi://NAME[?groups=G&extra-ips=IPS&bandwidth=highest|lowest&high-bit-depth]
//! ```
//!
//! `NAME` is a source's full NDI name (`STUDIO (Camera 1)`) or any part of
//! it that names one source; as an output, the stream name to announce.
//! Spaces may be written as is or as `%20` (any `%XX` is decoded). The query
//! carries how to reach or announce it, which belongs to the endpoint, not to
//! the job's spec.

use anyhow::{Context, Result, bail};

/// A parsed live URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiveUri {
    Ndi(NdiEndpoint),
}

/// Which stream an NDI receiver asks the sender for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NdiBandwidth {
    /// The full-quality stream (the default).
    #[default]
    Highest,
    /// The sender's low-bandwidth proxy, where it offers one.
    Lowest,
}

/// An NDI source to receive, or a stream to announce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdiEndpoint {
    /// The source (in) or stream (out) name.
    pub name: String,
    /// NDI groups, comma-separated.
    pub groups: Option<String>,
    /// Machines to ask directly, comma-separated IPs (in).
    pub extra_ips: Option<String>,
    /// Which stream to ask for (in).
    pub bandwidth: NdiBandwidth,
    /// Ask for the source's 16-bit stream (P216) when it sends one (in).
    pub high_bit_depth: bool,
}

/// Whether `s` is a live URI rather than a path.
pub fn is_live_uri(s: &str) -> bool {
    let lower = s.trim_start().to_ascii_lowercase();
    lower.starts_with("ndi://") || lower.starts_with("ndi:")
}

impl LiveUri {
    /// Parse `s`; an error names what is wrong with it.
    pub fn parse(s: &str) -> Result<Self> {
        let t = s.trim();
        let lower = t.to_ascii_lowercase();
        let rest = if lower.starts_with("ndi://") {
            &t[6..]
        } else if lower.starts_with("ndi:") {
            &t[4..]
        } else {
            bail!("'{s}' is not a live URI (ndi://NAME)");
        };
        let (name, query) = match rest.split_once('?') {
            Some((n, q)) => (n, Some(q)),
            None => (rest, None),
        };
        let name = percent_decode(name.trim_end_matches('/'))
            .with_context(|| format!("the source name in '{s}'"))?;
        if name.trim().is_empty() {
            bail!("'{s}' names no NDI source: ndi://NAME, e.g. \"ndi://STUDIO (Camera 1)\"");
        }
        let mut endpoint = NdiEndpoint {
            name,
            groups: None,
            extra_ips: None,
            bandwidth: NdiBandwidth::Highest,
            high_bit_depth: false,
        };
        for pair in query.unwrap_or("").split('&').filter(|p| !p.is_empty()) {
            let (key, value) = match pair.split_once('=') {
                Some((k, v)) => (k, Some(percent_decode(v)?)),
                None => (pair, None),
            };
            let flag = |v: &Option<String>| {
                v.as_deref().is_none_or(|v| {
                    matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
                })
            };
            match key.to_ascii_lowercase().replace('_', "-").as_str() {
                "groups" | "group" => endpoint.groups = value.filter(|v| !v.is_empty()),
                "extra-ips" | "ips" => endpoint.extra_ips = value.filter(|v| !v.is_empty()),
                "bandwidth" => {
                    endpoint.bandwidth = match value.as_deref().map(str::to_ascii_lowercase) {
                        Some(v) if v == "highest" || v == "high" => NdiBandwidth::Highest,
                        Some(v) if v == "lowest" || v == "low" || v == "proxy" => {
                            NdiBandwidth::Lowest
                        }
                        other => bail!(
                            "'{s}': bandwidth must be highest or lowest, got {:?}",
                            other.unwrap_or_default()
                        ),
                    }
                }
                "high-bit-depth" | "16bit" => endpoint.high_bit_depth = flag(&value),
                other => bail!(
                    "'{s}': unknown NDI option `{other}` (groups, extra-ips, bandwidth, high-bit-depth)"
                ),
            }
        }
        Ok(LiveUri::Ndi(endpoint))
    }

    /// The name a default output file is made from: the source's name, its
    /// characters a file name cannot hold replaced.
    pub fn file_stem(&self) -> String {
        match self {
            LiveUri::Ndi(e) => sanitize_stem(&e.name),
        }
    }
}

impl std::fmt::Display for LiveUri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiveUri::Ndi(e) => write!(f, "ndi://{}", e.name),
        }
    }
}

/// `name` as a file stem: letters, digits, `-`, `_`, `.` kept, the rest `_`,
/// runs collapsed, the ends trimmed.
pub fn sanitize_stem(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        let keep = c.is_alphanumeric() || matches!(c, '-' | '_' | '.');
        let c = if keep { c } else { '_' };
        if c == '_' && out.ends_with('_') {
            continue;
        }
        out.push(c);
    }
    let out = out.trim_matches(['_', '.']).to_string();
    if out.is_empty() { "live".into() } else { out }
}

fn percent_decode(s: &str) -> Result<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = s
                .get(i + 1..i + 3)
                .and_then(|h| u8::from_str_radix(h, 16).ok())
                .with_context(|| format!("a bad %-escape at byte {i} of '{s}'"))?;
            out.push(hex);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).context("the decoded name is not UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ndi(s: &str) -> NdiEndpoint {
        match LiveUri::parse(s).unwrap() {
            LiveUri::Ndi(e) => e,
        }
    }

    #[test]
    fn a_name_with_spaces_and_parentheses_is_the_source() {
        let e = ndi("ndi://STUDIO (Camera 1)");
        assert_eq!(e.name, "STUDIO (Camera 1)");
        assert_eq!(e.bandwidth, NdiBandwidth::Highest);
        assert_eq!(ndi("ndi://STUDIO%20(Camera%201)").name, "STUDIO (Camera 1)");
        assert_eq!(ndi("NDI:Program").name, "Program");
    }

    #[test]
    fn the_query_says_how_to_reach_it() {
        let e =
            ndi("ndi://Cam?groups=studio,news&extra-ips=10.0.0.5&bandwidth=lowest&high-bit-depth");
        assert_eq!(e.groups.as_deref(), Some("studio,news"));
        assert_eq!(e.extra_ips.as_deref(), Some("10.0.0.5"));
        assert_eq!(e.bandwidth, NdiBandwidth::Lowest);
        assert!(e.high_bit_depth);
        assert!(!ndi("ndi://Cam?high-bit-depth=0").high_bit_depth);
    }

    #[test]
    fn mistakes_are_refused_by_name() {
        let err = |s| LiveUri::parse(s).unwrap_err().to_string();
        assert!(err("ndi://").contains("names no NDI source"));
        assert!(err("ndi://Cam?colour=1").contains("unknown NDI option `colour`"));
        assert!(err("ndi://Cam?bandwidth=medium").contains("highest or lowest"));
        assert!(err("clip.mp4").contains("not a live URI"));
        assert!(is_live_uri("ndi://x") && !is_live_uri("C:\\clips\\x.mp4"));
    }

    #[test]
    fn a_source_name_makes_a_file_stem() {
        assert_eq!(sanitize_stem("STUDIO (Camera 1)"), "STUDIO_Camera_1");
        assert_eq!(sanitize_stem("??"), "live");
    }
}
