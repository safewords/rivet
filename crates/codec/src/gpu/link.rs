//! The PCI Express link between a card and the CPU — how fast frames can move
//! to and from it.
//!
//! Two cards of one family can differ by several times in what they get done
//! for a transcode, and the slot is often why: devbox's Arc A380 sits on a
//! chipset slot that trains at PCIe 3.0 x2 (about 2 GB/s), its A750 on the
//! CPU's 4.0 x16. Every decoded frame comes back over that link and every frame
//! to encode goes out over it, so the narrow one is the slow one. The
//! multi-GPU scheduler uses this as a *prior* — what to expect of a card
//! before it has been timed — never as the answer.
//!
//! Linux only, from sysfs, without privileges. The link that limits a card is
//! the narrowest one on the path from the root port down to the GPU function:
//! a card behind a chipset is limited by the chipset's own uplink too. The
//! links *inside* a card are left out where they can be told apart: Arc cards
//! (and some AMD ones) carry a PCIe switch on the board, and the GPU function
//! behind it reports a nominal 2.5 GT/s x1 that says nothing about the slot.

use super::types::GpuDevice;

/// One trained PCIe link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcieLink {
    /// Transfer rate per lane, in GT/s (2.5, 5, 8, 16, 32, 64).
    pub gts: f32,
    /// Lane count.
    pub width: u32,
}

impl PcieLink {
    /// Usable bandwidth in one direction, GB/s: the per-lane rate after line
    /// coding (8b/10b to 5 GT/s, 128b/130b from 8 GT/s) times the lanes.
    pub fn gbytes_per_s(&self) -> f64 {
        let gts = f64::from(self.gts);
        let per_lane = if gts <= 5.0 {
            gts * 0.8 / 8.0
        } else {
            gts * (128.0 / 130.0) / 8.0
        };
        per_lane * f64::from(self.width)
    }

    /// The nominal link a function behind an on-card switch reports.
    #[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
    fn is_nominal(&self) -> bool {
        self.gts <= 2.5 && self.width <= 1
    }

    /// `PCIe 4.0 x16 (31.5 GB/s)`.
    pub fn describe(&self) -> String {
        let generation = match self.gts {
            g if g <= 2.5 => "1.0",
            g if g <= 5.0 => "2.0",
            g if g <= 8.0 => "3.0",
            g if g <= 16.0 => "4.0",
            g if g <= 32.0 => "5.0",
            _ => "6.0",
        };
        format!(
            "PCIe {generation} x{} ({:.1} GB/s)",
            self.width,
            self.gbytes_per_s()
        )
    }
}

/// The links on a card's path to the CPU and the one that limits it.
#[derive(Debug, Clone, PartialEq)]
pub struct PcieReport {
    /// `(pci address, link)` for each device from the root port down to the
    /// GPU function, as sysfs reports them.
    pub chain: Vec<(String, PcieLink)>,
    /// The narrowest link that is not an on-card nominal one.
    pub bottleneck: PcieLink,
}

/// The narrowest link of `chain`, leaving out the nominal 2.5 GT/s x1 links
/// an on-card switch reports — unless that is all there is.
#[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
pub fn bottleneck(chain: &[PcieLink]) -> Option<PcieLink> {
    let real: Vec<PcieLink> = chain.iter().copied().filter(|l| !l.is_nominal()).collect();
    let pick = if real.is_empty() {
        chain.to_vec()
    } else {
        real
    };
    pick.into_iter()
        .min_by(|a, b| a.gbytes_per_s().total_cmp(&b.gbytes_per_s()))
}

/// Parse sysfs `current_link_speed` (`"16.0 GT/s PCIe"`, `"8 GT/s"`) and
/// `current_link_width` (`"16"`). `None` for `Unknown`, zero, or anything else.
#[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
pub fn parse_link(speed: &str, width: &str) -> Option<PcieLink> {
    let gts: f32 = speed.split_whitespace().next()?.parse().ok()?;
    let width: u32 = width.trim().parse().ok()?;
    (gts > 0.0 && width > 0).then_some(PcieLink { gts, width })
}

/// The PCI addresses (`0000:03:00.0`) on a canonical sysfs device path, root
/// port first: `/sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0` gives
/// `["0000:00:01.1", "0000:01:00.0"]`.
#[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
pub fn pci_path_components(path: &str) -> Vec<String> {
    path.split('/')
        .filter(|c| {
            let b = c.as_bytes();
            // dddd:bb:dd.f
            b.len() == 12 && b[4] == b':' && b[7] == b':' && b[10] == b'.'
        })
        .map(str::to_string)
        .collect()
}

/// The PCIe path of `device` to the CPU, Linux only. `None` where sysfs is
/// not there or says nothing (an integrated GPU has no link; a VM's virtual
/// bridges may report none).
pub fn pcie_report(device: &GpuDevice) -> Option<PcieReport> {
    #[cfg(target_os = "linux")]
    {
        linux::pcie_report(device)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = device;
        None
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use std::path::Path;

    use super::*;

    pub(super) fn pcie_report(device: &GpuDevice) -> Option<PcieReport> {
        if device.host_pci_address.is_empty() {
            return None;
        }
        let bdf = if device.host_pci_address.matches(':').count() >= 2 {
            device.host_pci_address.clone()
        } else {
            format!("0000:{}", device.host_pci_address)
        };
        let real = std::fs::canonicalize(Path::new("/sys/bus/pci/devices").join(&bdf)).ok()?;
        let components = pci_path_components(&real.to_string_lossy());
        let mut chain = Vec::new();
        for addr in components {
            let dir = Path::new("/sys/bus/pci/devices").join(&addr);
            let speed = std::fs::read_to_string(dir.join("current_link_speed")).unwrap_or_default();
            let width = std::fs::read_to_string(dir.join("current_link_width")).unwrap_or_default();
            if let Some(link) = parse_link(&speed, &width) {
                chain.push((addr, link));
            }
        }
        let links: Vec<PcieLink> = chain.iter().map(|(_, l)| *l).collect();
        let bottleneck = bottleneck(&links)?;
        Some(PcieReport { chain, bottleneck })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_bandwidth_follows_the_line_coding() {
        let gen3x2 = PcieLink { gts: 8.0, width: 2 };
        let gen4x16 = PcieLink {
            gts: 16.0,
            width: 16,
        };
        let gen1x1 = PcieLink { gts: 2.5, width: 1 };
        assert!(
            (gen3x2.gbytes_per_s() - 1.97).abs() < 0.01,
            "{}",
            gen3x2.gbytes_per_s()
        );
        assert!(
            (gen4x16.gbytes_per_s() - 31.5).abs() < 0.1,
            "{}",
            gen4x16.gbytes_per_s()
        );
        assert!((gen1x1.gbytes_per_s() - 0.25).abs() < 1e-9);
        assert_eq!(gen3x2.describe(), "PCIe 3.0 x2 (2.0 GB/s)");
        assert_eq!(gen4x16.describe(), "PCIe 4.0 x16 (31.5 GB/s)");
    }

    #[test]
    fn sysfs_link_attributes_parse() {
        assert_eq!(
            parse_link("16.0 GT/s PCIe\n", "16\n"),
            Some(PcieLink {
                gts: 16.0,
                width: 16
            })
        );
        assert_eq!(
            parse_link("8 GT/s", "2"),
            Some(PcieLink { gts: 8.0, width: 2 })
        );
        assert_eq!(
            parse_link("2.5 GT/s PCIe", "1"),
            Some(PcieLink { gts: 2.5, width: 1 })
        );
        assert_eq!(parse_link("Unknown", "16"), None);
        assert_eq!(parse_link("8.0 GT/s PCIe", "0"), None);
        assert_eq!(parse_link("", ""), None);
    }

    #[test]
    fn the_path_lists_pci_functions_root_port_first() {
        let path = "/sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:01.0/0000:03:00.0";
        assert_eq!(
            pci_path_components(path),
            vec![
                "0000:00:01.1",
                "0000:01:00.0",
                "0000:02:01.0",
                "0000:03:00.0"
            ]
        );
        assert!(pci_path_components("/sys/devices/platform/foo").is_empty());
    }

    /// An Arc card on a chipset slot: the root port's link to the chipset
    /// (4.0 x4), the chipset's link to the card (3.0 x2), then the card's
    /// own switch, whose functions report a nominal 2.5 GT/s x1. The slot is
    /// the limit, not the nominal links.
    #[test]
    fn the_bottleneck_skips_on_card_nominal_links() {
        let chain = [
            PcieLink {
                gts: 16.0,
                width: 4,
            },
            PcieLink { gts: 8.0, width: 2 },
            PcieLink { gts: 2.5, width: 1 },
            PcieLink { gts: 2.5, width: 1 },
        ];
        assert_eq!(bottleneck(&chain), Some(PcieLink { gts: 8.0, width: 2 }));
        // Nothing but nominal links: that is the answer, for want of another.
        assert_eq!(
            bottleneck(&chain[2..]),
            Some(PcieLink { gts: 2.5, width: 1 })
        );
        assert_eq!(bottleneck(&[]), None);
    }
}
