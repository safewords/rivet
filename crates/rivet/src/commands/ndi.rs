//! Implementation of `rivet ndi` (the `ndi` feature): discovery. Receiving
//! and sending are `rivet transcode` with an `ndi://` input or output.

use std::time::Duration;

use anyhow::Result;

use super::esc;

/// `rivet ndi`'s subcommands.
#[derive(clap::Subcommand)]
pub(crate) enum NdiCommand {
    /// List the NDI sources on the network. Record one with `rivet
    /// transcode "ndi://NAME" -o out.mp4`; send a file with `rivet transcode
    /// FILE -o ndi://NAME`.
    Sources {
        /// Seconds to listen for sources before listing them.
        #[arg(long, default_value_t = 3.0, value_name = "SECONDS")]
        wait: f64,
        /// NDI groups to look in, comma-separated (default: the runtime's).
        #[arg(long, value_name = "GROUPS")]
        groups: Option<String>,
        /// Further machines to ask directly, comma-separated IP addresses
        /// (for networks mDNS does not cross).
        #[arg(long = "extra-ips", value_name = "IPS")]
        extra_ips: Option<String>,
        /// Print JSON instead of a list.
        #[arg(long)]
        json: bool,
    },
}

pub(crate) fn run(command: NdiCommand) -> Result<()> {
    match command {
        NdiCommand::Sources {
            wait,
            groups,
            extra_ips,
            json,
        } => {
            anyhow::ensure!(
                wait.is_finite() && wait >= 0.0,
                "--wait must be a number of seconds"
            );
            let wait = Duration::from_secs_f64(wait);
            let options = ndi::FindOptions {
                groups,
                extra_ips,
                ..Default::default()
            };
            let found = rivet::ndi::list_sources(&options, wait)?;
            if json {
                let items: Vec<String> = found
                    .iter()
                    .map(|s| {
                        format!(
                            "{{\"name\":\"{}\",\"uri\":\"ndi://{}\",\"url\":{}}}",
                            esc(&s.name),
                            esc(&s.name),
                            s.url
                                .as_deref()
                                .map_or("null".to_string(), |u| format!("\"{}\"", esc(u)))
                        )
                    })
                    .collect();
                println!("[{}]", items.join(","));
            } else if found.is_empty() {
                eprintln!("no NDI sources seen in {:.1} s", wait.as_secs_f64());
            } else {
                for s in &found {
                    match &s.url {
                        Some(u) => println!("ndi://{}  ({u})", s.name),
                        None => println!("ndi://{}", s.name),
                    }
                }
            }
            Ok(())
        }
    }
}
