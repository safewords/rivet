#!/usr/bin/env python3
"""Run a command and record which GPUs it used, from the kernel's side.

    drm_usage.py OUT.json -- COMMAND [ARGS...]

While COMMAND runs, its /proc/<pid>/fdinfo is sampled every 50 ms. For each
open DRM render node the kernel reports the card (`drm-pdev`, a PCI address),
the client, and the nanoseconds each engine class (render, copy, video,
video-enhance, compute) has spent on that client's work. The last value seen
per client is kept, and summed per card. OUT.json then holds

    {"wall_s": ..., "exit": ..., "cards": {"0000:06:00.0": {"video": ns, ...}}}

This is the independent evidence for which card did the work: it comes from
the i915 driver, not from rivet's own logs. A card the command never opened
does not appear. COMMAND's exit status is this script's exit status.
"""

import json
import os
import subprocess
import sys
import threading
import time


def sample(pid, clients):
    try:
        fds = os.listdir(f"/proc/{pid}/fdinfo")
    except OSError:
        return
    for fd in fds:
        try:
            with open(f"/proc/{pid}/fdinfo/{fd}") as f:
                text = f.read()
        except OSError:
            continue
        if "drm-pdev" not in text:
            continue
        fields = {}
        for line in text.splitlines():
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
        pdev = fields.get("drm-pdev", "").lower()
        client = fields.get("drm-client-id", fd)
        engines = clients.setdefault((pdev, client), {})
        for key, value in fields.items():
            if key.startswith("drm-engine-") and value.endswith(" ns"):
                name = key[len("drm-engine-"):]
                engines[name] = max(engines.get(name, 0), int(value.split()[0]))


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        sys.exit(__doc__)
    out, cmd = sys.argv[1], sys.argv[3:]
    clients = {}
    start = time.monotonic()
    proc = subprocess.Popen(cmd)
    done = threading.Event()

    def watch():
        while not done.is_set():
            sample(proc.pid, clients)
            time.sleep(0.05)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    code = proc.wait()
    done.set()
    watcher.join()
    wall = time.monotonic() - start

    cards = {}
    for (pdev, _client), engines in clients.items():
        card = cards.setdefault(pdev, {})
        for name, ns in engines.items():
            card[name] = card.get(name, 0) + ns
    with open(out, "w") as f:
        json.dump({"cmd": cmd, "wall_s": round(wall, 3), "exit": code, "cards": cards}, f, indent=1)
    busy = ", ".join(
        f"{pdev}: video {c.get('video', 0) / 1e9:.2f}s, enhance {c.get('video-enhance', 0) / 1e9:.2f}s, render {c.get('render', 0) / 1e9:.2f}s"
        for pdev, c in sorted(cards.items())
    )
    print(f"drm_usage: {wall:.1f}s wall, exit {code}; {busy or 'no DRM client'}", file=sys.stderr)
    sys.exit(code)


if __name__ == "__main__":
    main()
