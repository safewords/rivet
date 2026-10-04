#!/usr/bin/env python3
"""Assertions for the Intel GPU jobs (ci.yml, `gpu-*`). A runner holds every
card of its host, however many that is.

    multi_gpu_check.py devices DEVICES.json CARDS.json
        `rivet devices --json` lists at least one Intel card, every one at its
        own PCI address and able to encode AV1, H.264 and H.265; writes
        {"<index>": "<pci>", ...} (rivet's index to the card) to CARDS.json
        and prints the count as `cards=N` (for $GITHUB_OUTPUT).
    multi_gpu_check.py pinned USAGE.json CARDS.json N
        drm_usage.py saw the video engines of card N busy, and nothing at all
        on any other card: a job pinned to N ran on N.
    multi_gpu_check.py all USAGE.json CARDS.json
        drm_usage.py saw the video engines of every card busy.
    multi_gpu_check.py chunks LOG CARDS.json
        rivet's log: the ladder workers' `rung chunk done` lines name every
        card.
    multi_gpu_check.py faster LABEL SECONDS BASELINE_SECONDS MAX_RATIO
        SECONDS <= BASELINE_SECONDS * MAX_RATIO.
    multi_gpu_check.py timings CARDS.json OUT.json
        Collects the wall time of every usage-*.json in the working directory
        (drm_usage.py's records, named after the run: ladder-gpu-N,
        ladder-family-intel, busy-CODEC-..., alone-N, together-N) with the
        cards, into OUT.json, and prints them as a Markdown table.
"""

import json
import re
import sys


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def pci(addr):
    addr = addr.strip().lower()
    return addr if addr.count(":") == 2 else f"0000:{addr}"


def video_ns(card):
    return card.get("video", 0) + card.get("video-enhance", 0)


def load(path):
    with open(path) as f:
        return json.load(f)


def devices(devices_json, cards_json):
    gpus = load(devices_json)["gpus"]
    intel = [g for g in gpus if g.get("vendor", "").lower() == "intel"]
    for g in intel:
        print(f"gpu {g['index']}: {g['name']} ({g.get('generation')}), {g.get('vram_mib')} MiB, PCI {g['pci']}, encode {g.get('encode')}")
    if not intel:
        fail(f"rivet sees no Intel card: {gpus}")
    addrs = {pci(g["pci"]) for g in intel}
    if len(addrs) != len(intel):
        fail(f"cards share a PCI address: {[g['pci'] for g in intel]}")
    for g in intel:
        if not all(g.get("encode", {}).get(c) for c in ("av1", "h264", "h265")):
            fail(f"gpu {g['index']} cannot encode all of AV1, H.264, H.265: {g.get('encode')}")
    with open(cards_json, "w") as f:
        json.dump({str(g["index"]): pci(g["pci"]) for g in intel}, f)
    print(f"cards={len(intel)}")


def pinned(usage_json, cards_json, n):
    usage, cards = load(usage_json), load(cards_json)
    seen = {pci(k): v for k, v in usage["cards"].items()}
    mine = cards[n]
    if video_ns(seen.get(mine, {})) <= 0:
        fail(f"{usage['cmd']}: card {n} ({mine}) did no video work; the kernel saw {seen}")
    for idx, other in cards.items():
        if idx != n and any(ns > 0 for ns in seen.get(other, {}).values()):
            fail(f"{usage['cmd']}: pinned to card {n}, but card {idx} ({other}) was used: {seen[other]}")
    print(f"card {n} ({mine}) only: video {video_ns(seen[mine]) / 1e9:.2f}s in {usage['wall_s']}s")


def every(usage_json, cards_json):
    usage, cards = load(usage_json), load(cards_json)
    seen = {pci(k): v for k, v in usage["cards"].items()}
    for idx, addr in sorted(cards.items()):
        ns = video_ns(seen.get(addr, {}))
        print(f"card {idx} ({addr}): video {ns / 1e9:.2f}s in {usage['wall_s']}s")
        if ns <= 0:
            fail(f"{usage['cmd']}: card {idx} ({addr}) did no video work; the kernel saw {seen}")


def chunks(log, cards_json):
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    counts = {}
    with open(log, errors="replace") as f:
        for line in f:
            line = ansi.sub("", line)
            if "rung chunk done" not in line:
                continue
            m = re.search(r"gpu_index=Some\((\d+)\)", line)
            key = m.group(1) if m else "none"
            counts[key] = counts.get(key, 0) + 1
    print(f"chunks per card: {counts}")
    missing = [idx for idx in sorted(load(cards_json)) if idx not in counts]
    if missing:
        fail(f"{log}: no chunk was encoded on card(s) {missing}: {counts}")


def faster(label, seconds, baseline, ratio):
    seconds, baseline, ratio = float(seconds), float(baseline), float(ratio)
    print(f"{label}: {seconds:.1f}s against {baseline:.1f}s, {baseline / seconds:.2f}x")
    if seconds > baseline * ratio:
        fail(f"{label}: {seconds:.1f}s is not under {ratio} x {baseline:.1f}s")


def timings(cards_json, out_json):
    import glob
    import os
    cards = load(cards_json) if os.path.exists(cards_json) else {}
    devices = load("devices.json")["gpus"] if os.path.exists("devices.json") else []
    names = {str(g["index"]): g["name"] for g in devices}
    runs = {}
    for path in sorted(glob.glob("usage-*.json")):
        usage = load(path)
        seen = {pci(k): v for k, v in usage["cards"].items()}
        busy = {idx: round(video_ns(seen.get(addr, {})) / 1e9, 3) for idx, addr in cards.items()}
        runs[path[len("usage-"):-len(".json")]] = {"wall_s": usage["wall_s"], "video_busy_s": busy}
    with open(out_json, "w") as f:
        json.dump({"cards": {i: {"pci": a, "name": names.get(i)} for i, a in cards.items()}, "runs": runs}, f, indent=1)
    order = sorted(cards, key=int)
    print("| run | wall (s) | " + " | ".join(f"card {i} {names.get(i, '')} video busy (s)" for i in order) + " |")
    print("|---|---|" + "---|" * len(order))
    for name, r in runs.items():
        print(f"| {name} | {r['wall_s']} | " + " | ".join(str(r["video_busy_s"][i]) for i in order) + " |")


COMMANDS = {"devices": devices, "pinned": pinned, "all": every, "chunks": chunks, "faster": faster, "timings": timings}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    COMMANDS[sys.argv[1]](*sys.argv[2:])
