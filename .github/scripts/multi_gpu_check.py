#!/usr/bin/env python3
"""Assertions for the two-card Intel GPU job (ci.yml, `intel-gpu-x2`).

    multi_gpu_check.py devices DEVICES.json CARDS.json
        `rivet devices --json` lists exactly two Intel cards, at different PCI
        addresses; writes {"0": "<pci>", "1": "<pci>"} (rivet's index to the
        card) to CARDS.json.
    multi_gpu_check.py pinned USAGE.json CARDS.json N
        drm_usage.py saw the video engines of card N busy, and nothing at all
        on the other card: a job pinned to N ran on N.
    multi_gpu_check.py both USAGE.json CARDS.json
        drm_usage.py saw the video engines of both cards busy.
    multi_gpu_check.py chunks LOG
        rivet's log: the ladder workers' `rung chunk done` lines name both
        cards (gpu_index 0 and 1).
    multi_gpu_check.py faster LABEL SECONDS BASELINE_SECONDS MAX_RATIO
        SECONDS <= BASELINE_SECONDS * MAX_RATIO.
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
    if len(intel) != 2:
        fail(f"expected 2 Intel cards, rivet sees {len(intel)}: {gpus}")
    addrs = {pci(g["pci"]) for g in intel}
    if len(addrs) != 2:
        fail(f"the two cards share a PCI address: {addrs}")
    for g in intel:
        if not all(g.get("encode", {}).get(c) for c in ("av1", "h264", "h265")):
            fail(f"gpu {g['index']} cannot encode all of AV1, H.264, H.265: {g.get('encode')}")
    with open(cards_json, "w") as f:
        json.dump({str(g["index"]): pci(g["pci"]) for g in intel}, f)


def pinned(usage_json, cards_json, n):
    usage, cards = load(usage_json), load(cards_json)
    seen = {pci(k): v for k, v in usage["cards"].items()}
    mine = cards[n]
    other = next(addr for idx, addr in cards.items() if idx != n)
    if video_ns(seen.get(mine, {})) <= 0:
        fail(f"{usage['cmd']}: card {n} ({mine}) did no video work; the kernel saw {seen}")
    if any(ns > 0 for ns in seen.get(other, {}).values()):
        fail(f"{usage['cmd']}: pinned to card {n}, but the other card ({other}) was used: {seen[other]}")
    print(f"card {n} ({mine}) only: video {video_ns(seen[mine]) / 1e9:.2f}s in {usage['wall_s']}s")


def both(usage_json, cards_json):
    usage, cards = load(usage_json), load(cards_json)
    seen = {pci(k): v for k, v in usage["cards"].items()}
    for idx, addr in sorted(cards.items()):
        ns = video_ns(seen.get(addr, {}))
        print(f"card {idx} ({addr}): video {ns / 1e9:.2f}s in {usage['wall_s']}s")
        if ns <= 0:
            fail(f"{usage['cmd']}: card {idx} ({addr}) did no video work; the kernel saw {seen}")


def chunks(log):
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
    if not ("0" in counts and "1" in counts):
        fail(f"{log}: the ladder's chunks did not land on both cards: {counts}")


def faster(label, seconds, baseline, ratio):
    seconds, baseline, ratio = float(seconds), float(baseline), float(ratio)
    print(f"{label}: {seconds:.1f}s against {baseline:.1f}s, {baseline / seconds:.2f}x")
    if seconds > baseline * ratio:
        fail(f"{label}: {seconds:.1f}s is not under {ratio} x {baseline:.1f}s")


COMMANDS = {"devices": devices, "pinned": pinned, "both": both, "chunks": chunks, "faster": faster}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    COMMANDS[sys.argv[1]](*sys.argv[2:])
