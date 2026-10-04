#!/usr/bin/env python3
"""Times rivet on each card alone and on all of them, and checks that the
cards together are no slower than the fastest card alone.

    speed_aware_timings.py RIVET SRC_LADDER SRC_SINGLE [--base RIVET_BASE]
                           [--vendor intel] [--repeat N]

For every plan — each card pinned (`--encode gpu:N --decode gpu:N`), and all
the cards (`--encode family:VENDOR`, decode `auto` and `whole`) — it runs a
three-rung AV1 HLS ladder on SRC_LADDER and a one-rung H.265 single file on
SRC_SINGLE and on SRC_LADDER (reported only), takes the best of N wall times, and prints a Markdown table
(`--base` adds the same runs with another build, e.g. develop's, as a
"before" column). With two or more cards it then asserts, for this build:

- the ladder on all the cards beats the fastest card alone (<= 0.95x);
- the single rung on all the cards is no slower than the fastest card alone
  (<= 1.10x): a short single file is one or two chunks, which no split can
  make faster than one fast card, but must not make slower either.

Exits 0 with "skipped" on a host with fewer than two cards of the vendor.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time


def cards(rivet, vendor):
    out = subprocess.run([rivet, "devices", "--json"], check=True, capture_output=True, text=True).stdout
    gpus = json.loads(out)["gpus"]
    found = [g for g in gpus if vendor in json.dumps(g).lower()]
    for g in found:
        pcie = g.get("pcie") or {}
        print(f"gpu {g['index']}: {g['name']}, {g.get('vram_mib', 0)} MiB, "
              f"PCIe {pcie.get('gts', '?')} GT/s x{pcie.get('width', '?')}, chain {pcie.get('chain')}")
    return sorted(int(g["index"]) for g in found)


def run(rivet, args, log):
    with tempfile.TemporaryDirectory() as out_dir:
        out = os.path.join(out_dir, "out" if "--mode" in args else "out.mp4")
        t = time.monotonic()
        with open(log, "w") as f:
            done = subprocess.run([rivet, "transcode", *args[:1], "-o", out, *args[1:]], stdout=f, stderr=subprocess.STDOUT)
        wall = time.monotonic() - t
        if done.returncode != 0:
            sys.stdout.write(open(log).read()[-4000:])
            raise SystemExit(f"{' '.join(args)} failed ({done.returncode}); log {log}")
        return wall


def fmt(t):
    return "failed" if t != t else f"{t:.2f} s"


def best_of(rivet, args, log, repeat):
    return min(run(rivet, args, log) for _ in range(repeat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rivet")
    ap.add_argument("ladder_src")
    ap.add_argument("single_src")
    ap.add_argument("--base")
    ap.add_argument("--vendor", default="intel")
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()

    found = cards(a.rivet, a.vendor)
    if len(found) < 2:
        print(f"skipped: {len(found)} {a.vendor} card(s) on this host")
        return
    ladder = [a.ladder_src, "--mode", "hls", "--codec", "av1", "--rung", "1920x1080", "--rung", "1280x720", "--rung", "854x480"]
    single = [a.single_src, "--codec", "h265", "--rung", "1920x1080"]
    single_long = [a.ladder_src, "--codec", "h265", "--rung", "1920x1080"]
    plans = [(f"gpu {g} alone", ["--encode", f"gpu:{g}", "--decode", f"gpu:{g}"]) for g in found]
    plans += [
        ("all cards, decode auto", ["--encode", f"family:{a.vendor}", "--decode", "auto"]),
        ("all cards, decode whole", ["--encode", f"family:{a.vendor}", "--decode", "whole"]),
    ]
    builds = [("after", a.rivet)] + ([("before", a.base)] if a.base else [])
    times = {}
    jobs = (("ladder", ladder), ("single", single), ("single-long", single_long))
    for job, job_args in jobs:
        for plan, plan_args in plans:
            for build, binary in builds:
                tag = f"{job}-{plan}-{build}".replace(" ", "-").replace(",", "")
                try:
                    times[(job, plan, build)] = best_of(binary, job_args + plan_args, f"timing-{tag}.log", a.repeat)
                except SystemExit as e:
                    # The "before" build is only a comparison: its failure is
                    # reported, not fatal.
                    if build != "before":
                        raise
                    print(f"before build failed: {e}")
                    times[(job, plan, build)] = float("nan")
                print(f"{job:6} {plan:26} {build:6} {times[(job, plan, build)]:6.2f} s", flush=True)

    head = "| job | plan | " + " | ".join(b for b, _ in builds) + " |"
    lines = [head, "|" + "---|" * (2 + len(builds))]
    for job, _ in jobs:
        for plan, _ in plans:
            lines.append(f"| {job} | {plan} | " + " | ".join(fmt(times[(job, plan, b)]) for b, _ in builds) + " |")
    table = "\n".join(lines)
    print(table)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("\n### Speed-aware scheduling: wall times (best of %d)\n\n%s\n" % (a.repeat, table))

    failures = []
    # single-long (the ladder source as one rung: two chunks) is reported,
    # not judged: two chunks of unequal length on cards 3x apart.
    for job, limit in (("ladder", 0.95), ("single", 1.10), ("single-long", None)):
        alone = {g: times[(job, f"gpu {g} alone", "after")] for g in found}
        fastest = min(alone, key=alone.get)
        together = times[(job, "all cards, decode auto", "after")]
        ratio = together / alone[fastest]
        verdict = "reported" if limit is None else ("ok" if ratio <= limit else "FAIL")
        line = f"{job}: all cards {together:.2f} s vs gpu {fastest} alone {alone[fastest]:.2f} s = {ratio:.2f}x (limit {limit}x) {verdict}"
        print(line)
        if summary:
            with open(summary, "a") as f:
                f.write(f"- {line}\n")
        if limit is not None and ratio > limit:
            failures.append(line)
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
