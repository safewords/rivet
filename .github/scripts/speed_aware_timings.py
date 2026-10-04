#!/usr/bin/env python3
"""Times rivet on each card alone and on all of them, interleaved, and checks
that the cards together beat (or, for a short file, keep up with) the
fastest card alone.

    speed_aware_timings.py RIVET SRC_LADDER SRC_SINGLE [--base RIVET_BASE]
                           [--vendor intel] [--repeat N] [--json OUT.json]

Jobs, each run under every plan:

- `ladder`: a three-rung AV1 HLS ladder on SRC_LADDER. Plans: each card
  pinned (`--encode gpu:N --decode gpu:N`), and all the cards
  (`--encode family:VENDOR`, decode `auto` — the default — and `whole`).
- `single`: a one-rung H.265 single file on SRC_SINGLE; same plans.
- `single-long`: the same on SRC_LADDER (reported only: two chunks of
  unequal length on cards 3x apart).
- `busy-av1`, `busy-h265`: the ladder at `--video-speed archive` with the
  decode on the fastest card, so the encode is the limit. Plans: the
  fastest card's encoder alone, and every card's (`family:VENDOR`).

The measurement is built to compare within one run, not across runs:

- **Interleaved**: N rounds, and every round runs every plan of a job once,
  in an order rotated from round to round. Load that comes and goes on the
  host (another pod's CPU work, a clock ramp) lands on all the plans alike
  instead of on whichever plan happened to run while it lasted.
- **Best of N**: each plan's time is its fastest round. Outside load only
  ever slows a run, so the minimum is the closest to the hardware's own
  time; the ratios are of these minima.
- **Recorded**: every round's time, each plan's spread ((max - min) / min)
  and every round's paired ratio go to the step summary and to OUT.json.

With two or more cards it then asserts, for this build:

- `ladder`: all the cards at most 0.95x the fastest card alone. devbox
  (A750 + A380) measures 0.78-0.79x run after run, so 0.95 leaves room for
  noise and still fails the moment a second card stops paying for itself.
- `single`: at most 1.10x. A 10 s file is one or two chunks, which no split
  can make faster than one fast card, but which must not be made slower
  either; devbox measures 1.06-1.07x (a fixed ~60 ms of opening both cards).
- `busy-av1`, `busy-h265`: at most 0.85x. devbox measures 0.66-0.71x.

Exits 0 with "skipped" on a host with fewer than two cards of the vendor.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

LIMITS = {"ladder": 0.95, "single": 1.10, "single-long": None, "busy-av1": 0.85, "busy-h265": 0.85}


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


def rounds(jobs, plans, builds, repeat, samples):
    """Runs every (job, plan, build) `repeat` times: round by round, each
    job's plans in an order rotated by the round number."""
    for r in range(repeat):
        for job, job_args in jobs:
            cells = [(plan, plan_args, build, binary) for plan, plan_args in plans[job] for build, binary in builds]
            k = r % len(cells)
            for plan, plan_args, build, binary in cells[k:] + cells[:k]:
                tag = f"{job}-{plan}-{build}".replace(" ", "-").replace(",", "")
                try:
                    t = run(binary, job_args + plan_args, f"timing-{tag}.log")
                except SystemExit as e:
                    # The "before" build is only a comparison: its failure is
                    # reported, not fatal.
                    if build != "before":
                        raise
                    print(f"before build failed: {e}")
                    t = float("nan")
                samples.setdefault((job, plan, build), []).append(t)
                print(f"round {r + 1}/{repeat} {job:11} {plan:26} {build:6} {t:6.2f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rivet")
    ap.add_argument("ladder_src")
    ap.add_argument("single_src")
    ap.add_argument("--base")
    ap.add_argument("--vendor", default="intel")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--json", default="speed_aware_timings.json")
    a = ap.parse_args()

    found = cards(a.rivet, a.vendor)
    if len(found) < 2:
        print(f"skipped: {len(found)} {a.vendor} card(s) on this host")
        return
    rungs = ["--rung", "1920x1080", "--rung", "1280x720", "--rung", "854x480"]
    ladder = [a.ladder_src, "--mode", "hls", "--codec", "av1", *rungs]
    single = [a.single_src, "--codec", "h265", "--rung", "1920x1080"]
    single_long = [a.ladder_src, "--codec", "h265", "--rung", "1920x1080"]
    alone = [(f"gpu {g} alone", ["--encode", f"gpu:{g}", "--decode", f"gpu:{g}"]) for g in found]
    together = [
        ("all cards, decode auto", ["--encode", f"family:{a.vendor}", "--decode", "auto"]),
        ("all cards, decode whole", ["--encode", f"family:{a.vendor}", "--decode", "whole"]),
    ]
    builds = [("after", a.rivet)] + ([("before", a.base)] if a.base else [])
    samples = {}

    def best(job, plan, build="after"):
        return min(samples[(job, plan, build)])

    # Phase 1: every card alone and all of them, for each job.
    jobs = [("ladder", ladder), ("single", single), ("single-long", single_long)]
    rounds(jobs, {job: alone + together for job, _ in jobs}, builds, a.repeat, samples)
    alone_ladder = {g: best("ladder", f"gpu {g} alone") for g in found}
    fast = min(alone_ladder, key=alone_ladder.get)
    print(f"the fastest card (ladder, best of {a.repeat}): gpu {fast}")

    # Phase 2: encode-bound, on the fastest card alone and on all of them.
    busy_jobs = [(f"busy-{codec}", [a.ladder_src, "--mode", "hls", "--codec", codec, *rungs,
                                   "--video-speed", "archive", "--decode", f"gpu:{fast}"]) for codec in ("av1", "h265")]
    busy_plans = [(f"gpu {fast} alone", ["--encode", f"gpu:{fast}"]), ("all cards", ["--encode", f"family:{a.vendor}"])]
    rounds(busy_jobs, {job: busy_plans for job, _ in busy_jobs}, builds, a.repeat, samples)
    jobs += busy_jobs

    # Every round of every plan, its best and its spread.
    head = "| job | plan | build | rounds (s) | best | spread |"
    lines = [head, "|---|---|---|---|---|---|"]
    for (job, plan, build), ts in samples.items():
        ok = [t for t in ts if t == t]
        spread = f"{(max(ok) - min(ok)) / min(ok) * 100:.1f}%" if ok else "-"
        lines.append(f"| {job} | {plan} | {build} | " + ", ".join(f"{t:.2f}" for t in ts)
                     + f" | {fmt(min(ok) if ok else float('nan'))} | {spread} |")
    table = "\n".join(lines)
    print(table)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")

    def out(text):
        if summary:
            with open(summary, "a") as f:
                f.write(text)

    out(f"\n### Speed-aware scheduling: wall times, {a.repeat} interleaved rounds\n\n{table}\n\n")

    # The judgement: best of N on all cards against best of N on the fastest
    # card alone, both from this run. The paired ratios (round r against
    # round r) are reported as the measure of how noisy this run was.
    failures, verdicts = [], {}
    for job, _ in jobs:
        limit = LIMITS[job]
        if job.startswith("busy-"):
            fastest, all_plan = f"gpu {fast} alone", "all cards"
        else:
            fastest = min((f"gpu {g} alone" for g in found), key=lambda p: best(job, p))
            all_plan = "all cards, decode auto"
        ratio = best(job, all_plan) / best(job, fastest)
        paired = [t / s for t, s in zip(samples[(job, all_plan, "after")], samples[(job, fastest, "after")])]
        verdict = "reported" if limit is None else ("ok" if ratio <= limit else "FAIL")
        line = (f"{job}: all cards {best(job, all_plan):.2f} s vs {fastest} {best(job, fastest):.2f} s "
                f"= {ratio:.3f}x (limit {limit}x; rounds {', '.join(f'{p:.3f}' for p in paired)}) {verdict}")
        print(line)
        out(f"- {line}\n")
        verdicts[job] = {"all": all_plan, "fastest": fastest, "ratio": ratio, "paired": paired,
                         "limit": limit, "verdict": verdict}
        if verdict == "FAIL":
            failures.append(line)

    with open(a.json, "w") as f:
        json.dump({"repeat": a.repeat, "cards": found,
                   "samples": [{"job": j, "plan": p, "build": b, "wall_s": ts} for (j, p, b), ts in samples.items()],
                   "verdicts": verdicts}, f, indent=1)
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
