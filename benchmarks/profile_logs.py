#!/usr/bin/env python3
"""Sample benchmark resources and summarize JSON engine logs using bounded memory."""
import argparse
import collections
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def summarize(source, output):
    stages = collections.defaultdict(lambda: [0, 0, 0, collections.Counter()])
    events, failures = collections.Counter(), collections.Counter()
    pending = set()
    malformed = correlated = 0
    with open(source, errors="replace") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                malformed += 1
                continue
            fields = record.get("fields", {})
            event = fields.get("event")
            if event:
                events[event] += 1
                correlated += any("request_id" in span for span in record.get("spans", []))
            if record.get("level") in ("WARN", "ERROR"):
                failures[(record.get("level"), fields.get("stage", fields.get("message", "unknown")))] += 1
            if event == "stage_enter":
                pending.add(fields["call_id"])
            if event in ("stage_exit", "blocking_queue"):
                pending.discard(fields.get("call_id"))
                name = fields.get("operation", f"blocking_queue:{fields.get('file')}:{fields.get('line')}")
                elapsed = int(fields["elapsed_us"])
                row = stages[name]
                row[0] += 1
                row[1] += elapsed
                row[2] = max(row[2], elapsed)
                row[3][0 if elapsed == 0 else elapsed.bit_length()] += 1

    def quantile(row, fraction):
        target = math.ceil(row[0] * fraction)
        seen = 0
        for bucket, count in sorted(row[3].items()):
            seen += count
            if seen >= target:
                return 0 if bucket == 0 else (1 << bucket) - 1
        return 0

    lines = ["# Engine profiling summary", "",
             "Times are inclusive wall times: nested stages and concurrent workers overlap. Do not add rows together.",
             "Percentiles are upper bounds from power-of-two microsecond buckets. Logging affects measured latency.", "",
             f"Source: `{source}`", f"Non-JSON lines: {malformed}; correlated events: {correlated}; open stage calls: {len(pending)}.", "",
             "| Operation | Calls | Inclusive seconds | Mean ms | p50 upper ms | p95 upper ms | Max ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in sorted(stages.items(), key=lambda item: item[1][1], reverse=True):
        count, total, maximum, _ = row
        lines.append(f"| {name} | {count} | {total / 1e6:.3f} | {total / count / 1e3:.3f} | {quantile(row, .5) / 1e3:.3f} | {quantile(row, .95) / 1e3:.3f} | {maximum / 1e3:.3f} |")
    lines.extend(["", "## Event counts", ""])
    lines.extend(f"- {event}: {count}" for event, count in events.most_common())
    lines.extend(["", "## Warnings and errors", ""])
    lines.extend(f"- {level}: {message}: {count}" for (level, message), count in failures.most_common())
    Path(output).write_text("\n".join(lines) + "\n")


def sample(pid, work_dir, output, interval):
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with open(output, "a", buffering=1) as stream:
        while running and Path(f"/proc/{pid}").exists():
            row = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "pid": pid}
            try:
                # Strip the parenthesized executable name before indexing fields.
                stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                row["cpu_ticks"] = {"user": int(stat[11]), "system": int(stat[12]), "hz": os.sysconf("SC_CLK_TCK")}
            except (OSError, ValueError, IndexError) as error:
                row["cpu_error"] = str(error)
            row["load_average"] = os.getloadavg()
            for name in ("status", "io"):
                try:
                    lines = Path(f"/proc/{pid}/{name}").read_text().splitlines()
                    row[name] = {key: value.strip() for line in lines if ":" in line for key, value in [line.split(":", 1)]}
                except OSError as error:
                    row[name] = {"error": str(error)}
            row["memory"] = {key: value.strip() for key, value in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()) if key in ("MemAvailable", "SwapFree", "SwapTotal", "Dirty", "Writeback")}
            usage = os.statvfs(work_dir)
            row["disk_free_bytes"] = usage.f_bavail * usage.f_frsize
            sizes = collections.Counter()
            for root, _, files in os.walk(work_dir):
                for name in files:
                    try:
                        kind = "wal" if name.endswith("-wal") else "shm" if name.endswith("-shm") else "other"
                        sizes[kind] += (Path(root) / name).stat().st_size
                    except OSError:
                        pass
            row["work_bytes"] = dict(sizes)
            try:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                row["gpu_csv"] = gpu.stdout.strip()
            except (OSError, subprocess.TimeoutExpired) as error:
                row["gpu_error"] = str(error)
            stream.write(json.dumps(row) + "\n")
            # Short sleeps allow prompt cleanup on TERM.
            deadline = time.monotonic() + interval
            while running and time.monotonic() < deadline:
                time.sleep(min(.2, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("source")
    summary.add_argument("output")
    sampler = commands.add_parser("sample")
    sampler.add_argument("--pid", type=int, required=True)
    sampler.add_argument("--work-dir", required=True)
    sampler.add_argument("--output", required=True)
    sampler.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.source, args.output)
    else:
        if args.interval <= 0:
            parser.error("interval must be positive")
        sample(args.pid, args.work_dir, args.output, args.interval)
