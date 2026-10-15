"""Small concurrent load generator for the deployed API.

Sends POST /predict requests from several threads and reports throughput,
latency percentiles, error count and - via the X-Served-By header - how
many requests each pod answered. That shows the Kubernetes Service load
balancing across replicas, and a longer run generates enough CPU for the
HorizontalPodAutoscaler to add pods.

Uses only the standard library.

Usage:
    python scripts/load_test.py                         # 200 requests, 8 threads
    python scripts/load_test.py -n 2000 -c 16           # heavier
    python scripts/load_test.py --duration 180 -c 16    # sustained (HPA demo)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def one_request(url: str, body: bytes) -> tuple[int, float, str]:
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Connection": "close"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
            status, pod = r.status, r.headers.get("X-Served-By", "?")
    except urllib.error.HTTPError as e:
        status, pod = e.code, e.headers.get("X-Served-By", "?")
    except (urllib.error.URLError, OSError):
        status, pod = 0, "unreachable"
    return status, (time.perf_counter() - start) * 1000, pod


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("-n", "--requests", type=int, default=200)
    ap.add_argument("-c", "--concurrency", type=int, default=8)
    ap.add_argument("--duration", type=int, default=0,
                    help="run for N seconds instead of a fixed count")
    args = ap.parse_args()

    url = args.url.rstrip("/") + "/predict"
    body = (ROOT / "samples" / "sample_patient.json").read_bytes()
    json.loads(body)  # fail early if the sample is broken

    lock = threading.Lock()
    latencies: list[float] = []
    statuses: Counter = Counter()
    pods: Counter = Counter()
    deadline = time.time() + args.duration if args.duration else None
    issued = [0]

    def take() -> bool:
        with lock:
            if deadline is not None:
                return time.time() < deadline
            if issued[0] >= args.requests:
                return False
            issued[0] += 1
            return True

    def worker() -> None:
        while take():
            status, ms, pod = one_request(url, body)
            if status == 0:          # server unreachable: don't spin
                time.sleep(0.5)
            with lock:
                statuses[status] += 1
                pods[pod] += 1
                if status == 200:
                    latencies.append(ms)

    mode = (f"{args.duration}s" if deadline else f"{args.requests} requests")
    print(f"Load test: {mode}, {args.concurrency} threads -> {url}")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        for _ in range(args.concurrency):
            pool.submit(worker)
    elapsed = time.perf_counter() - t0

    total = sum(statuses.values())
    ok = statuses.get(200, 0)
    print(f"\nTotal {total} requests in {elapsed:.1f}s "
          f"-> {total / elapsed:.1f} req/s")
    print(f"Status codes: {dict(statuses)}")
    if latencies:
        print(f"Latency ms: mean {statistics.mean(latencies):.1f}  "
              f"p50 {pct(latencies, 50):.1f}  p95 {pct(latencies, 95):.1f}  "
              f"p99 {pct(latencies, 99):.1f}  max {max(latencies):.1f}")
    print("\nRequests per pod (X-Served-By):")
    for pod, n in pods.most_common():
        print(f"  {pod:<40} {n:>6}  ({100 * n / total:.0f}%)")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
