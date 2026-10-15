"""Verify the monitoring stack end to end (Prometheus + Grafana).

Checks that:
  * Prometheus is ready and scraping the heart-api pods (all targets up)
  * the heart-api metrics are present, and the alert rules are loaded
  * Grafana is healthy, its Prometheus data source works and the
    heart-api dashboard is provisioned

Prints a short summary of live values (pods, request rate, predictions,
active alerts). Exits 0 if every check passes, 1 otherwise. Standard library
only, so it runs in CI and on Windows alike.

Usage:
    python scripts/check_monitoring.py
    python scripts/check_monitoring.py --prometheus http://localhost:9090 \
        --grafana http://localhost:3000 --min-targets 2 --wait 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def get_json(url: str, timeout: float = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def query(prom: str, expr: str) -> list:
    url = f"{prom}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    return get_json(url)["data"]["result"]


def scalar(prom: str, expr: str) -> float:
    res = query(prom, expr)
    return float(res[0]["value"][1]) if res else 0.0


def wait_until(fn, seconds: int) -> bool:
    deadline = time.time() + seconds
    while True:
        try:
            if fn():
                return True
        except (urllib.error.URLError, OSError, KeyError, ValueError):
            pass
        if time.time() > deadline:
            return False
        time.sleep(3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prometheus", default="http://localhost:9090")
    ap.add_argument("--grafana", default="http://localhost:3000")
    ap.add_argument("--min-targets", type=int, default=2,
                    help="heart-api pods that must be up")
    ap.add_argument("--wait", type=int, default=90,
                    help="seconds to wait for targets to come up")
    args = ap.parse_args()
    prom, graf = args.prometheus.rstrip("/"), args.grafana.rstrip("/")
    failures = 0

    def check(name: str, ok: bool, detail="") -> None:
        nonlocal failures
        failures += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")

    print(f"Prometheus {prom}")
    ready = wait_until(lambda: urllib.request.urlopen(
        f"{prom}/-/ready", timeout=5).status == 200, args.wait)
    check("Prometheus ready", ready)
    if not ready:
        return 1

    def targets_up() -> bool:
        return scalar(prom, 'count(up{job="heart-api"} == 1)') \
            >= args.min_targets
    wait_until(targets_up, args.wait)
    targets = get_json(f"{prom}/api/v1/targets?state=active")["data"]
    api = [t for t in targets["activeTargets"]
           if t["labels"].get("job") == "heart-api"]
    up = [t for t in api if t["health"] == "up"]
    check(f"heart-api pods scraped (>= {args.min_targets} up)",
          len(up) >= args.min_targets,
          f"{len(up)}/{len(api)} up: "
          + ", ".join(t["labels"].get("pod", "?") for t in up))

    names = set(get_json(f"{prom}/api/v1/label/__name__/values")["data"])
    wanted = {"heart_api_requests_total", "heart_api_predictions_total",
              "heart_api_request_duration_seconds_bucket",
              "heart_api_model_loaded"}
    check("heart-api metrics present", wanted <= names,
          f"missing: {sorted(wanted - names)}" if wanted - names else "")

    groups = get_json(f"{prom}/api/v1/rules")["data"]["groups"]
    rules = [r for g in groups for r in g["rules"]]
    unhealthy = [r["name"] for r in rules if r.get("health") != "ok"]
    check("alert rules loaded and healthy", len(rules) >= 7 and not unhealthy,
          f"{len(rules)} rules" + (f", unhealthy: {unhealthy}"
                                   if unhealthy else ""))

    print(f"Grafana {graf}")
    healthy = wait_until(lambda: get_json(f"{graf}/api/health")
                         .get("database") == "ok", args.wait)
    check("Grafana healthy", healthy)
    if healthy:
        try:
            dash = get_json(f"{graf}/api/dashboards/uid/heart-api")
            n = len([p for p in dash["dashboard"]["panels"]
                     if p["type"] != "row"])
            check("dashboard provisioned", n > 0,
                  f"'{dash['dashboard']['title']}', {n} panels")
        except urllib.error.HTTPError as e:
            check("dashboard provisioned", False, f"HTTP {e.code}")
        try:
            body = json.dumps({"queries": [{
                "refId": "A", "expr": 'up{job="heart-api"}',
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "instant": True}], "from": "now-5m", "to": "now"}).encode()
            req = urllib.request.Request(
                f"{graf}/api/ds/query", data=body, method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                frames = json.loads(r.read())["results"]["A"]["frames"]
            check("Grafana -> Prometheus data source query", len(frames) > 0,
                  f"{len(frames)} series")
        except (urllib.error.URLError, KeyError) as e:
            check("Grafana -> Prometheus data source query", False, str(e))

    print("Live values")
    live = {
        "pods up": 'count(up{job="heart-api"} == 1)',
        "prediction req/s":
            'sum(rate(heart_api_requests_total{endpoint=~"/predict.*"}[1m]))',
        "patients scored": "sum(heart_api_predictions_total)",
    }
    for label, expr in live.items():
        print(f"  {label:<20} {round(scalar(prom, expr), 2):g}")
    firing = query(prom, "ALERTS")
    print("  active alerts        " + (", ".join(
        f"{a['metric']['alertname']} ({a['metric']['alertstate']})"
        for a in firing) or "none"))

    print("Monitoring check:", "PASSED" if failures == 0 else
          f"FAILED ({failures} check(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
