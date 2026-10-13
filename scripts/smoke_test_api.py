"""End-to-end smoke test for a running API (local, Docker or Kubernetes).

Waits for /health, then checks /model-info, a valid /predict, a batch with
missing values and an invalid request (must be rejected with 422).
Exits 0 if every check passes, 1 otherwise. Uses only the standard
library, so it runs anywhere Python does (CI runner, Windows, a pod).

Usage:
    python scripts/smoke_test_api.py
    python scripts/smoke_test_api.py --url http://localhost:8000 --wait 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def call(method: str, url: str, body=None, timeout: float = 10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def wait_for_health(base: str, seconds: int) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            status, _ = call("GET", f"{base}/health", timeout=3)
            if status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--wait", type=int, default=60,
                    help="seconds to wait for /health")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    patient = json.loads((ROOT / "samples" / "sample_patient.json")
                         .read_text())
    batch = json.loads((ROOT / "samples" / "sample_batch.json").read_text())
    failures = 0

    def check(name: str, ok: bool, detail="") -> None:
        nonlocal failures
        failures += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")

    print(f"Smoke-testing {base}")
    up = wait_for_health(base, args.wait)
    check("GET /health -> 200", up)
    if not up:
        print("API never became healthy.")
        return 1

    s, info = call("GET", f"{base}/model-info")
    check("GET /model-info -> 200", s == 200,
          f"model={info.get('model_name')} v{info.get('registered_version')}"
          if s == 200 else str(info))

    s, r = call("POST", f"{base}/predict", patient)
    check("POST /predict (valid) -> 200",
          s == 200 and r.get("label") in ("disease", "no_disease"),
          json.dumps(r))

    s, r = call("POST", f"{base}/predict/batch", {"patients": batch})
    check("POST /predict/batch (3, incl. missing ca/thal) -> 200",
          s == 200 and r.get("n") == len(batch),
          f"risk levels={[x['risk_level'] for x in r.get('results', [])]}")

    s, r = call("POST", f"{base}/predict", {**patient, "sex": 5})
    check("POST /predict (invalid sex=5) -> 422", s == 422)

    print("Smoke test:", "PASSED" if failures == 0 else
          f"FAILED ({failures} check(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
