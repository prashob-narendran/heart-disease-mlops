"""Prometheus metrics for the prediction API.

Exposed at GET /metrics in the Prometheus text format and scraped by the
Prometheus server deployed in k8s/monitoring/. Two groups:

* service metrics: request count, latency histogram and requests in flight,
  labelled by endpoint (the route template, so the label set stays small)
* model metrics: predictions by class and risk band, the distribution of the
  predicted disease probability (a simple drift signal), and which model
  version is loaded

Metric names follow the Prometheus conventions (snake_case, unit suffix,
`_total` for counters).
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    disable_created_metrics,
    generate_latest,
)

# Drop the extra *_created timestamp series (not used by Prometheus queries
# or the dashboard) to keep /metrics small and readable.
disable_created_metrics()

PREFIX = "heart_api"

REQUESTS = Counter(
    f"{PREFIX}_requests_total",
    "HTTP requests handled, by method, endpoint and status code.",
    ["method", "endpoint", "status"],
)
LATENCY = Histogram(
    f"{PREFIX}_request_duration_seconds",
    "HTTP request latency in seconds, by method and endpoint.",
    ["method", "endpoint"],
    # SVM inference takes ~5-50 ms; buckets cover 2.5 ms to 5 s
    buckets=(0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
IN_FLIGHT = Gauge(
    f"{PREFIX}_requests_in_flight",
    "Requests currently being processed.",
)
PREDICTIONS = Counter(
    f"{PREFIX}_predictions_total",
    "Patients scored, by predicted label and risk level.",
    ["label", "risk_level"],
)
PROBABILITY = Histogram(
    f"{PREFIX}_prediction_probability",
    "Predicted probability of heart disease (distribution over patients).",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
BATCH_SIZE = Histogram(
    f"{PREFIX}_batch_size",
    "Number of patients per /predict/batch request.",
    buckets=(1, 2, 5, 10, 20, 50, 100),
)
MODEL_LOADED = Gauge(
    f"{PREFIX}_model_loaded",
    "1 if the model is loaded and the service can predict, else 0.",
)
MODEL_INFO = Gauge(
    f"{PREFIX}_model_info",
    "Always 1; labels identify the model being served.",
    ["model_name", "model_version"],
)


def endpoint_label(request) -> str:
    """Route template (e.g. '/predict') instead of the raw URL, so unknown
    paths cannot create unlimited label values."""
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


def observe_request(method: str, endpoint: str, status: int,
                    seconds: float) -> None:
    REQUESTS.labels(method, endpoint, str(status)).inc()
    LATENCY.labels(method, endpoint).observe(seconds)


def observe_predictions(results: list[dict]) -> None:
    for r in results:
        PREDICTIONS.labels(r["label"], r["risk_level"]).inc()
        PROBABILITY.observe(r["disease_probability"])


def set_model_state(loaded: bool, metadata: dict) -> None:
    MODEL_LOADED.set(1 if loaded else 0)
    MODEL_INFO.clear()
    if loaded:
        MODEL_INFO.labels(str(metadata.get("model_name", "unknown")),
                          str(metadata.get("registered_version", ""))).set(1)


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
