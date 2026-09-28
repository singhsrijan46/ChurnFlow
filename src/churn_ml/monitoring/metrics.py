from prometheus_client import Counter, Gauge, Histogram

# HTTP Request Metrics
REQUEST_COUNT = Counter(
    "churn_http_requests_total",
    "Total count of HTTP requests received",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "churn_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Inference Metrics
PREDICTION_COUNT = Counter(
    "churn_predictions_total",
    "Total customer churn predictions executed",
    ["model_version", "prediction"],
)

PREDICTION_DISTRIBUTION = Histogram(
    "churn_prediction_probability",
    "Distribution of churn probabilities calculated by model",
    ["model_version"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Data Drift & Health Metrics
DATA_DRIFT_SCORE = Gauge(
    "churn_dataset_drift_score",
    "Overall dataset drift share score calculated by drift engine",
)

DATA_DRIFT_STATUS = Gauge(
    "churn_dataset_drift_detected",
    "Boolean status indicating whether dataset drift threshold was exceeded (1 = drifted, 0 = healthy)",
)

ERROR_COUNT = Counter(
    "churn_prediction_errors_total",
    "Total count of unhandled or validation errors during prediction",
    ["error_type"],
)
