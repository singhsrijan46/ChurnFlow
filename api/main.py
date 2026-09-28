import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.routes import health, prediction
from src.churn_ml.config import settings
from src.churn_ml.inference.predictor import get_predictor
from src.churn_ml.logging_config import logger
from src.churn_ml.monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for application startup and shutdown."""
    logger.info(f"Initializing {settings.app.name} API v{settings.app.version} ({settings.app.environment})...")
    # Pre-warm model predictor
    try:
        predictor = get_predictor()
        if predictor.is_ready():
            logger.info("Champion model preloaded successfully into inference memory.")
        else:
            logger.warning("Inference predictor started without loaded model. Will await first training run.")
    except Exception as e:
        logger.warning(f"Initial model loading deferred: {e}")
    yield
    logger.info("Shutting down ChurnFlow API...")


app = FastAPI(
    title=f"{settings.app.name} - Enterprise Customer Churn MLOps Platform",
    description=(
        "Production-grade REST API for Customer Churn Prediction, Model Governance, "
        "Prometheus Telemetry, Data Drift Detection, and Continuous Automated Retraining."
    ),
    version=settings.app.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    """
    Middleware that records request count, latency histogram, and status codes for Prometheus.
    """
    start_time = time.perf_counter()
    response = Response("Internal Server Error", status_code=500)
    try:
        response = await call_next(request)
    finally:
        latency = time.perf_counter() - start_time
        endpoint = request.url.path

        # Ignore metrics scraping endpoint itself from telemetry pollution
        if endpoint != "/metrics":
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=response.status_code,
            ).inc()
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

    return response


@app.get("/metrics", summary="Prometheus Metrics", tags=["Monitoring & Telemetry"])
async def metrics():
    """Exposes application and model metrics in Prometheus text exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Mount Route Handlers
app.include_router(health.router)
app.include_router(prediction.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
