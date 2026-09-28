from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas import HealthResponse, ModelInfoResponse, ReadinessResponse
from api.dependencies import get_model_predictor, get_settings
from src.churn_ml.config import Settings
from src.churn_ml.inference.predictor import ChurnPredictor

router = APIRouter(tags=["Health & System"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Probe",
    description="Returns service liveness status for Kubernetes and orchestrator health checks.",
)
async def health(app_settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="healthy",
        app_name=app_settings.app.name,
        version=app_settings.app.version,
        environment=app_settings.app.environment,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness Probe",
    description="Returns readiness status indicating whether the ML model is loaded into memory.",
)
async def ready(
    predictor: ChurnPredictor = Depends(get_model_predictor),
    app_settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Prediction artifact not loaded.",
        )
    return ReadinessResponse(
        status="ready",
        model_loaded=True,
        model_name=app_settings.registry.model_name,
        model_version=predictor.model_version,
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Metadata",
    description="Returns metadata, required features, and version information for active ML model.",
)
async def model_info(
    predictor: ChurnPredictor = Depends(get_model_predictor),
) -> ModelInfoResponse:
    info = predictor.get_model_info()
    return ModelInfoResponse(**info)
