import uuid
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftCheckResponse,
    RetrainingResponse,
)
from api.dependencies import get_model_predictor, get_settings
from src.churn_ml.config import Settings
from src.churn_ml.inference.predictor import ChurnPredictor, ModelNotReadyError
from src.churn_ml.logging_config import logger
from src.churn_ml.monitoring.metrics import (
    PREDICTION_COUNT,
    PREDICTION_DISTRIBUTION,
    ERROR_COUNT,
)
from src.churn_ml.monitoring.drift import detect_data_drift
from src.churn_ml.training.train import train_and_compare_models
from src.churn_ml.data.ingestion import ingest_data, save_processed_splits

router = APIRouter(tags=["Prediction & Operations"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Customer Churn (Single)",
    description="Calculates churn probability and classification (0/1) for a single customer payload.",
)
async def predict_single_customer(
    payload: CustomerFeatures,
    predictor: ChurnPredictor = Depends(get_model_predictor),
) -> PredictionResponse:
    request_id = str(uuid.uuid4())
    try:
        # Convert pydantic model to dictionary with original column names
        data_dict = payload.model_dump(by_alias=True)
        res = predictor.predict_single(data_dict)

        # Update Prometheus metrics
        PREDICTION_COUNT.labels(
            model_version=res["model_version"],
            prediction=str(res["prediction"]),
        ).inc()

        PREDICTION_DISTRIBUTION.labels(
            model_version=res["model_version"],
        ).observe(res["churn_probability"])

        return PredictionResponse(
            prediction=res["prediction"],
            churn_probability=res["churn_probability"],
            risk_category=res["risk_category"],
            model_version=res["model_version"],
            request_id=request_id,
        )
    except ModelNotReadyError as e:
        ERROR_COUNT.labels(error_type="model_not_ready").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        ERROR_COUNT.labels(error_type="prediction_error").inc()
        logger.error(f"Prediction failed for request {request_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict Customer Churn (Batch)",
    description="Calculates churn probabilities and classifications for a batch of up to 1000 customers.",
)
async def predict_batch_customers(
    payload: BatchPredictionRequest,
    predictor: ChurnPredictor = Depends(get_model_predictor),
) -> BatchPredictionResponse:
    try:
        data_dicts = [item.model_dump(by_alias=True) for item in payload.customers]
        results = predictor.predict_batch(data_dicts)

        predictions_resp = []
        for r in results:
            PREDICTION_COUNT.labels(
                model_version=r["model_version"],
                prediction=str(r["prediction"]),
            ).inc()
            PREDICTION_DISTRIBUTION.labels(
                model_version=r["model_version"],
            ).observe(r["churn_probability"])

            predictions_resp.append(
                PredictionResponse(
                    prediction=r["prediction"],
                    churn_probability=r["churn_probability"],
                    risk_category=r["risk_category"],
                    model_version=r["model_version"],
                    request_id=str(uuid.uuid4()),
                )
            )

        return BatchPredictionResponse(
            predictions=predictions_resp,
            total_count=len(predictions_resp),
            model_version=predictor.model_version,
        )
    except ModelNotReadyError as e:
        ERROR_COUNT.labels(error_type="model_not_ready").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        ERROR_COUNT.labels(error_type="batch_prediction_error").inc()
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}",
        )


@router.post(
    "/monitoring/drift",
    response_model=DriftCheckResponse,
    summary="Trigger Data Drift Check",
    description="Compares reference baseline data with current production data to detect distribution drift.",
)
async def check_drift(
    app_settings: Settings = Depends(get_settings),
) -> DriftCheckResponse:
    ref_path = app_settings.data.reference_data_path
    curr_path = app_settings.data.current_data_path

    try:
        ref_df = pd.read_csv(ref_path)
        curr_df = pd.read_csv(curr_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference or Current datasets not found ({e}). Run ingestion pipeline first.",
        )

    drift_report = detect_data_drift(
        reference_df=ref_df,
        current_df=curr_df,
        drift_threshold=app_settings.monitoring.drift_share_threshold,
    )

    return DriftCheckResponse(
        drift_detected=drift_report.drift_detected,
        drift_share=drift_report.drift_share,
        threshold=drift_report.threshold,
        number_of_features=drift_report.number_of_features,
        number_of_drifted_features=drift_report.number_of_drifted_features,
        drifted_features=drift_report.drifted_features,
        report_html_path=drift_report.report_html_path,
        retraining_recommended=drift_report.drift_detected,
    )


@router.post(
    "/pipeline/retrain",
    response_model=RetrainingResponse,
    summary="Trigger Automated Retraining Pipeline",
    description="Executes data ingestion, training of models, evaluation, validation, and conditional promotion.",
)
async def trigger_retraining(
    predictor: ChurnPredictor = Depends(get_model_predictor),
    app_settings: Settings = Depends(get_settings),
) -> RetrainingResponse:
    try:
        logger.info("Retraining pipeline triggered via API.")
        df_clean = ingest_data(clean=True, validate=True)
        train_df, test_df = save_processed_splits(df_clean)

        train_results = train_and_compare_models(train_df, test_df, register_best=True)

        # Reload the active predictor instance with the newly trained champion model
        predictor._load_model()

        return RetrainingResponse(
            status="completed",
            champion_model=train_results["best_model_type"],
            best_metrics=train_results["best_metrics"].to_dict(),
            run_id=train_results["run_id"],
            is_promoted=train_results["is_eligible_for_promotion"],
            promotion_reason=train_results["promotion_reason"],
        )
    except Exception as e:
        logger.error(f"Retraining pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining error: {str(e)}",
        )
