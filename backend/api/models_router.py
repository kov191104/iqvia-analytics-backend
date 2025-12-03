"""
ML model prediction routes for IQVIA Campaign Analytics API.
"""
from typing import Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.api.schemas import (
    StandardResponse, SegmentationRequest, SegmentationResult,
    UpliftRequest, UpliftResult, ConfidenceInterval,
    RecommendRequest, RecommendedDoctor, ModelsStatus, ModelInfo
)
from backend.api.utils import apply_filters
from backend.ml.segmentation import segmentation_model
from backend.ml.uplift import uplift_model
from backend.ml.recommender import recommender_model


router = APIRouter(prefix="/api/v1/models", tags=["models"])

doctors_df: Optional[pd.DataFrame] = None
campaigns_df: Optional[pd.DataFrame] = None


def set_data(doctors: pd.DataFrame, campaigns: pd.DataFrame):
    """Set the in-memory DataFrames for model predictions."""
    global doctors_df, campaigns_df
    doctors_df = doctors
    campaigns_df = campaigns


@router.get("/status", response_model=StandardResponse)
async def get_models_status():
    """Get status of all loaded models."""
    import json
    from pathlib import Path
    
    metrics_path = Path("backend/models/metrics.json")
    metrics_data = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics_data = json.load(f)
    
    seg_metrics = metrics_data.get("models", {}).get("segmentation", {}).get("metrics")
    uplift_metrics = metrics_data.get("models", {}).get("uplift", {}).get("metrics")
    rec_metrics = metrics_data.get("models", {}).get("recommender", {}).get("metrics")
    
    status = ModelsStatus(
        segmentation=ModelInfo(
            name="KMeans Segmentation",
            version=segmentation_model.version,
            loaded=segmentation_model.loaded,
            last_trained=metrics_data.get("training_timestamp"),
            metrics=seg_metrics
        ),
        uplift=ModelInfo(
            name="Random Forest Uplift",
            version=uplift_model.version,
            loaded=uplift_model.loaded,
            last_trained=metrics_data.get("training_timestamp"),
            metrics=uplift_metrics
        ),
        recommender=ModelInfo(
            name="Content-Based Recommender",
            version=recommender_model.version,
            loaded=recommender_model.loaded,
            last_trained=metrics_data.get("training_timestamp"),
            metrics=rec_metrics
        )
    )
    
    return StandardResponse(status="ok", data=status)


@router.post("/segmentation/predict", response_model=StandardResponse)
async def predict_segmentation(request: SegmentationRequest):
    """Predict doctor segments using KMeans clustering."""
    if not segmentation_model.loaded:
        raise HTTPException(
            status_code=503,
            detail="Segmentation model not loaded. Please ensure models are trained."
        )
    
    if doctors_df is None or doctors_df.empty:
        raise HTTPException(status_code=404, detail="No doctor data available")
    
    filtered_doctors = doctors_df[doctors_df["doctor_id"].isin(request.doctor_ids)]
    
    if filtered_doctors.empty:
        return StandardResponse(
            status="ok",
            data=[]
        )
    
    try:
        predictions = segmentation_model.predict(filtered_doctors)
        results = [SegmentationResult(**p) for p in predictions]
        
        return StandardResponse(
            status="ok",
            data=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/uplift/predict", response_model=StandardResponse)
async def predict_uplift(request: UpliftRequest):
    """Predict uplift percentage for doctors in a campaign."""
    if not uplift_model.loaded:
        raise HTTPException(
            status_code=503,
            detail="Uplift model not loaded. Please ensure models are trained."
        )
    
    if campaigns_df is None or campaigns_df.empty:
        raise HTTPException(status_code=404, detail="No campaign data available")
    
    campaign_doctors = campaigns_df[
        (campaigns_df["campaign_id"] == request.campaign_id) &
        (campaigns_df["doctor_id"].isin(request.doctor_ids))
    ]
    
    if campaign_doctors.empty:
        if doctors_df is not None:
            doctor_data = doctors_df[doctors_df["doctor_id"].isin(request.doctor_ids)]
            if not doctor_data.empty:
                campaign_doctors = doctor_data.copy()
                campaign_doctors["campaign_id"] = request.campaign_id
                campaign_doctors["calls_in_campaign"] = 0
                campaign_doctors["engagement_in_campaign"] = 0.0
    
    if campaign_doctors.empty:
        return StandardResponse(status="ok", data=[])
    
    try:
        predictions = uplift_model.predict_with_confidence(campaign_doctors)
        results = [
            UpliftResult(
                doctor_id=p["doctor_id"],
                predicted_uplift_pct=p["predicted_uplift_pct"],
                confidence_interval=ConfidenceInterval(**p["confidence_interval"])
            )
            for p in predictions
        ]
        
        return StandardResponse(status="ok", data=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/recommend/doctors", response_model=StandardResponse)
async def recommend_doctors(request: RecommendRequest):
    """Recommend top doctors for a product."""
    if not recommender_model.loaded:
        raise HTTPException(
            status_code=503,
            detail="Recommender model not loaded. Please ensure models are trained."
        )
    
    if doctors_df is None or doctors_df.empty:
        raise HTTPException(status_code=404, detail="No doctor data available")
    
    filtered_doctors = doctors_df.copy()
    
    if "doctor_id" in filtered_doctors.columns:
        filtered_doctors = filtered_doctors.drop_duplicates(subset=["doctor_id"])
    
    if request.filters:
        filtered_doctors = apply_filters(
            filtered_doctors,
            specialties=request.filters.specialties,
            provinces=request.filters.provinces,
            tenure_min=request.filters.tenure_min,
            tenure_max=request.filters.tenure_max,
            min_rx=request.filters.min_rx,
            max_rx=request.filters.max_rx,
            campaign_target=request.filters.campaign_target
        )
    
    if filtered_doctors.empty:
        return StandardResponse(status="ok", data=[])
    
    try:
        uplift_preds = {}
        if uplift_model.loaded:
            pred_df = filtered_doctors.copy()
            pred_df["campaign_id"] = 0
            pred_df["calls_in_campaign"] = 0
            pred_df["engagement_in_campaign"] = 0.0
            predictions = uplift_model.predict_with_confidence(pred_df)
            uplift_preds = {p["doctor_id"]: p["predicted_uplift_pct"] for p in predictions}
        
        recommendations = recommender_model.recommend(
            product=request.product,
            n=request.n,
            doctor_df=filtered_doctors,
            uplift_predictions=uplift_preds
        )
        
        seen_ids = set()
        unique_recommendations = []
        for r in recommendations:
            if r["doctor_id"] not in seen_ids:
                seen_ids.add(r["doctor_id"])
                unique_recommendations.append(r)
        
        results = [RecommendedDoctor(**r) for r in unique_recommendations]
        
        return StandardResponse(status="ok", data=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")
