"""
IQVIA Campaign Analytics API - Main Application.
"""
import os
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from backend.api.routes import router as campaigns_router, set_data as set_campaigns_data
from backend.api.models_router import router as models_router, set_data as set_models_data
from backend.api.schemas import StandardResponse, HealthResponse, ErrorResponse
from backend.ml.train_models import load_data, train_all_models, load_all_models, generate_sample_data
from backend.ml.segmentation import segmentation_model
from backend.ml.uplift import uplift_model
from backend.ml.recommender import recommender_model


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("iqvia-api")

START_TIME = time.time()
APP_VERSION = "1.0.0"


campaigns_df: pd.DataFrame = pd.DataFrame()
doctors_df: pd.DataFrame = pd.DataFrame()


def initialize_data():
    """Load data into memory at startup."""
    global campaigns_df, doctors_df
    
    data_dir = Path("backend/data")
    models_dir = Path("backend/models")
    
    try:
        data = load_data()
        
        if "full" in data:
            campaigns_df = data["full"]
        elif "train_val_full" in data:
            campaigns_df = data["train_val_full"]
        
        if "train_doctors" in data:
            doctors_df = data["train_doctors"]
        elif not campaigns_df.empty:
            doctor_cols = ["doctor_id", "doctor_name", "specialty", "province", "tenure",
                          "avg_rx_6mo", "avg_rx_12mo", "rx_total_12mo"]
            available_cols = [c for c in doctor_cols if c in campaigns_df.columns]
            doctors_df = campaigns_df[available_cols].drop_duplicates(subset=["doctor_id"])
        
        logger.info(f"Loaded {len(campaigns_df)} campaign records and {len(doctors_df)} doctors")
        
        set_campaigns_data(campaigns_df, doctors_df)
        set_models_data(doctors_df, campaigns_df)
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.info("Generating sample data...")
        data = generate_sample_data()
        campaigns_df = data["full"]
        doctors_df = data["train_doctors"]
        set_campaigns_data(campaigns_df, doctors_df)
        set_models_data(doctors_df, campaigns_df)


def initialize_models():
    """Load or train ML models at startup."""
    models_dir = Path("backend/models")
    
    load_status = load_all_models()
    logger.info(f"Model load status: {load_status}")
    
    all_loaded = all(load_status.values())
    
    if not all_loaded:
        skip_training = os.getenv("SKIP_STARTUP_TRAINING", "false").lower() == "true"
        
        if skip_training:
            logger.warning("Models not loaded and training is skipped. Some endpoints may not work.")
        else:
            logger.info("Training models...")
            try:
                data = load_data()
                train_all_models(data)
                load_all_models()
                logger.info("Models trained and loaded successfully")
            except Exception as e:
                logger.error(f"Error training models: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting IQVIA Campaign Analytics API...")
    
    initialize_data()
    initialize_models()
    
    logger.info("Application startup complete")
    
    yield
    
    logger.info("Shutting down IQVIA Campaign Analytics API...")


app = FastAPI(
    title="IQVIA Campaign Analytics API",
    description="API for campaign analytics, doctor segmentation, uplift prediction, and recommendations",
    version=APP_VERSION,
    lifespan=lifespan
)


allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_cache_control(request: Request, call_next):
    """Add cache control headers to prevent caching issues."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": str(exc)}
    )


app.include_router(campaigns_router)
app.include_router(models_router)


@app.get("/", response_model=StandardResponse)
async def root():
    """Root endpoint with API information."""
    return StandardResponse(
        status="ok",
        data={
            "name": "IQVIA Campaign Analytics API",
            "version": APP_VERSION,
            "docs": "/docs",
            "health": "/api/v1/health"
        }
    )


@app.get("/api/v1/health", response_model=StandardResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - START_TIME
    
    return StandardResponse(
        status="ok",
        data=HealthResponse(
            status="healthy",
            version=APP_VERSION,
            uptime_seconds=round(uptime, 2)
        )
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
