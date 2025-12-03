"""
Pydantic schemas for IQVIA Campaign Analytics API.
"""
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field
from datetime import date


class PaginationMeta(BaseModel):
    page: int = 1
    per_page: int = 10
    total: int = 0


class StandardResponse(BaseModel):
    status: str = "ok"
    meta: Optional[PaginationMeta] = None
    data: Any = None


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str


class FilterParams(BaseModel):
    specialties: Optional[str] = Field(None, description="Comma-separated list of specialties")
    provinces: Optional[str] = Field(None, description="Comma-separated list of provinces")
    tenure_min: Optional[int] = Field(None, ge=0)
    tenure_max: Optional[int] = Field(None, ge=0)
    min_rx: Optional[float] = Field(None, ge=0)
    max_rx: Optional[float] = Field(None, ge=0)
    campaign_target: Optional[int] = Field(None, ge=0, le=1)
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class CampaignSummary(BaseModel):
    campaign_id: int
    campaign_name: str
    product: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: str
    total_doctors: int


class CampaignMetrics(BaseModel):
    campaign_id: int
    avg_pre_rx: float
    avg_post_rx: float
    avg_uplift_pct: float
    engagement_rate: float
    total_target_doctors: int
    total_reached: int


class RxTrendPoint(BaseModel):
    month: str
    rx_volume: float


class RxTrendResponse(BaseModel):
    campaign_id: int
    trend: List[RxTrendPoint]


class SegmentationRequest(BaseModel):
    doctor_ids: List[int]


class SegmentationResult(BaseModel):
    doctor_id: int
    cluster: int
    cluster_name: str


class UpliftRequest(BaseModel):
    campaign_id: int
    doctor_ids: List[int]


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float


class UpliftResult(BaseModel):
    doctor_id: int
    predicted_uplift_pct: float
    confidence_interval: ConfidenceInterval


class RecommendRequest(BaseModel):
    product: str
    n: int = Field(default=10, ge=1, le=100)
    filters: Optional[FilterParams] = None


class RecommendedDoctor(BaseModel):
    doctor_id: int
    doctor_name: str
    specialty: str
    province: str
    score: float
    predicted_uplift: float


class TopResponder(BaseModel):
    doctor_id: int
    doctor_name: str
    specialty: str
    province: str
    rx_uplift_pct: float
    engagement_score: float


class ExportJobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: Optional[int] = None
    file_url: Optional[str] = None
    error: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    version: str
    loaded: bool
    last_trained: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class ModelsStatus(BaseModel):
    segmentation: ModelInfo
    uplift: ModelInfo
    recommender: ModelInfo


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
