"""
Campaign data routes for IQVIA Campaign Analytics API.
"""
import io
import uuid
import asyncio
from datetime import datetime
from typing import Optional
from pathlib import Path

import aiofiles
import pandas as pd
from fastapi import APIRouter, Query, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from backend.api.schemas import (
    StandardResponse, PaginationMeta, CampaignSummary,
    CampaignMetrics, RxTrendResponse, RxTrendPoint,
    TopResponder, ExportJobStatus
)
from backend.api.utils import apply_filters, paginate, calculate_campaign_metrics, calculate_rx_trend


router = APIRouter(prefix="/api/v1", tags=["campaigns"])

campaigns_df: Optional[pd.DataFrame] = None
doctors_df: Optional[pd.DataFrame] = None

export_jobs: dict = {}
EXPORTS_DIR = Path("backend/exports")
MAX_STREAMING_ROWS = 50000


def set_data(campaigns: pd.DataFrame, doctors: pd.DataFrame):
    """Set the in-memory DataFrames."""
    global campaigns_df, doctors_df
    campaigns_df = campaigns
    doctors_df = doctors


@router.get("/campaigns", response_model=StandardResponse)
async def list_campaigns(
    specialties: Optional[str] = Query(None, description="Comma-separated specialties"),
    provinces: Optional[str] = Query(None, description="Comma-separated provinces"),
    tenure_min: Optional[int] = Query(None, ge=0),
    tenure_max: Optional[int] = Query(None, ge=0),
    min_rx: Optional[float] = Query(None, ge=0),
    max_rx: Optional[float] = Query(None, ge=0),
    campaign_target: Optional[int] = Query(None, ge=0, le=1),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=500)
):
    """List all campaigns with optional filtering."""
    if campaigns_df is None or campaigns_df.empty:
        return StandardResponse(
            status="ok",
            meta=PaginationMeta(page=page, per_page=per_page, total=0),
            data=[]
        )
    
    filtered = apply_filters(
        campaigns_df,
        specialties=specialties,
        provinces=provinces,
        tenure_min=tenure_min,
        tenure_max=tenure_max,
        min_rx=min_rx,
        max_rx=max_rx,
        campaign_target=campaign_target
    )
    
    campaign_summary = []
    for campaign_id in filtered["campaign_id"].unique():
        campaign_data = filtered[filtered["campaign_id"] == campaign_id].iloc[0]
        campaign_summary.append(CampaignSummary(
            campaign_id=int(campaign_id),
            campaign_name=campaign_data.get("campaign_name", f"Campaign {campaign_id}"),
            product=campaign_data.get("product", "Unknown"),
            start_date=None,
            end_date=None,
            status="active",
            total_doctors=int(len(filtered[filtered["campaign_id"] == campaign_id]["doctor_id"].unique()))
        ))
    
    summary_df = pd.DataFrame([c.model_dump() for c in campaign_summary])
    paginated, pg, pp, total = paginate(summary_df, page, per_page)
    
    return StandardResponse(
        status="ok",
        meta=PaginationMeta(page=pg, per_page=pp, total=total),
        data=[CampaignSummary(**row) for _, row in paginated.iterrows()]
    )


@router.get("/campaign/{campaign_id}/metrics", response_model=StandardResponse)
async def get_campaign_metrics(
    campaign_id: int,
    specialties: Optional[str] = Query(None),
    provinces: Optional[str] = Query(None),
    tenure_min: Optional[int] = Query(None, ge=0),
    tenure_max: Optional[int] = Query(None, ge=0),
    min_rx: Optional[float] = Query(None, ge=0),
    max_rx: Optional[float] = Query(None, ge=0),
    campaign_target: Optional[int] = Query(None, ge=0, le=1)
):
    """Get metrics for a specific campaign."""
    if campaigns_df is None or campaigns_df.empty:
        raise HTTPException(status_code=404, detail="No campaign data available")
    
    filtered = apply_filters(
        campaigns_df[campaigns_df["campaign_id"] == campaign_id],
        specialties=specialties,
        provinces=provinces,
        tenure_min=tenure_min,
        tenure_max=tenure_max,
        min_rx=min_rx,
        max_rx=max_rx,
        campaign_target=campaign_target
    )
    
    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")
    
    metrics = calculate_campaign_metrics(filtered, campaign_id)
    
    return StandardResponse(
        status="ok",
        data=CampaignMetrics(**metrics)
    )


@router.get("/campaign/{campaign_id}/rx-trend", response_model=StandardResponse)
async def get_rx_trend(
    campaign_id: int,
    months: int = Query(12, ge=1, le=24),
    specialties: Optional[str] = Query(None),
    provinces: Optional[str] = Query(None)
):
    """Get Rx volume trend for a campaign."""
    if campaigns_df is None or campaigns_df.empty:
        raise HTTPException(status_code=404, detail="No campaign data available")
    
    filtered = apply_filters(
        campaigns_df[campaigns_df["campaign_id"] == campaign_id],
        specialties=specialties,
        provinces=provinces
    )
    
    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")
    
    trend_data = calculate_rx_trend(filtered, campaign_id, months)
    
    return StandardResponse(
        status="ok",
        data=RxTrendResponse(
            campaign_id=campaign_id,
            trend=[RxTrendPoint(**t) for t in trend_data]
        )
    )


async def generate_csv_export(job_id: str, df: pd.DataFrame, filename: str):
    """Background task to generate CSV export."""
    try:
        export_jobs[job_id]["status"] = "processing"
        
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = EXPORTS_DIR / filename
        
        async with aiofiles.open(file_path, mode='w') as f:
            await f.write(df.to_csv(index=False))
        
        export_jobs[job_id]["status"] = "completed"
        export_jobs[job_id]["file_url"] = f"/api/v1/exports/{job_id}/download"
        export_jobs[job_id]["progress"] = 100
        
    except Exception as e:
        export_jobs[job_id]["status"] = "failed"
        export_jobs[job_id]["error"] = str(e)


def generate_csv_stream(df: pd.DataFrame):
    """Generator for streaming CSV response."""
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    while True:
        chunk = output.read(8192)
        if not chunk:
            break
        yield chunk


@router.get("/campaign/{campaign_id}/top-responders")
async def get_top_responders(
    campaign_id: int,
    background_tasks: BackgroundTasks,
    limit: int = Query(10, ge=1, le=1000),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=500),
    export: Optional[str] = Query(None, description="Set to 'csv' for CSV export"),
    specialties: Optional[str] = Query(None),
    provinces: Optional[str] = Query(None),
    tenure_min: Optional[int] = Query(None, ge=0),
    tenure_max: Optional[int] = Query(None, ge=0)
):
    """Get top responders for a campaign with optional CSV export."""
    if campaigns_df is None or campaigns_df.empty:
        raise HTTPException(status_code=404, detail="No campaign data available")
    
    filtered = apply_filters(
        campaigns_df[campaigns_df["campaign_id"] == campaign_id],
        specialties=specialties,
        provinces=provinces,
        tenure_min=tenure_min,
        tenure_max=tenure_max
    )
    
    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")
    
    sorted_df = filtered.sort_values("rx_uplift_pct", ascending=False)
    
    if export == "csv":
        export_df = sorted_df.head(limit) if limit else sorted_df
        
        if len(export_df) > MAX_STREAMING_ROWS:
            job_id = str(uuid.uuid4())
            filename = f"top_responders_{campaign_id}_{job_id}.csv"
            
            export_jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "progress": 0,
                "file_url": None,
                "error": None
            }
            
            background_tasks.add_task(generate_csv_export, job_id, export_df, filename)
            
            return StandardResponse(
                status="accepted",
                data=ExportJobStatus(**export_jobs[job_id])
            )
        
        return StreamingResponse(
            generate_csv_stream(export_df),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=top_responders_{campaign_id}.csv"
            }
        )
    
    top_df = sorted_df.head(limit)
    paginated, pg, pp, total = paginate(top_df, page, per_page)
    
    responders = []
    for _, row in paginated.iterrows():
        responders.append(TopResponder(
            doctor_id=int(row.get("doctor_id", 0)),
            doctor_name=row.get("doctor_name", "Unknown"),
            specialty=row.get("specialty", "Unknown"),
            province=row.get("province", "Unknown"),
            rx_uplift_pct=float(row.get("rx_uplift_pct", 0)),
            engagement_score=float(row.get("engagement_in_campaign", 0))
        ))
    
    return StandardResponse(
        status="ok",
        meta=PaginationMeta(page=pg, per_page=pp, total=min(total, limit)),
        data=responders
    )


@router.get("/exports/{job_id}", response_model=StandardResponse)
async def get_export_status(job_id: str):
    """Check status of an export job."""
    if job_id not in export_jobs:
        raise HTTPException(status_code=404, detail="Export job not found")
    
    return StandardResponse(
        status="ok",
        data=ExportJobStatus(**export_jobs[job_id])
    )


@router.get("/exports/{job_id}/download")
async def download_export(job_id: str):
    """Download a completed export file."""
    if job_id not in export_jobs:
        raise HTTPException(status_code=404, detail="Export job not found")
    
    job = export_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Export not ready")
    
    file_path = EXPORTS_DIR / f"top_responders_{job_id.split('_')[-1]}.csv"
    
    for f in EXPORTS_DIR.glob(f"*{job_id}*.csv"):
        file_path = f
        break
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    
    async def file_stream():
        async with aiofiles.open(file_path, mode='rb') as f:
            while chunk := await f.read(8192):
                yield chunk
    
    return StreamingResponse(
        file_stream(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={file_path.name}"}
    )
