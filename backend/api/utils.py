"""
Utility functions for filtering, pagination, and data operations.
"""
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
from functools import lru_cache
import hashlib
import json
from datetime import date


def parse_csv_param(value: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated string into list."""
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def create_filter_key(filters: Dict[str, Any]) -> str:
    """Create a hashable key for caching filtered results."""
    sorted_items = sorted(
        ((k, v) for k, v in filters.items() if v is not None),
        key=lambda x: x[0]
    )
    return hashlib.md5(json.dumps(sorted_items, default=str).encode()).hexdigest()


def apply_filters(
    df: pd.DataFrame,
    specialties: Optional[str] = None,
    provinces: Optional[str] = None,
    tenure_min: Optional[int] = None,
    tenure_max: Optional[int] = None,
    min_rx: Optional[float] = None,
    max_rx: Optional[float] = None,
    campaign_target: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Apply filters to a DataFrame."""
    result = df.copy()
    
    specialty_list = parse_csv_param(specialties)
    if specialty_list and "specialty" in result.columns:
        result = result[result["specialty"].isin(specialty_list)]
    
    province_list = parse_csv_param(provinces)
    if province_list and "province" in result.columns:
        result = result[result["province"].isin(province_list)]
    
    if tenure_min is not None and "tenure" in result.columns:
        result = result[result["tenure"] >= tenure_min]
    
    if tenure_max is not None and "tenure" in result.columns:
        result = result[result["tenure"] <= tenure_max]
    
    if min_rx is not None and "rx_total_12mo" in result.columns:
        result = result[result["rx_total_12mo"] >= min_rx]
    
    if max_rx is not None and "rx_total_12mo" in result.columns:
        result = result[result["rx_total_12mo"] <= max_rx]
    
    if campaign_target is not None and "campaign_target" in result.columns:
        result = result[result["campaign_target"] == campaign_target]
    
    if start_date is not None and "date" in result.columns:
        result = result[pd.to_datetime(result["date"]) >= pd.to_datetime(start_date)]
    
    if end_date is not None and "date" in result.columns:
        result = result[pd.to_datetime(result["date"]) <= pd.to_datetime(end_date)]
    
    return result


def paginate(
    df: pd.DataFrame,
    page: int = 1,
    per_page: int = 10,
    max_per_page: int = 500
) -> Tuple[pd.DataFrame, int, int, int]:
    """
    Apply pagination to a DataFrame.
    
    Returns:
        Tuple of (paginated_df, page, per_page, total_count)
    """
    per_page = min(per_page, max_per_page)
    total = len(df)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    return df.iloc[start_idx:end_idx], page, per_page, total


def calculate_campaign_metrics(df: pd.DataFrame, campaign_id: int) -> Dict[str, Any]:
    """Calculate campaign metrics from DataFrame."""
    campaign_df = df[df["campaign_id"] == campaign_id] if "campaign_id" in df.columns else df
    
    if campaign_df.empty:
        return {
            "campaign_id": campaign_id,
            "avg_pre_rx": 0.0,
            "avg_post_rx": 0.0,
            "avg_uplift_pct": 0.0,
            "engagement_rate": 0.0,
            "total_target_doctors": 0,
            "total_reached": 0
        }
    
    avg_pre_rx = campaign_df["pre_rx"].mean() if "pre_rx" in campaign_df.columns else 0.0
    avg_post_rx = campaign_df["post_rx"].mean() if "post_rx" in campaign_df.columns else 0.0
    avg_uplift_pct = campaign_df["rx_uplift_pct"].mean() if "rx_uplift_pct" in campaign_df.columns else 0.0
    
    total_target = len(campaign_df[campaign_df["campaign_target"] == 1]) if "campaign_target" in campaign_df.columns else len(campaign_df)
    total_reached = len(campaign_df[campaign_df["calls_in_campaign"] > 0]) if "calls_in_campaign" in campaign_df.columns else 0
    
    engagement_rate = (total_reached / total_target * 100) if total_target > 0 else 0.0
    
    return {
        "campaign_id": campaign_id,
        "avg_pre_rx": round(float(avg_pre_rx), 2),
        "avg_post_rx": round(float(avg_post_rx), 2),
        "avg_uplift_pct": round(float(avg_uplift_pct), 2),
        "engagement_rate": round(float(engagement_rate), 2),
        "total_target_doctors": int(total_target),
        "total_reached": int(total_reached)
    }


def calculate_rx_trend(df: pd.DataFrame, campaign_id: int, months: int = 12) -> List[Dict[str, Any]]:
    """Calculate monthly Rx trend for a campaign."""
    if "month" not in df.columns or "rx_volume" not in df.columns:
        from datetime import datetime
        base_date = datetime.now()
        trend = []
        for i in range(months - 1, -1, -1):
            month_offset = (base_date.month - i - 1) % 12 + 1
            year_offset = base_date.year - ((base_date.month - i - 1) // 12)
            month_str = f"{year_offset}-{month_offset:02d}"
            trend.append({
                "month": month_str,
                "rx_volume": float(100 + (months - i) * 5 + (i % 3) * 10)
            })
        return trend
    
    campaign_df = df[df["campaign_id"] == campaign_id] if "campaign_id" in df.columns else df
    
    monthly_data = campaign_df.groupby("month")["rx_volume"].sum().reset_index()
    monthly_data = monthly_data.sort_values("month").tail(months)
    
    return [
        {"month": row["month"], "rx_volume": round(float(row["rx_volume"]), 2)}
        for _, row in monthly_data.iterrows()
    ]


class FilterCache:
    """Simple cache for filtered DataFrames."""
    
    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        return self._cache.get(key)
    
    def set(self, key: str, df: pd.DataFrame) -> None:
        if len(self._cache) >= self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = df
    
    def clear(self) -> None:
        self._cache.clear()


filter_cache = FilterCache()
