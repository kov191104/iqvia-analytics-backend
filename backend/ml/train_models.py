"""
Model training script for IQVIA Campaign Analytics.

Usage:
    python -m backend.ml.train_models --pretrain
    python -m backend.ml.train_models --skip-startup-training
"""
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from backend.ml.segmentation import segmentation_model, CLUSTER_NAMES
from backend.ml.uplift import uplift_model
from backend.ml.recommender import recommender_model


DATA_DIR = Path("backend/data")
MODELS_DIR = Path("backend/models")


def generate_sample_data(n_doctors: int = 500, n_campaigns: int = 5) -> Dict[str, pd.DataFrame]:
    """Generate sample training data for development/testing."""
    np.random.seed(42)
    
    specialties = ["Cardiology", "Oncology", "Neurology", "General Practice", "Pediatrics"]
    provinces = ["Ontario", "Quebec", "British Columbia", "Alberta", "Manitoba"]
    
    doctors_data = []
    for i in range(n_doctors):
        doctors_data.append({
            "doctor_id": i + 1,
            "doctor_name": f"Dr. {chr(65 + i % 26)}. {'Smith' if i % 3 == 0 else 'Johnson' if i % 3 == 1 else 'Williams'}",
            "specialty": np.random.choice(specialties),
            "province": np.random.choice(provinces),
            "tenure": np.random.randint(1, 30),
            "avg_rx_6mo": np.random.uniform(10, 200),
            "avg_rx_12mo": np.random.uniform(20, 400),
            "rx_total_12mo": np.random.uniform(100, 2000),
        })
    
    doctors_df = pd.DataFrame(doctors_data)
    
    campaign_data = []
    for doc in doctors_data:
        for campaign_id in range(1, n_campaigns + 1):
            campaign_target = np.random.choice([0, 1], p=[0.3, 0.7])
            calls = np.random.randint(0, 10) if campaign_target else 0
            engagement = np.random.uniform(0, 1) if calls > 0 else 0
            
            pre_rx = doc["avg_rx_6mo"] * np.random.uniform(0.8, 1.2)
            uplift = np.random.uniform(-5, 25) if calls > 0 else np.random.uniform(-10, 5)
            post_rx = pre_rx * (1 + uplift / 100)
            
            campaign_data.append({
                "doctor_id": doc["doctor_id"],
                "doctor_name": doc["doctor_name"],
                "specialty": doc["specialty"],
                "province": doc["province"],
                "tenure": doc["tenure"],
                "campaign_id": campaign_id,
                "campaign_name": f"Campaign {campaign_id}",
                "product": f"Product {'ABC'[campaign_id % 3]}",
                "campaign_target": campaign_target,
                "calls_in_campaign": calls,
                "engagement_in_campaign": engagement,
                "pre_rx": round(pre_rx, 2),
                "post_rx": round(post_rx, 2),
                "rx_uplift_pct": round(uplift, 2),
                "avg_rx_6mo": doc["avg_rx_6mo"],
                "avg_rx_12mo": doc["avg_rx_12mo"],
                "rx_total_12mo": doc["rx_total_12mo"],
            })
    
    full_df = pd.DataFrame(campaign_data)
    
    train_idx = int(len(full_df) * 0.7)
    val_idx = int(len(full_df) * 0.85)
    
    train_df = full_df.iloc[:train_idx].copy()
    val_df = full_df.iloc[train_idx:val_idx].copy()
    test_df = full_df.iloc[val_idx:].copy()
    
    return {
        "train_val_full": pd.concat([train_df, val_df]),
        "train_doctors": doctors_df,
        "test": test_df,
        "full": full_df
    }


def load_data() -> Dict[str, pd.DataFrame]:
    """Load training data from Excel files or generate sample data."""
    data = {}
    
    train_val_path = DATA_DIR / "train_val_full.xlsx"
    doctors_path = DATA_DIR / "train_doctors.xlsx"
    test_path = DATA_DIR / "test.xlsx"
    
    if train_val_path.exists():
        data["train_val_full"] = pd.read_excel(train_val_path)
    
    if doctors_path.exists():
        data["train_doctors"] = pd.read_excel(doctors_path)
    
    if test_path.exists():
        data["test"] = pd.read_excel(test_path)
    
    if not data:
        print("No data files found. Generating sample data...")
        data = generate_sample_data()
        
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data["train_val_full"].to_excel(train_val_path, index=False)
        data["train_doctors"].to_excel(doctors_path, index=False)
        data["test"].to_excel(test_path, index=False)
        print(f"Sample data saved to {DATA_DIR}")
    
    return data


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for model training."""
    df = df.copy()
    
    required_cols = {
        "avg_rx_6mo": 0.0,
        "avg_rx_12mo": 0.0,
        "rx_total_12mo": 0.0,
        "calls_in_campaign": 0,
        "engagement_in_campaign": 0.0,
        "tenure": 0,
        "specialty": "Unknown",
        "province": "Unknown",
        "campaign_target": 0,
        "rx_uplift_pct": 0.0
    }
    
    for col, default in required_cols.items():
        if col not in df.columns:
            df[col] = default
    
    return df


def train_all_models(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Train all ML models and save metrics."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "training_timestamp": datetime.now().isoformat(),
        "models": {}
    }
    
    train_df = data.get("train_val_full", data.get("full", pd.DataFrame()))
    train_df = engineer_features(train_df)
    
    print("Training segmentation model...")
    seg_metrics = segmentation_model.train(train_df)
    metrics["models"]["segmentation"] = {
        "version": segmentation_model.version,
        "metrics": seg_metrics
    }
    print(f"  Segmentation: {seg_metrics}")
    
    print("Training uplift model...")
    uplift_metrics = uplift_model.train(train_df)
    metrics["models"]["uplift"] = {
        "version": uplift_model.version,
        "metrics": uplift_metrics
    }
    print(f"  Uplift: MAE={uplift_metrics['mae']:.3f}, R2={uplift_metrics['r2']:.3f}")
    
    print("Training recommender model...")
    products = train_df["product"].unique().tolist() if "product" in train_df.columns else None
    rec_metrics = recommender_model.train(train_df, products=products)
    metrics["models"]["recommender"] = {
        "version": recommender_model.version,
        "metrics": rec_metrics
    }
    print(f"  Recommender: {rec_metrics}")
    
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    return metrics


def load_all_models() -> Dict[str, bool]:
    """Load all trained models."""
    return {
        "segmentation": segmentation_model.load(),
        "uplift": uplift_model.load(),
        "recommender": recommender_model.load()
    }


def main():
    parser = argparse.ArgumentParser(description="Train IQVIA Campaign Analytics ML models")
    parser.add_argument("--pretrain", action="store_true", help="Run training locally")
    parser.add_argument("--skip-startup-training", action="store_true", 
                       help="Skip training on startup (for production)")
    parser.add_argument("--generate-data", action="store_true",
                       help="Generate sample data only")
    
    args = parser.parse_args()
    
    if args.generate_data:
        print("Generating sample data...")
        data = generate_sample_data()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data["train_val_full"].to_excel(DATA_DIR / "train_val_full.xlsx", index=False)
        data["train_doctors"].to_excel(DATA_DIR / "train_doctors.xlsx", index=False)
        data["test"].to_excel(DATA_DIR / "test.xlsx", index=False)
        print(f"Sample data saved to {DATA_DIR}")
        return
    
    if args.skip_startup_training:
        print("Loading pre-trained models...")
        load_status = load_all_models()
        print(f"Model load status: {load_status}")
        return
    
    if args.pretrain:
        print("Starting model training...")
        data = load_data()
        metrics = train_all_models(data)
        print("\nTraining complete!")
        print(f"Models saved to {MODELS_DIR}")
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()
