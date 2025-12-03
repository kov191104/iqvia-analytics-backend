"""
Doctor segmentation using KMeans clustering.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional


CLUSTER_NAMES = {
    0: "High Value Engaged",
    1: "Growth Potential",
    2: "Low Engagement",
    3: "New Prospects"
}

NUMERIC_FEATURES = ["avg_rx_6mo", "avg_rx_12mo", "rx_total_12mo", "tenure"]
CATEGORICAL_FEATURES = ["specialty", "province"]


class SegmentationModel:
    """Doctor segmentation model wrapper."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.pipeline: Optional[Pipeline] = None
        self.model_path = model_path or Path("backend/models/segmentation_kmeans.joblib")
        self.version = "1.0.0"
        self.loaded = False
        self.feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    
    def load(self) -> bool:
        """Load the trained model from disk."""
        try:
            if self.model_path.exists():
                model_data = joblib.load(self.model_path)
                self.pipeline = model_data.get("pipeline")
                self.version = model_data.get("version", "1.0.0")
                self.loaded = True
                return True
        except Exception as e:
            print(f"Error loading segmentation model: {e}")
        return False
    
    def save(self) -> None:
        """Save the trained model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            "pipeline": self.pipeline,
            "version": self.version,
            "feature_columns": self.feature_columns
        }
        joblib.dump(model_data, self.model_path)
    
    def train(self, df: pd.DataFrame, n_clusters: int = 4) -> Dict[str, Any]:
        """Train the segmentation model."""
        df_clean = df.dropna(subset=self.feature_columns)
        
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_FEATURES),
                ("cat", categorical_transformer, CATEGORICAL_FEATURES)
            ]
        )
        
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
        ])
        
        X = df_clean[self.feature_columns]
        self.pipeline.fit(X)
        
        labels = self.pipeline.predict(X)
        inertia = self.pipeline.named_steps["kmeans"].inertia_
        
        self.loaded = True
        self.save()
        
        return {
            "n_clusters": n_clusters,
            "n_samples": len(df_clean),
            "inertia": float(inertia),
            "cluster_distribution": pd.Series(labels).value_counts().to_dict()
        }
    
    def predict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict cluster assignments for doctors."""
        if not self.loaded or self.pipeline is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        for col in self.feature_columns:
            if col not in df.columns:
                if col in NUMERIC_FEATURES:
                    df[col] = 0.0
                else:
                    df[col] = "Unknown"
        
        X = df[self.feature_columns]
        clusters = self.pipeline.predict(X)
        
        results = []
        for idx, (_, row) in enumerate(df.iterrows()):
            cluster = int(clusters[idx])
            results.append({
                "doctor_id": int(row.get("doctor_id", idx)),
                "cluster": cluster,
                "cluster_name": CLUSTER_NAMES.get(cluster, f"Cluster {cluster}")
            })
        
        return results


segmentation_model = SegmentationModel()
