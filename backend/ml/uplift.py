"""
Uplift prediction using Random Forest.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


NUMERIC_FEATURES = [
    "avg_rx_6mo", "avg_rx_12mo", "rx_total_12mo", 
    "calls_in_campaign", "engagement_in_campaign", "tenure"
]
CATEGORICAL_FEATURES = ["specialty", "province"]
TARGET = "rx_uplift_pct"


class UpliftModel:
    """Uplift prediction model wrapper."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.pipeline: Optional[Pipeline] = None
        self.model_path = model_path or Path("backend/models/uplift_rf.joblib")
        self.version = "1.0.0"
        self.loaded = False
        self.feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        self.metrics: Dict[str, float] = {}
    
    def load(self) -> bool:
        """Load the trained model from disk."""
        try:
            if self.model_path.exists():
                model_data = joblib.load(self.model_path)
                self.pipeline = model_data.get("pipeline")
                self.version = model_data.get("version", "1.0.0")
                self.feature_columns = model_data.get("feature_columns", self.feature_columns)
                self.metrics = model_data.get("metrics", {})
                self.loaded = True
                return True
        except Exception as e:
            print(f"Error loading uplift model: {e}")
        return False
    
    def save(self) -> None:
        """Save the trained model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            "pipeline": self.pipeline,
            "version": self.version,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics
        }
        joblib.dump(model_data, self.model_path)
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the uplift prediction model."""
        required_cols = self.feature_columns + [TARGET]
        df_clean = df.dropna(subset=required_cols)
        
        X = df_clean[self.feature_columns]
        y = df_clean[TARGET]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
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
            ("regressor", RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                max_depth=10
            ))
        ])
        
        self.pipeline.fit(X_train, y_train)
        
        y_pred = self.pipeline.predict(X_val)
        
        self.metrics = {
            "mae": float(mean_absolute_error(y_val, y_pred)),
            "mse": float(mean_squared_error(y_val, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
            "r2": float(r2_score(y_val, y_pred))
        }
        
        self.loaded = True
        self.save()
        
        return {
            "n_train": len(X_train),
            "n_val": len(X_val),
            **self.metrics
        }
    
    def predict_with_confidence(
        self, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Predict uplift with confidence intervals.
        Uses variance from tree predictions for confidence bounds.
        """
        if not self.loaded or self.pipeline is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        for col in self.feature_columns:
            if col not in df.columns:
                if col in NUMERIC_FEATURES:
                    df[col] = 0.0
                else:
                    df[col] = "Unknown"
        
        X = df[self.feature_columns]
        
        mean_pred = self.pipeline.predict(X)
        
        rf = self.pipeline.named_steps["regressor"]
        preprocessor = self.pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(X)
        
        tree_predictions = np.array([
            tree.predict(X_transformed) for tree in rf.estimators_
        ])
        
        std_pred = np.std(tree_predictions, axis=0)
        
        results = []
        for idx, (_, row) in enumerate(df.iterrows()):
            pred = float(mean_pred[idx])
            std = float(std_pred[idx])
            results.append({
                "doctor_id": int(row.get("doctor_id", idx)),
                "predicted_uplift_pct": round(pred, 2),
                "confidence_interval": {
                    "lower": round(pred - 1.96 * std, 2),
                    "upper": round(pred + 1.96 * std, 2)
                }
            })
        
        return results


uplift_model = UpliftModel()
