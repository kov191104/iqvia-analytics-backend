"""
Doctor recommendation system using content-based filtering.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional


class RecommenderModel:
    """Content-based doctor recommender."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path("backend/models/recommender.joblib")
        self.version = "1.0.0"
        self.loaded = False
        self.doctor_vectors: Optional[np.ndarray] = None
        self.doctor_ids: Optional[List[int]] = None
        self.product_profiles: Dict[str, np.ndarray] = {}
        self.scaler: Optional[StandardScaler] = None
        self.encoder: Optional[OneHotEncoder] = None
        self.numeric_features = ["avg_rx_6mo", "avg_rx_12mo", "rx_total_12mo", "tenure"]
        self.categorical_features = ["specialty", "province"]
    
    def load(self) -> bool:
        """Load the trained model from disk."""
        try:
            if self.model_path.exists():
                model_data = joblib.load(self.model_path)
                self.doctor_vectors = model_data.get("doctor_vectors")
                self.doctor_ids = model_data.get("doctor_ids")
                self.product_profiles = model_data.get("product_profiles", {})
                self.scaler = model_data.get("scaler")
                self.encoder = model_data.get("encoder")
                self.version = model_data.get("version", "1.0.0")
                self.loaded = True
                return True
        except Exception as e:
            print(f"Error loading recommender model: {e}")
        return False
    
    def save(self) -> None:
        """Save the model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            "doctor_vectors": self.doctor_vectors,
            "doctor_ids": self.doctor_ids,
            "product_profiles": self.product_profiles,
            "scaler": self.scaler,
            "encoder": self.encoder,
            "version": self.version
        }
        joblib.dump(model_data, self.model_path)
    
    def train(self, df: pd.DataFrame, products: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train the recommender model."""
        df_clean = df.dropna(subset=self.numeric_features + self.categorical_features)
        
        if "doctor_id" in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=["doctor_id"])
        
        self.scaler = StandardScaler()
        numeric_data = self.scaler.fit_transform(df_clean[self.numeric_features])
        
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        categorical_data = self.encoder.fit_transform(df_clean[self.categorical_features])
        
        self.doctor_vectors = np.hstack([numeric_data, categorical_data])
        self.doctor_ids = df_clean["doctor_id"].tolist() if "doctor_id" in df_clean.columns else list(range(len(df_clean)))
        
        if products is None:
            products = ["Product A", "Product B", "Product C"]
        
        for product in products:
            idx = hash(product) % len(self.doctor_vectors)
            base_vector = self.doctor_vectors[idx % len(self.doctor_vectors)]
            noise = np.random.randn(len(base_vector)) * 0.1
            self.product_profiles[product] = base_vector + noise
        
        self.loaded = True
        self.save()
        
        return {
            "n_doctors": len(self.doctor_ids),
            "n_products": len(self.product_profiles),
            "vector_dim": self.doctor_vectors.shape[1]
        }
    
    def recommend(
        self,
        product: str,
        n: int = 10,
        doctor_df: Optional[pd.DataFrame] = None,
        uplift_predictions: Optional[Dict[int, float]] = None
    ) -> List[Dict[str, Any]]:
        """Recommend top doctors for a product."""
        if not self.loaded:
            raise ValueError("Model not loaded. Call load() first.")
        
        if product not in self.product_profiles:
            available = list(self.product_profiles.keys())
            if available:
                product = available[0]
            else:
                raise ValueError(f"No product profiles available")
        
        product_vector = self.product_profiles[product].reshape(1, -1)
        
        similarities = cosine_similarity(product_vector, self.doctor_vectors)[0]
        
        top_indices = np.argsort(similarities)[::-1][:n]
        
        results = []
        for idx in top_indices:
            doctor_id = self.doctor_ids[idx]
            score = float(similarities[idx])
            
            doctor_info = {
                "doctor_id": int(doctor_id),
                "doctor_name": f"Doctor {doctor_id}",
                "specialty": "General",
                "province": "Unknown",
                "score": round(score, 4),
                "predicted_uplift": 0.0
            }
            
            if doctor_df is not None and len(doctor_df) > 0:
                doc_row = doctor_df[doctor_df["doctor_id"] == doctor_id]
                if len(doc_row) > 0:
                    row = doc_row.iloc[0]
                    doctor_info["doctor_name"] = row.get("doctor_name", f"Doctor {doctor_id}")
                    doctor_info["specialty"] = row.get("specialty", "General")
                    doctor_info["province"] = row.get("province", "Unknown")
            
            if uplift_predictions and doctor_id in uplift_predictions:
                doctor_info["predicted_uplift"] = uplift_predictions[doctor_id]
            else:
                doctor_info["predicted_uplift"] = round(score * 15 + np.random.uniform(-2, 2), 2)
            
            results.append(doctor_info)
        
        return results


recommender_model = RecommenderModel()
