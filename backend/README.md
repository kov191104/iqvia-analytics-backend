# IQVIA Campaign Analytics API

A robust FastAPI backend for campaign analytics, doctor segmentation, uplift prediction, and recommendations.

## Features

- **Campaign Analytics**: Aggregate metrics, Rx trends, and top responders
- **ML Models**:
  - KMeans segmentation for doctor clustering
  - Random Forest for uplift prediction with confidence intervals
  - Content-based recommender for doctor targeting
- **Data Export**: Streaming CSV exports with background job support for large datasets
- **Filtering & Pagination**: Comprehensive query parameters for all endpoints

## Project Structure

```
backend/
├── api/
│   ├── main.py          # FastAPI application entry point
│   ├── routes.py        # Campaign data endpoints
│   ├── models_router.py # ML model prediction endpoints
│   ├── schemas.py       # Pydantic models
│   └── utils.py         # Utility functions
├── ml/
│   ├── train_models.py  # Model training script
│   ├── segmentation.py  # KMeans segmentation model
│   ├── uplift.py        # Random Forest uplift model
│   └── recommender.py   # Content-based recommender
├── data/                # Training data (Excel files)
├── models/              # Trained model artifacts (joblib)
├── exports/             # Temporary CSV exports
├── requirements.txt
└── README.md
```

## Quick Start

### Development

1. Install dependencies:
```bash
pip install -r backend/requirements.txt
```

2. Generate sample data and train models:
```bash
python -m backend.ml.train_models --pretrain
```

3. Run the API:
```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 5000 --reload
```

4. Access the API documentation at `http://localhost:5000/docs`

### Production

1. Pretrain models in CI/CD:
```bash
python -m backend.ml.train_models --pretrain
```

2. Deploy with trained models and set environment variables:
```bash
SKIP_STARTUP_TRAINING=true
ALLOWED_ORIGINS=https://your-frontend.com
```

3. Start the server:
```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT
```

## API Endpoints

### Campaign Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/campaigns` | List all campaigns |
| GET | `/api/v1/campaign/{id}/metrics` | Get campaign metrics |
| GET | `/api/v1/campaign/{id}/rx-trend` | Get Rx volume trend |
| GET | `/api/v1/campaign/{id}/top-responders` | Get top responders |

### ML Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/models/status` | Get model status |
| POST | `/api/v1/models/segmentation/predict` | Predict doctor segments |
| POST | `/api/v1/models/uplift/predict` | Predict uplift percentage |
| POST | `/api/v1/models/recommend/doctors` | Get doctor recommendations |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/exports/{job_id}` | Check export status |

## Query Parameters

All endpoints support these filters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `specialties` | string | Comma-separated list of specialties |
| `provinces` | string | Comma-separated list of provinces |
| `tenure_min` | int | Minimum tenure (years) |
| `tenure_max` | int | Maximum tenure (years) |
| `min_rx` | float | Minimum Rx volume |
| `max_rx` | float | Maximum Rx volume |
| `campaign_target` | int | Campaign target (0 or 1) |
| `page` | int | Page number (default: 1) |
| `per_page` | int | Items per page (default: 10, max: 500) |

## Response Format

All responses follow this structure:

```json
{
  "status": "ok",
  "meta": {
    "page": 1,
    "per_page": 10,
    "total": 123
  },
  "data": [...]
}
```

Error responses:

```json
{
  "status": "error",
  "detail": "Error message"
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 5000 |
| `ALLOWED_ORIGINS` | CORS origins (comma-separated) | * |
| `SKIP_STARTUP_TRAINING` | Skip model training on startup | false |

## Model Training

### Pretrain locally:
```bash
python -m backend.ml.train_models --pretrain
```

### Generate sample data only:
```bash
python -m backend.ml.train_models --generate-data
```

### Skip training (production):
```bash
python -m backend.ml.train_models --skip-startup-training
```

## Data Files

Place your Excel files in `backend/data/`:

- `train_val_full.xlsx` - Training and validation data
- `train_doctors.xlsx` - Doctor master data
- `test.xlsx` - Test data

If no files are found, sample data will be generated automatically.

## Example Requests

### List campaigns with filters
```bash
curl "http://localhost:5000/api/v1/campaigns?specialties=Cardiology,Oncology&page=1&per_page=10"
```

### Predict doctor segments
```bash
curl -X POST "http://localhost:5000/api/v1/models/segmentation/predict" \
  -H "Content-Type: application/json" \
  -d '{"doctor_ids": [1, 2, 3, 4, 5]}'
```

### Predict uplift
```bash
curl -X POST "http://localhost:5000/api/v1/models/uplift/predict" \
  -H "Content-Type: application/json" \
  -d '{"campaign_id": 1, "doctor_ids": [1, 2, 3]}'
```

### Get doctor recommendations
```bash
curl -X POST "http://localhost:5000/api/v1/models/recommend/doctors" \
  -H "Content-Type: application/json" \
  -d '{"product": "Product A", "n": 10}'
```

### Export to CSV
```bash
curl "http://localhost:5000/api/v1/campaign/1/top-responders?export=csv" -o responders.csv
```

## Deployment

### Railway / Heroku

Create a `Procfile`:
```
web: uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r backend/requirements.txt
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "5000"]
```

## Rate Limiting (Recommendation)

For production, consider adding FastAPI-Limiter:

```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

# In your startup
await FastAPILimiter.init(redis_connection)

# On endpoints
@app.get("/api/v1/campaigns", dependencies=[Depends(RateLimiter(times=100, seconds=60))])
```

## License

Proprietary - IQVIA
