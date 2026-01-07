# RoofVision Measurement Service

Production-grade roof measurement backend using photogrammetry and computer vision.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RoofVision Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐ │
│  │  Mobile App  │────▶│   FastAPI    │────▶│      Celery Worker       │ │
│  │  (Expo/RN)   │     │   (REST)     │     │   (GPU Processing)       │ │
│  └──────────────┘     └──────────────┘     └──────────────────────────┘ │
│         │                    │                         │                 │
│         │                    ▼                         ▼                 │
│         │             ┌──────────────┐     ┌──────────────────────────┐ │
│         │             │    Redis     │     │        COLMAP            │ │
│         │             │   (Queue)    │     │  (3D Reconstruction)     │ │
│         │             └──────────────┘     └──────────────────────────┘ │
│         │                    │                         │                 │
│         ▼                    ▼                         ▼                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐ │
│  │    MinIO     │     │  PostgreSQL  │     │        Open3D            │ │
│  │  (Storage)   │     │  (Database)  │     │   (Plane Extraction)     │ │
│  └──────────────┘     └──────────────┘     └──────────────────────────┘ │
│                                                        │                 │
│                                                        ▼                 │
│                                            ┌──────────────────────────┐ │
│                                            │    Google Solar API      │ │
│                                            │     (Validation)         │ │
│                                            └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Processing Pipeline

1. **Photo Upload** → S3/MinIO storage
2. **Photo Validation** → Blur, resolution, angle detection
3. **3D Reconstruction** → COLMAP SfM/MVS
4. **Plane Extraction** → Open3D RANSAC
5. **Measurement** → Area, pitch, facets calculation
6. **Validation** → Google Solar API comparison
7. **Confidence Scoring** → Multi-factor analysis
8. **Report Generation** → PDF with measurements

## Quick Start

### Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA GPU (RTX 3080+ recommended)
- Google Solar API key (optional, for validation)

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit with your values
nano .env
```

Required environment variables:
```
GOOGLE_SOLAR_API_KEY=your_google_api_key
POSTGRES_PASSWORD=secure_password
MINIO_ROOT_PASSWORD=secure_password
```

### Start Services

```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f api worker

# Check health
curl http://localhost:8000/health
```

### API Usage

#### Create a Job
```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "address": "123 Main St, Austin, TX 78701",
    "latitude": 30.2672,
    "longitude": -97.7431
  }'
```

#### Upload Photos
```bash
JOB_ID="abc123"

# Upload each photo
for photo in photos/*.jpg; do
  curl -X POST "http://localhost:8000/jobs/${JOB_ID}/photos" \
    -F "file=@${photo}"
done
```

#### Start Processing
```bash
curl -X POST "http://localhost:8000/jobs/${JOB_ID}/start"
```

#### Check Status
```bash
curl "http://localhost:8000/jobs/${JOB_ID}"
```

#### Get Results
```bash
curl "http://localhost:8000/jobs/${JOB_ID}/result"
```

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/jobs` | Create new measurement job |
| POST | `/jobs/{id}/photos` | Upload photo for job |
| POST | `/jobs/{id}/start` | Start processing |
| GET | `/jobs/{id}` | Get job status |
| GET | `/jobs/{id}/result` | Get measurement result |
| GET | `/jobs/{id}/report` | Download PDF report |
| GET | `/jobs` | List all jobs |
| GET | `/health` | Health check |

### Response Models

#### MeasurementResult
```json
{
  "job_id": "abc123",
  "address": "123 Main St, Austin, TX 78701",
  "total_area_sqft": 2450.75,
  "total_area_m2": 227.67,
  "roofing_squares": 24.51,
  "ground_footprint_sqft": 1850.0,
  "dominant_pitch_12": 6.0,
  "dominant_pitch_degrees": 26.57,
  "facets": [...],
  "facet_count": 4,
  "photos_used": 8,
  "photos_rejected": 0,
  "confidence_score": 87.5,
  "confidence_level": "high",
  "confidence_reasons": ["Full 360° coverage", "Validated with Google"],
  "google_validated": true,
  "google_variance_percent": 2.3,
  "processing_time_seconds": 245.5
}
```

## Configuration

### COLMAP Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `COLMAP_BIN` | `colmap` | Path to COLMAP binary |
| `COLMAP_GPU_INDEX` | `0` | GPU device index |

### RANSAC Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Distance threshold | 0.02m | Inlier distance for plane fitting |
| Iterations | 2000 | RANSAC iterations |
| Min points | 500 | Minimum points per facet |

### Confidence Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Photo quality | 20% | Image sharpness, resolution |
| Angle coverage | 25% | 8-angle completeness |
| Reconstruction | 25% | COLMAP stability |
| Facet consistency | 15% | Pitch/area consistency |
| Google validation | 15% | API comparison |

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.override.yml
services:
  worker:
    deploy:
      replicas: 4  # One per GPU
```

### GPU Requirements

| Workload | GPU | Memory | Jobs/Hour |
|----------|-----|--------|-----------|
| Light | RTX 3060 | 12GB | ~10 |
| Medium | RTX 4080 | 16GB | ~20 |
| Heavy | A100 | 40GB | ~50 |

### Cost Estimation

At 10,000 jobs/month on AWS:
- g5.xlarge Spot: ~$0.05/job
- Storage: ~$0.02/job
- **Total: ~$0.07/job**

## Monitoring

### Flower Dashboard
Access Celery monitoring at `http://localhost:5555`

### Metrics
- Job success rate
- Processing time distribution
- Confidence score distribution
- Queue depth

## Troubleshooting

### Common Issues

**COLMAP fails to reconstruct**
- Ensure 70%+ photo overlap
- Check for motion blur
- Verify camera angles cover all sides

**Low confidence score**
- Add missing angles
- Improve photo quality
- Check for obstructions

**Google validation fails**
- Verify coordinates are accurate
- Check API key permissions
- Some locations lack coverage

### Logs

```bash
# API logs
docker-compose logs api

# Worker logs
docker-compose logs worker

# All logs
docker-compose logs -f
```

## Development

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn main:app --reload

# Run worker
celery -A worker worker --loglevel=debug
```

### Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=services --cov-report=html
```

## License

Proprietary - RoofVision Inc.
