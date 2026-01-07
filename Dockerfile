# RoofVision Measurement Service
# Simplified build for Windows Docker Desktop

# Use standard Python image (COLMAP will be installed via apt)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including COLMAP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    colmap \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV COLMAP_BIN=/usr/bin/colmap

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (API server)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
