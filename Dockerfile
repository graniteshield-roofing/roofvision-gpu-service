# RoofVision Measurement Service
# Multi-stage build for production deployment

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt


# Stage 2: COLMAP installation
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as colmap-builder

# Install COLMAP build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and build COLMAP
RUN git clone https://github.com/colmap/colmap.git /colmap && \
    cd /colmap && \
    git checkout 3.8 && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" && \
    ninja && \
    ninja install


# Stage 3: Production image
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libboost-program-options1.74.0 \
    libboost-filesystem1.74.0 \
    libboost-graph1.74.0 \
    libboost-system1.74.0 \
    libfreeimage3 \
    libmetis5 \
    libgoogle-glog0v5 \
    libsqlite3-0 \
    libglew2.2 \
    libqt5core5a \
    libqt5opengl5 \
    libcgal-dev \
    libceres2 \
    && rm -rf /var/lib/apt/lists/*

# Copy COLMAP from builder
COPY --from=colmap-builder /usr/local/bin/colmap /usr/local/bin/colmap
COPY --from=colmap-builder /usr/local/lib/colmap /usr/local/lib/colmap
COPY --from=colmap-builder /usr/local/share/colmap /usr/local/share/colmap

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV COLMAP_BIN=/usr/local/bin/colmap
ENV COLMAP_GPU_INDEX=0

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (API server)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
