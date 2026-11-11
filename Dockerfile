# Production Multi-Stage Dockerfile for OmniAgent
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code and frontend
COPY . .

# Generate sample demo images if not already present
RUN python src/generate_samples.py

# Expose ports: 8501 (Streamlit UI), 8000 (FastAPI Server)
EXPOSE 8501 8000

# Default command: Launch interactive UI
CMD ["python", "run.py", "--ui"]

