# PlantPath AI backend
# Python 3.11 for PyTorch CPU wheels.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Application code
COPY backend/app /app/app
COPY data /app/data
COPY models /app/models

# Default config (override via environment). Production = real inference:
# the backend refuses to serve predictions unless the checkpoint passes its
# verification gate (models/classifier_audit.json).
ENV MODEL_PATH=/app/models/efficientnet_v2_s_best.pth \
    MODEL_CLASSES_PATH=/app/data/model_classes.json \
    MODEL_AUDIT_PATH=/app/models/classifier_audit.json \
    CLASS_NAMES_PATH=/app/data/class_names.json \
    MODEL_METRICS_PATH=/app/data/model_metrics.json \
    DISEASES_PATH=/app/data/diseases.json \
    DB_PATH=/app/app/database/plantpath.db \
    DEMO_MODE=false \
    MAX_UPLOAD_SIZE_MB=10 \
    CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
