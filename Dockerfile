# Heart Disease Risk API - production image
#   docker build -t heart-api .
#   docker run -p 8000:8000 heart-api
FROM python:3.11-slim

LABEL org.opencontainers.image.title="heart-disease-api" \
      org.opencontainers.image.description="FastAPI service for heart disease risk prediction" \
      org.opencontainers.image.source="https://github.com/prashob-narendran/heart-disease-mlops"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/model.joblib \
    METADATA_PATH=/app/models/metadata.json \
    PORT=8000

WORKDIR /app

# 1. Dependencies first, so this layer is cached when only code changes
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 2. Application code and the trained model (preprocessing + estimator)
COPY src/ src/
COPY api/ api/
COPY models/model.joblib models/metadata.json models/

# 3. Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Container reports unhealthy if /health stops returning 200
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)" || exit 1

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS:-1} --no-access-log"]
