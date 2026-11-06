# Aegis-V2X backend container -- built from the repo root so the image
# can also pick up ../dashboard/dist if it exists (see app/main.py's
# AEGIS_SKIP_DASHBOARD_MOUNT). Run `docker compose --profile backend --profile monitoring up`
# from the repo root to bring this up with Postgres, Prometheus and Grafana.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AEGIS_SKIP_DASHBOARD_MOUNT=1

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
