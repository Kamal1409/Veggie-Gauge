# CPU-only image for the sorting station: FastAPI on :8000 (default) or the Gradio demo on :7860.
#
#   docker build -t veggie-gauge .
#   docker run -p 8000:8000 veggie-gauge
#   docker run -p 7860:7860 veggie-gauge python -m app.gradio_app --host 0.0.0.0
#
# Needs the rig's calibration (data/calibration.json) and the trained models (models/*.joblib):
# bake them in, or mount them: -v ./models:/app/models -v ./data/calibration.json:/app/data/calibration.json

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY train/ train/
COPY eval/ eval/
COPY tools/ tools/
COPY models/ models/
COPY data/calibration.json data/calibration.json
COPY pyproject.toml .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
