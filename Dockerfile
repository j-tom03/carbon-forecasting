# 1. Base Image
FROM python:3.11-slim

# 2. Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Writing metadata
LABEL maintainer="Carbon Forecast Team"
LABEL version="2.0"
LABEL description="Inference service for Carbon Intensity TFT model"

# 3. Set Working Directory
WORKDIR /app

# 4. Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Assets
COPY src/ src/
COPY configs/ configs/
COPY models/tft_prod/ models/tft_prod/
COPY data/training/val.pt data/training/val.pt

# 6. Expose the Port
EXPOSE 8000

# API Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health') or exit(1)"

# 7. Define the Entrypoint
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]