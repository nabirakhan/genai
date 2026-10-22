# Local/CI image built from the repo root:  docker build -f api/Dockerfile -t dr-gan-api .
# The HF Space uses api/space/Dockerfile (clones the GitHub repo at build time).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r api/requirements.txt
COPY src ./src
COPY api ./api
COPY third_party/stylegan3 ./third_party/stylegan3
COPY third_party/patches ./third_party/patches
COPY scripts/apply_patches.py ./scripts/apply_patches.py
ENV DEMO_ASSETS=/data/assets PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
RUN mkdir -p /data/assets
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
