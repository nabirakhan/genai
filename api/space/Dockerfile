# Hugging Face Space (Docker SDK). Put this Dockerfile + README.md in the Space repo; it clones the
# public GitHub repo at build time so the Space never needs the code pushed twice.
# Space secrets: HF_MODEL_REPO (e.g. nabirakhan/dr-gan-demo), HF_TOKEN (if that model repo is private),
# CORS_ORIGINS (your Vercel URL), PUBLIC_MODE=1
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
ARG REPO=https://github.com/nabirakhan/genai
ARG REF=main
RUN useradd -m -u 1000 user
WORKDIR /app
RUN git clone --depth 1 --branch ${REF} --recurse-submodules --shallow-submodules ${REPO} /app \
 && chown -R user:user /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r api/requirements.txt
USER user
ENV DEMO_ASSETS=/home/user/assets PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 HOME=/home/user
RUN mkdir -p /home/user/assets
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
