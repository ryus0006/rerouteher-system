# ReRouteHer It1 backend image.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # never touch the Hugging Face Hub at runtime; the model is vendored below
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBEDDING_MODEL=models/all-MiniLM-L6-v2

WORKDIR /app

# ca-certificates + curl: fetch the parakeet model over HTTPS at build time. ffmpeg:
# E7 audio normalisation at runtime (ParakeetTranscriber shells out to it).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps first for layer caching.
# Install CPU-only torch up front so sentence-transformers does not pull the ~1GB
# NVIDIA CUDA wheels (useless on CPU); the rest then sees torch already satisfied.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# spaCy English model (fetched from spaCy's own release, not Hugging Face).
RUN python -m spacy download en_core_web_sm

# Vendored models (checked into the repo): all-MiniLM-L6-v2 embedder (~87MB) and the
# ms-marco-MiniLM-L6-v2 cross-encoder reranker (~87MB). Copied straight in, so the
# build performs no Hugging Face download and startup loads them from these paths.
COPY models ./models

# E7 Parakeet TDT 0.6B v2 model (~630MB, int8): downloaded and checksummed at build
# time rather than committed to the repo. English-only (faster/smaller than the v3
# multilingual build on the CPU-only host), loaded by ParakeetTranscriber at runtime
# from parakeet_model_dir. Int8 encoder/decoder files only -- the fp32 variants in the
# same upstream repo are not fetched.
RUN mkdir -p models/parakeet \
    && cd models/parakeet \
    && curl -fsSL -O https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/config.json \
    && curl -fsSL -O https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/vocab.txt \
    && curl -fsSL -O https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/nemo128.onnx \
    && curl -fsSL -O https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/encoder-model.int8.onnx \
    && curl -fsSL -O https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/decoder_joint-model.int8.onnx \
    && echo "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466  config.json" \
       | sha256sum -c - \
    && echo "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d  vocab.txt" \
       | sha256sum -c - \
    && echo "a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f  nemo128.onnx" \
       | sha256sum -c - \
    && echo "3e0581fda6ab843888b51e56d7ee78b6d5bc3237ec113af1f732d1d5286aa155  encoder-model.int8.onnx" \
       | sha256sum -c - \
    && echo "a449f49acd68979d418651dd2dcb737cc0f1bf0225e009e29ee326354edbf7d3  decoder_joint-model.int8.onnx" \
       | sha256sum -c -

# Vendored TF-IDF occupation classifier (~25MB). Baked in so Tier 1 works on hosts
# that do not mount the compose volumes (e.g. Coolify).
COPY ml ./ml

# App code (db/, tests/ excluded via .dockerignore where not needed at runtime).
COPY app ./app

# Run as a non-root user (matches workspace convention).
RUN useradd -u 3000 -m appuser && chown -R 3000:3000 /app
USER 3000

# Network interface (8080; 8000 is taken by Coolify on the host).
EXPOSE 8080

# --no-access-log: the RequestLoggingMiddleware logs method/path/status/duration itself.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
