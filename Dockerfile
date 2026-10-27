FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies (cached until pyproject/lock change)
COPY app/pyproject.toml app/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Models (cached separate layer; does not depend on code)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /app/models && cd /app/models \
    && K=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0 \
    && Z=https://github.com/opencv/opencv_zoo/raw/main/models \
    && curl -fsSL --retry 3 -o kokoro-v1.0.int8.onnx $K/kokoro-v1.0.int8.onnx \
    && curl -fsSL --retry 3 -o voices-v1.0.bin $K/voices-v1.0.bin \
    && curl -fsSL --retry 3 -o yunet.onnx $Z/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    && curl -fsSL --retry 3 -o sface.onnx $Z/face_recognition_sface/face_recognition_sface_2021dec.onnx \
    && for f in kokoro-v1.0.int8.onnx voices-v1.0.bin sface.onnx; do \
         [ "$(stat -c %s $f)" -gt 1000000 ] || { echo "$f too small"; exit 1; }; done \
    && [ "$(stat -c %s yunet.onnx)" -gt 100000 ] || { echo "yunet.onnx too small"; exit 1; }

# App code
COPY app/main.py ./
COPY app/static ./static

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
