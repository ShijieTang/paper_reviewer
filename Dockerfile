# Image to run doc_preprocess.py (marker-pdf 2.0 + torch) on an Intel Mac.
#
# Why a container: marker-pdf 2.0.0 pins torch>=2.7.0, and torch >=2.3 ships
# macOS wheels only for Apple Silicon — there is no Intel-x86_64 macOS wheel.
# Linux x86_64 has CPU torch 2.7.0 wheels, so we run the conversion here.
#
# Base is Python 3.12 (not 3.13): the PyTorch CPU index has torch 2.7.0+cpu for
# cp312 but only up to 2.6.0 for cp313, and marker-pdf 2.0.0 requires >=2.7.0.

FROM python:3.12-slim

# System libraries needed by marker-pdf / surya / pdftext / torch-CPU at runtime.
#   libgl1, libglib2.0-0 — OpenCV/Pillow backends used by surya
#   libgomp1            — OpenMP runtime for torch CPU
#   git, curl           — huggingface_hub model downloads / fetches
#   build-essential     — in case any dep needs to compile a C extension
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first (satisfies marker-pdf 2.0's `torch>=2.7.0,<3`).
# The +cpu local version satisfies `>=2.7.0` per PEP 440 and is ~10x smaller
# than the default CUDA wheel from PyPI, so the image builds fast and stays lean.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.7.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code: the preprocessing module and the parallel runner.
COPY doc_preprocess.py ./
COPY scripts/run_doc_preprocess_parallel.py ./scripts/

# Keep marker/surya model cache inside the container volume (persist via a
# named volume if you want to avoid re-downloading models on every run).
ENV HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    TORCH_DEVICE=cpu

# Default command: convert every PDF in data/openreview_pdf -> data/openreview_md
# in parallel. Override args to change inputs/workers, e.g.
#   docker run <img> data/openreview_pdf data/openreview_md --workers 2 --limit 3
ENTRYPOINT ["python", "scripts/run_doc_preprocess_parallel.py"]
CMD ["data/openreview_pdf", "data/openreview_md"]
