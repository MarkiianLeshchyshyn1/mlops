# Builder stage: here we install dependencies once and then copy them into the final image.
FROM python:3.12-slim AS builder

# Disable `.pyc` files, enable immediate logs, and avoid pip cache to keep layers smaller.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# All next commands run inside `/app`.
WORKDIR /app

# Install system packages needed to build Python dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency list first to improve Docker layer caching.
COPY requirements.txt .
# Install Python packages into a separate folder that we can copy into runtime stage.
RUN pip install --upgrade pip && pip install --prefix=/install -r requirements.txt

# Runtime stage: starts from a clean slim image so the final container stays smaller.
FROM python:3.12-slim AS runtime

# Keep runtime predictable and lightweight.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/usr/local/bin:${PATH}"

# Set working directory for the application.
WORKDIR /app

# Install only minimal runtime system dependency used by DVC/Git workflows.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python dependencies from the builder stage.
COPY --from=builder /install /usr/local
# Copy project source code.
COPY src ./src
# Copy project configuration files.
COPY config ./config
# Copy DVC pipeline definition.
COPY dvc.yaml .
# Copy DVC lock file for reproducibility.
COPY dvc.lock .
# Copy DVC metadata/configuration.
COPY .dvc ./.dvc
# Copy project data used by the training pipeline.
COPY data ./data

# Default command: run model training inside the container.
CMD ["python", "src/stages/train.py", "data/prepared", "data/models"]
