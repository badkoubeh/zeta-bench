# syntax=docker/dockerfile:1.7-labs
#
# Multi-stage build for zeta-bench.
#
# Base: nvidia/cuda runtime so PyTorch's CUDA path lights up on GPU hosts.
# On CPU-only hosts (no NVIDIA Container Runtime) the container still runs —
# PyTorch transparently falls back to CPU. One image, both deployment targets.

ARG PYTHON_VERSION=3.12
ARG CUDA_BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ---------- builder stage ----------
FROM ${CUDA_BASE} AS builder

ARG PYTHON_VERSION

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_CACHE_DIR=/root/.cache/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# uv binary from the official upstream image (pinned to a major release).
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /build

# Resolve interpreter + install deps from the lockfile, before copying the
# source. Keeps Docker's layer cache effective when only the source changes.
COPY pyproject.toml requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python ${PYTHON_VERSION} /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-deps --requirement requirements.lock

# Copy the project source and install zeta-bench itself (deps already resolved).
COPY . /build
RUN VIRTUAL_ENV=/opt/venv uv pip install --no-deps /build

# ---------- runtime stage ----------
FROM ${CUDA_BASE} AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv \
    PYTHONPATH=/workspace \
    HYDRA_FULL_ERROR=1
# WANDB_MODE is intentionally not pinned here: the config resolves it at runtime
# (online when WANDB_API_KEY is supplied, offline otherwise). Set WANDB_MODE
# explicitly at `docker run` time to override.

# Runtime-only system deps. ffmpeg is required for matplotlib MP4 writer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Carry the resolved venv and project source forward from the builder.
# /opt/uv/python must be copied alongside /opt/venv because the venv's python
# binary is a symlink into the uv-managed interpreter tree; without it the
# symlink is dangling and `python` cannot be exec'd in the runtime stage.
COPY --from=builder /opt/uv/python /opt/uv/python
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build /workspace

# Non-root user for safer container execution. Host bind mounts must allow
# UID 1000 to write — Docker Desktop handles this automatically; on bare
# Linux hosts, ensure your `results/` directory is group-writable.
RUN groupadd --gid 1000 zeta && \
    useradd --uid 1000 --gid zeta --create-home --shell /bin/bash zeta && \
    chown -R zeta:zeta /workspace
USER zeta

# Default command shows the train.py CLI; override at `docker run` time.
ENTRYPOINT ["python"]
CMD ["experiments/train.py", "--help"]

# ---------- sagemaker stage ----------
# Purpose-built image for Amazon SageMaker Training Jobs. Build explicitly with
# `docker build --target sagemaker ...`; the default target is `runtime`, so the
# hardened local/compose image above is untouched.
#
# SageMaker BYOC starts the container as `docker run <image> train` and mounts
# /opt/ml/* as root, so this stage runs as root and swaps in an entrypoint that
# maps the `train` invocation onto the project's Hydra entrypoint.
FROM runtime AS sagemaker

USER root

COPY docker/sm-entrypoint.sh /usr/local/bin/sm-entrypoint.sh
RUN chmod +x /usr/local/bin/sm-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/sm-entrypoint.sh"]
CMD ["train"]
