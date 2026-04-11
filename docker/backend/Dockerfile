ARG PYTHON_BASE_IMAGE=python:3.11-slim
ARG DEBIAN_MIRROR=deb.debian.org
ARG DEBIAN_SECURITY_MIRROR=security.debian.org

FROM ${PYTHON_BASE_IMAGE} AS builder

ARG DEBIAN_MIRROR
ARG DEBIAN_SECURITY_MIRROR

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i "s|http://deb.debian.org/debian|http://${DEBIAN_MIRROR}/debian|g; s|http://security.debian.org/debian-security|http://${DEBIAN_SECURITY_MIRROR}/debian-security|g; s|https://deb.debian.org/debian|http://${DEBIAN_MIRROR}/debian|g; s|https://security.debian.org/debian-security|http://${DEBIAN_SECURITY_MIRROR}/debian-security|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i "s|http://deb.debian.org/debian|http://${DEBIAN_MIRROR}/debian|g; s|http://security.debian.org/debian-security|http://${DEBIAN_SECURITY_MIRROR}/debian-security|g; s|https://deb.debian.org/debian|http://${DEBIAN_MIRROR}/debian|g; s|https://security.debian.org/debian-security|http://${DEBIAN_SECURITY_MIRROR}/debian-security|g" /etc/apt/sources.list; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY docker/backend/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --prefix=/install -r /tmp/requirements.txt


ARG PYTHON_BASE_IMAGE=python:3.11-slim
ARG DEBIAN_MIRROR=deb.debian.org
ARG DEBIAN_SECURITY_MIRROR=security.debian.org

FROM ${PYTHON_BASE_IMAGE}

ARG DEBIAN_MIRROR
ARG DEBIAN_SECURITY_MIRROR

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:${PATH}" \
    PYTHONPATH="/install/lib/python3.11/site-packages:/app/scann_v2/src"

WORKDIR /app

RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i "s|http://deb.debian.org/debian|http://${DEBIAN_MIRROR}/debian|g; s|http://security.debian.org/debian-security|http://${DEBIAN_SECURITY_MIRROR}/debian-security|g; s|https://deb.debian.org/debian|http://${DEBIAN_MIRROR}/debian|g; s|https://security.debian.org/debian-security|http://${DEBIAN_SECURITY_MIRROR}/debian-security|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i "s|http://deb.debian.org/debian|http://${DEBIAN_MIRROR}/debian|g; s|http://security.debian.org/debian-security|http://${DEBIAN_SECURITY_MIRROR}/debian-security|g; s|https://deb.debian.org/debian|http://${DEBIAN_MIRROR}/debian|g; s|https://security.debian.org/debian-security|http://${DEBIAN_SECURITY_MIRROR}/debian-security|g" /etc/apt/sources.list; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /install
COPY scann_v2/src /app/scann_v2/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "scann.native_annotation.app:app", "--host", "0.0.0.0", "--port", "8000"]
