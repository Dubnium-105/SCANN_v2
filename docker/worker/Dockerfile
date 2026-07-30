ARG PYTORCH_BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

FROM ${PYTORCH_BASE_IMAGE} AS backbone-fetcher

ENV TORCH_HOME=/opt/torch-cache \
    TORCH_HUB=/opt/torch-cache/hub

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        torchvision==0.19.1 \
    && python -c "\
from torchvision.models import (\
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ViT_B_16_Weights,\
    resnet18, resnet34, resnet50, vit_b_16,\
);\
resnet18(weights=ResNet18_Weights.DEFAULT);\
resnet34(weights=ResNet34_Weights.DEFAULT);\
resnet50(weights=ResNet50_Weights.DEFAULT);\
vit_b_16(weights=ViT_B_16_Weights.DEFAULT);\
print('Worker backbone cache populated.');\
"

FROM ${PYTORCH_BASE_IMAGE} AS runner

ARG SCANN_IMAGE_REVISION=unknown
LABEL org.opencontainers.image.title="scann-native-worker" \
      org.opencontainers.image.source="SCANN_v2" \
      org.opencontainers.image.revision="${SCANN_IMAGE_REVISION}" \
      org.opencontainers.image.description="SCANN GPU training and prelabel worker"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch-cache \
    TORCH_HUB=/opt/torch-cache/hub

WORKDIR /opt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libfontconfig1 \
        libxkbcommon0 \
        libdbus-1-3 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-sync1 \
        libxcb-xfixes0 \
        libxcb-xkb1 \
        libxext6 \
        libxrender1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "PyQt5>=5.15,<6"

COPY scann_v2 /opt/scann_v2
RUN pip install --no-cache-dir -e /opt/scann_v2 \
    && python -c "import scann; from scann.native_annotation import prelabel_worker, training_job_worker; print('scann worker package OK')"

COPY --from=backbone-fetcher /opt/torch-cache /opt/torch-cache

RUN mkdir -p /var/log/scann

HEALTHCHECK NONE
WORKDIR /opt/scann_v2
CMD ["scann-training-worker", "--log-level=INFO"]
