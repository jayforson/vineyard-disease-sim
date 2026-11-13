FROM osrf/ros:jazzy-desktop@sha256:2f520187e84304fffb60d19c9a7d8e2e79d563a6e579a090427e5994eb1b8c45
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    QT_QPA_PLATFORM=offscreen \
    LIBGL_ALWAYS_SOFTWARE=1 \
    GZ_SIM_RESOURCE_PATH=/workspace/simulation/generated \
    HF_HOME=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-gz ros-jazzy-cv-bridge python3-venv mesa-utils \
    libgl1-mesa-dri libegl1 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir 'pip>=24' && \
    /opt/venv/bin/pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu && \
    /opt/venv/bin/pip install --no-cache-dir 'transformers==4.57.6' 'pillow==11.3.0' 'numpy==1.26.4' 'pytest==8.4.2'
ENV PATH=/opt/venv/bin:$PATH HF_HOME=/workspace/data/cache/hf_home
WORKDIR /workspace
COPY scripts/container-entrypoint.sh /usr/local/bin/vineyard-entrypoint
ENTRYPOINT ["bash", "/usr/local/bin/vineyard-entrypoint"]
CMD ["bash"]
