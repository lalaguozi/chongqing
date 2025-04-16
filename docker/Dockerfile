# Stage 1: Builder - Installs build dependencies and Python packages
# Use a PyTorch image that might contain build tools, or install them
# Using the same base initially for consistency
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS builder

# Set environment variables (same as final stage)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1 \
    DEBIAN_FRONTEND=noninteractive # Avoid prompts during apt installs

# Install build essentials and minimal runtime deps needed for building wheels/package installs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # Build tools
    git gcc g++ gnupg libpython3-dev \
    # Runtime deps needed for building some packages or runtime
    wget curl htop zip unzip libgl1 libglib2.0-0 libusb-1.0-0 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Security updates (can be combined with above, but separate is also fine)
# RUN apt-get update && apt-get upgrade --no-install-recommends -y openssl tar && rm -rf /var/lib/apt/lists/*
# Note: Running upgrade here might update base packages beyond what the final stage expects.
# It might be safer to run the upgrade only in the final stage or accept the base image's state.

# Install uv
RUN pip install uv

# Create app directory
WORKDIR /opt/app

# Copy only files needed for dependency installation first
COPY pyproject.toml setup.py README.md ./
# If requirements files exist and are used, copy them too:
# COPY requirements.txt ./

# Install core dependencies and export dependencies explicitly
# Note: Includes packages previously auto-installed by 'yolo export'
# Check ultralytics pyproject.toml [export-dependencies] for the exact list
RUN uv pip install --system \
    # Core requirements (from setup.py/pyproject.toml install_requires)
    # These will be installed by -e . below, but listing key ones can be clearer
    "numpy==1.23.5" # Pinned version from original Dockerfile
    "torch>=1.8.0" # Already in base image, but ensures compatibility
    "torchvision>=0.9.0" # Already in base image
    "opencv-python>=4.6.0" \
    "pillow>=10.0.0" \
    "pyyaml>=5.3.1" \
    "requests>=2.23.0" \
    "scipy>=1.4.1" \
    "tqdm>=4.64.0" \
    "matplotlib>=3.3" \
    "pandas>=1.1.4" \
    "seaborn>=0.11.0" \
    # Export dependencies
    "ultralytics-thop>=0.2.5" # For profiling
    "onnx>=1.12.0" # ONNX export
    "onnxruntime-gpu>=1.16.0" # ONNX execution (use -gpu version)
    "onnxsim>=0.4.33" # ONNX simplification
    "scikit-learn>=1.3.1" # For NMS plugins in TensorRT
    "tensorrt-cu12>=8.6.0" # TensorRT (matches CUDA 12)
    "tensorflow>=2.13.0" # TensorFlow/TFLite export (consider tensorflow-cpu if GPU not needed for export itself)
    # "tflite-runtime" # Alternative for TFLite inference only (smaller than full TF) - install if needed
    "coremltools>=7.0" # CoreML export
    "paddlepaddle>=2.6.0" # PaddlePaddle export
    "paddle2onnx>=1.0.6" # PaddlePaddle export helper
    "paddleslim>=2.6.0" # PaddlePaddle export helper
    "intel-openvino>=2024.0" # OpenVINO export
    "onnx-graphsurgeon>=0.3.26" # Dependency for TensorRT export
    # NCNN - Requires manual build steps usually, check ultralytics docs if prebuilt wheels aren't sufficient
    # EdgeTPU - Requires specific libraries, often platform-dependent. The export command might handle this.
    # Extras & Others from original file
    "albumentations>=1.4.6" \
    "comet_ml>=3.39.0" \
    "pycocotools>=2.0.6"

# Copy the rest of the application code
COPY . .

# Install the ultralytics package itself in editable mode
# --no-deps because we installed them explicitly above
RUN uv pip install --system --no-deps -e ".[export]"


# -----------------------------------------------------------
# Stage 2: Final runtime image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1 \
    DEBIAN_FRONTEND=noninteractive

# Install only necessary runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git wget curl htop zip unzip libgl1 libglib2.0-0 libusb-1.0-0 libsm6 openssl tar \
    # Add lib Gomp if needed (PyTorch usually includes it)
    # libgomp1
    && apt-get upgrade --no-install-recommends -y openssl tar \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python*/dist-packages/ /usr/local/lib/python*/dist-packages/
COPY --from=builder /usr/local/bin /usr/local/bin/

# Copy the application code (needed for editable install -e)
WORKDIR /ultralytics
COPY --from=builder /opt/app /ultralytics

# Downloads fonts to user config dir
RUN mkdir -p /root/.config/Ultralytics
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Download the model (consider making this optional or an entrypoint argument)
RUN mkdir -p tmp
ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt tmp/yolo11n.pt

# Run exports to ensure environment is functional (optional, adds to image size)
# These might still trigger minor downloads/compilations if not fully covered by explicit installs
# Note: Running these might create caches or files you want to clean up
RUN echo "Running export checks..."
# Edge TPU export requires specific libs usually installed on the host or device
# The command might fail gracefully if libs aren't present, or install them if possible.
RUN yolo export model=tmp/yolo11n.pt format=edgetpu imgsz=32 || echo "EdgeTPU export requires additional setup/libraries."
# RUN yolo export model=tmp/yolo11n.pt format=edgetpu imgsz=32 # Second run if needed
RUN yolo export model=tmp/yolo11n.pt format=ncnn imgsz=32
RUN yolo export model=tmp/yolo11n.pt format=onnx opset=13 imgsz=32 # Example ONNX export
RUN yolo export model=tmp/yolo11n.pt format=openvino imgsz=32 # Example OpenVINO export

# Clean up downloaded model and any export caches if desired
RUN rm -rf tmp /root/.config/Ultralytics/persistent_cache.json /root/.cache

# Set default command (optional)
# CMD ["yolo", "detect", "predict", "model=yolov8n.pt", "source='https://ultralytics.com/images/bus.jpg'"]
CMD ["bash"]

# --- .dockerignore file ---
# Create a file named .dockerignore in the same directory as the Dockerfile
# with the following content:
# ```
# .git
# .github
# .vscode
# *.pyc
# __pycache__/
# *.so
# *.egg-info/
# build/
# dist/
# docs/
# examples/ # Unless needed in the image
# tests/    # Unless needed in the image
# datasets/ # Usually mounted as volumes
# runs/     # Output directory
# .env
# *.md      # If not needed (like README outside root)
# Dockerfile
# .dockerignore
# # Add any other files/directories not needed in the final image
# ```